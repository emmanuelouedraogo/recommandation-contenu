import azure.functions as func
import logging
import asyncio
import os
import joblib
import json
import pandas as pd
from io import BytesIO
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential  # type: ignore
from .predict import Recommender  # Importe la classe Recommender

# --- Configuration ---
# La chaîne de connexion est récupérée depuis les paramètres de l'application
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "reco-data")

# Noms des blobs pour le modèle et les données
model_blob_name = os.getenv("AZURE_STORAGE_MODEL_BLOB", "models/hybrid_recommender_pipeline.pkl")
articles_blob_name = "articles_metadata.csv"
clicks_blob_name = "clicks_sample.csv"

# Variable globale pour stocker le modèle et un verrou pour gérer le chargement concurrent
recommender_instance = None  # type: Recommender | None
model_load_lock = asyncio.Lock()


async def load_model_from_blob(storage_account_name: str, container_name: str):
    """
    Télécharge et charge le modèle depuis Azure Blob Storage.
    Retourne le modèle chargé ou None en cas d'erreur.
    """
    try:
        if not storage_account_name:
            logging.critical("AZURE_STORAGE_ACCOUNT_NAME n'est pas définie. Le service ne peut pas démarrer.")
            return None

        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=DefaultAzureCredential()
        )

        def download_blob_to_memory(blob_name):
            logging.info(f"Téléchargement de {blob_name}...")
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            downloader = blob_client.download_blob(timeout=300)
            return downloader.readall()

        # Télécharger tous les artefacts nécessaires en parallèle
        model_data, articles_data, clicks_data = await asyncio.gather(
            asyncio.to_thread(download_blob_to_memory, model_blob_name),
            asyncio.to_thread(download_blob_to_memory, articles_blob_name),
            asyncio.to_thread(download_blob_to_memory, clicks_blob_name),
        )

        # Charger les objets en mémoire
        logging.info("Chargement des artefacts en mémoire...")
        pipeline_obj = joblib.load(BytesIO(model_data))
        articles_df = pd.read_csv(BytesIO(articles_data))
        clicks_df = pd.read_csv(BytesIO(clicks_data), nrows=100000)

        # Instancier la classe Recommender avec les objets chargés
        recommender = Recommender(pipeline=pipeline_obj, articles_df=articles_df, clicks_df=clicks_df)
        logging.info("Instance de Recommender créée avec succès.")
        return recommender

    except ResourceNotFoundError:
        logging.critical(f"Un artefact requis n'a pas été trouvé dans le conteneur '{container_name}'.")
        return None
    except Exception as e:
        logging.critical(
            "Erreur critique lors du chargement des artefacts, l'application ne pourra pas servir de recommandations : "
            f"{e}",
            exc_info=True,
        )
        return None


# --- Définition de l'application de fonction ---
# Pour la production, il est recommandé d'utiliser func.AuthLevel.FUNCTION pour sécuriser l'endpoint.
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="/health", methods=[func.HttpMethod.GET])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint de vérification de l'état.
    Vérifie si le modèle de recommandation est chargé et prêt à servir les requêtes.
    """
    global recommender_instance
    logging.info("Health check request received.")

    if recommender_instance is not None:
        return func.HttpResponse("Service is healthy and model is loaded.", status_code=200, mimetype="text/plain")
    else:
        return func.HttpResponse(
            "Service is unhealthy (model not loaded).", status_code=503, mimetype="text/plain"  # Service Unavailable
        )


@app.route(route="/recommend", methods=[func.HttpMethod.GET])
async def recommend(req: func.HttpRequest) -> func.HttpResponse:
    global recommender_instance
    logging.info("Requête de recommandation reçue.")

    # Chargement paresseux (lazy loading) du modèle au premier appel pour optimiser le démarrage à froid.
    # Le verrou (lock) garantit que même si plusieurs requêtes arrivent en même temps,
    # le modèle ne sera chargé qu'une seule fois.
    if recommender_instance is None:
        async with model_load_lock:
            # Double vérification pour éviter les chargements redondants
            if recommender_instance is None:
                logging.info("L'instance du Recommender n'est pas chargée. Tentative de chargement...")
                recommender_instance = await load_model_from_blob(STORAGE_ACCOUNT_NAME, container_name)

    # Si le chargement a échoué, le modèle sera toujours None
    if recommender_instance is None:
        logging.error("Le chargement du modèle a échoué. Impossible de servir la requête.")
        return func.HttpResponse(
            "Erreur: Le service de recommandation n'est pas disponible (échec du chargement du modèle).",
            status_code=503,
        )

    user_id = req.params.get("user_id")
    if not user_id:
        return func.HttpResponse("Le paramètre 'user_id' est manquant.", status_code=400)

    try:
        user_id_int = int(user_id)
        if user_id_int <= 0:
            raise ValueError("ID utilisateur doit être positif")
    except ValueError:
        return func.HttpResponse("Le paramètre 'user_id' doit être un entier.", status_code=400)

    try:
        # Utiliser l'instance du Recommender pour générer les recommandations
        recommendations = recommender_instance.generate_recommendations(user_id=user_id_int, top_n=10)
        if not recommendations:
            return func.HttpResponse("[]", mimetype="application/json", status_code=200)

        # Conversion directe en JSON sans passer par un DataFrame pour plus d'efficacité.
        result_json = json.dumps(recommendations)
        return func.HttpResponse(body=result_json, mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(
            f"Erreur lors de la génération des recommandations pour user_id {user_id_int}: {e}", exc_info=True
        )
        return func.HttpResponse("Erreur interne du serveur lors de la prédiction.", status_code=500)
