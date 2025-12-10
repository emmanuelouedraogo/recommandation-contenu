import azure.functions as func
import logging
import asyncio
import os
import joblib
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

# --- Configuration ---
# La chaîne de connexion est récupérée depuis les paramètres de l'application
connect_str = os.getenv("AZURE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "reco-data")
model_blob_name = os.getenv("AZURE_STORAGE_MODEL_BLOB", "models/hybrid_recommender_pipeline.pkl")
local_model_path = "/tmp/hybrid_recommender_pipeline.pkl"

# Variable globale pour stocker le modèle et un verrou pour gérer le chargement concurrent
model = None
model_load_lock = asyncio.Lock()


def load_model_from_blob():
    """
    Télécharge et charge le modèle depuis Azure Blob Storage.
    Retourne le modèle chargé ou None en cas d'erreur.
    """
    try:
        if not connect_str:
            logging.warning("AZURE_CONNECTION_STRING n'est pas définie. Le modèle ne peut pas être chargé.")
            return None

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_blob_name)

        logging.info(f"Début du téléchargement du modèle depuis {container_name}/{model_blob_name}...")
        with open(local_model_path, "wb") as download_file:
            download_file.write(blob_client.download_blob(timeout=300).readall())
        logging.info("Téléchargement terminé.")

        loaded_model = joblib.load(local_model_path)
        logging.info("Modèle chargé avec succès.")
        return loaded_model
    except ResourceNotFoundError:
        logging.critical(
            f"Le modèle blob '{model_blob_name}' n'a pas été trouvé dans le conteneur '{container_name}'."
        )
        return None
    except Exception as e:
        logging.critical(
            "Erreur critique lors du chargement du modèle, l'application ne pourra pas servir de recommandations : "
            f"{e}", exc_info=True
        )
        return None


# --- Définition de l'application de fonction ---
app = func.FunctionApp()


@app.route(route="recommend", methods=[func.HttpMethod.GET])
async def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Requête de recommandation reçue.')
    
    # Chargement paresseux (lazy loading) du modèle au premier appel pour optimiser le démarrage à froid.
    # Le verrou (lock) garantit que même si plusieurs requêtes arrivent en même temps,
    # le modèle ne sera chargé qu'une seule fois.
    if model is None:
        async with model_load_lock:
            if model is None:
                logging.info("Le modèle n'est pas chargé. Tentative de chargement...")
                model = load_model_from_blob()

    # Si le chargement a échoué, le modèle sera toujours None
    if model is None:
        logging.error(
            "Le chargement du modèle a échoué. Impossible de servir la requête."
        )
        return func.HttpResponse(
            "Erreur: Le service de recommandation n'est pas disponible (échec du chargement du modèle).",
            status_code=503
        )

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse("Le paramètre 'user_id' est manquant.", status_code=400)

    try:
        user_id_int = int(user_id)
        if user_id_int <= 0:
            raise ValueError("ID utilisateur doit être positif")
    except ValueError:
        return func.HttpResponse("Le paramètre 'user_id' doit être un entier.", status_code=400)

    try:
        # Utiliser l'ID entier
        recommendations_df = model.recommend_items(uid=user_id_int, topn=10)
        if recommendations_df is None or recommendations_df.empty:
            return func.HttpResponse("[]", mimetype="application/json", status_code=200)

        result_json = recommendations_df.to_json(orient="records")
        return func.HttpResponse(body=result_json, mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(
            f"Erreur lors de la génération des recommandations pour user_id {user_id_int}: {e}",
            exc_info=True
        )
        return func.HttpResponse("Erreur interne du serveur lors de la prédiction.", status_code=500)