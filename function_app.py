import azure.functions as func
import logging
import os
import joblib
from azure.storage.blob import BlobServiceClient

# --- Configuration ---
# La chaîne de connexion est récupérée depuis les paramètres de l'application
connect_str = os.getenv('AZURE_CONNECTION_STRING')
container_name = "reco-data" # Le nom de votre conteneur de blobs
model_blob_name = "models/hybrid_recommender_pipeline.pkl" # Le chemin complet vers votre modèle
local_model_path = "/tmp/model.pkl"

# --- Chargement du modèle ---
# Le modèle est téléchargé depuis le Blob Storage une seule fois au démarrage de la fonction (démarrage à froid)
model = None
try:
    if connect_str:
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_blob_name)

        logging.info(f"Téléchargement du modèle depuis {container_name}/{model_blob_name}...")
        with open(file=local_model_path, mode="wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        model = joblib.load(local_model_path)
        logging.info("Modèle chargé avec succès.")
    else:
        logging.warning("AZURE_CONNECTION_STRING n'est pas définie. Le modèle ne peut pas être chargé.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")

# --- Définition de l'application de fonction ---
app = func.FunctionApp()

@app.route(route="recommend", methods=["GET"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Requête de recommandation reçue.')

    if not model:
        return func.HttpResponse("Erreur: Le modèle n'est pas chargé.", status_code=503)

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse("Veuillez fournir un 'user_id' dans les paramètres de la requête.", status_code=400)

    try:
        # NOTE: Adaptez cette partie à la méthode de prédiction de votre modèle
        # predictions = model.predict(user_id)
        # Pour l'exemple, nous retournons des données factices
        predictions = {"user_id": user_id, "items": ["item_A", "item_B", "item_C"]}
        return func.HttpResponse(body=str(predictions), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        return func.HttpResponse("Erreur interne du serveur lors de la prédiction.", status_code=500)