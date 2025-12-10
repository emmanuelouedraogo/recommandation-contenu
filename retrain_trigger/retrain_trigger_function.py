# /home/epikaizo/Desktop/recommandation-contenu/retrain_trigger_function.py

import azure.functions as func
import logging
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from io import StringIO, BytesIO
from datetime import datetime, timezone

# Importer les scripts de modélisation
from reco_model_script import train_and_save_model

# --- Configuration ---
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = "reco-data"
CLICKS_BLOB_NAME = "clicks_sample.csv"
METADATA_BLOB_NAME = "articles_metadata.csv"
EMBEDDINGS_BLOB_NAME = "articles_embeddings.pickle"
MODEL_BLOB_NAME = "models/hybrid_recommender_pipeline.pkl"
STATE_BLOB_NAME = "training_state.json"  # Fichier pour suivre l'état

STATUS_BLOB_NAME = "status/retraining_status.json"  # Fichier pour le statut en direct
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"  # Fichier pour l'historique
RETRAIN_THRESHOLD_INCREMENT = 1000  # Seuil configurable pour le ré-entraînement

# Définition de l'application de fonction
retrain_app = func.FunctionApp()

if not STORAGE_ACCOUNT_NAME:
    logging.error("AZURE_STORAGE_ACCOUNT_NAME is not set. Unable to proceed.")
    exit(1)


def get_training_state(blob_service_client: BlobServiceClient):
    """Récupère le dernier état d'entraînement (ex: le nombre de clics du dernier run)."""
    state_blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=STATE_BLOB_NAME
    )  # type: ignore
    try:
        state_data = state_blob_client.download_blob().readall()
        return pd.read_json(StringIO(state_data.decode("utf-8")), typ="series")  # type: ignore
    except Exception:
        # Si le fichier n'existe pas, on part de 0
        return pd.Series({"last_training_click_count": 0})


def save_training_state(blob_service_client, new_count):
    """Sauvegarde le nouvel état d'entraînement."""
    state = pd.Series({"last_training_click_count": new_count})
    state_blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=STATE_BLOB_NAME
    )  # type: ignore
    state_blob_client.upload_blob(state.to_json(), overwrite=True)


def update_retraining_status(blob_service_client, status: str, details: dict = None):
    """Met à jour le statut du réentraînement dans un fichier JSON dédié."""  # type: ignore
    status_data = {
        "status": status,
        "last_update": datetime.now(timezone.utc).isoformat(),
        **(details or {}),
    }  # type: ignore
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=STATUS_BLOB_NAME)  # type: ignore
    blob_client.upload_blob(pd.Series(status_data).to_json(), overwrite=True)


def log_training_run(blob_service_client, metrics, click_count):
    """Ajoute une entrée à l'historique d'entraînement."""
    # Utiliser un Append Blob pour une journalisation efficace
    append_blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=TRAINING_LOG_BLOB_NAME
    )

    # S'assurer que le blob existe et est bien un Append Blob
    try:
        append_blob_client.get_blob_properties()
    except Exception:  # Si le blob n'existe pas, le créer
        append_blob_client.create_blob()
        # Ajouter l'en-tête CSV au nouveau fichier
        header = "timestamp,click_count," + ",".join(metrics.keys()) + "\n"
        append_blob_client.append_block(header.encode("utf-8"))

    new_log_entry = f"{pd.Timestamp.now()},{click_count}," + ",".join(map(str, metrics.values())) + "\n"
    append_blob_client.append_block(new_log_entry.encode("utf-8"))


@retrain_app.schedule(schedule="0 * * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False)
def timer_trigger_retrain(myTimer: func.TimerRequest) -> None:
    logging.info("Déclencheur de ré-entraînement activé.")
    if not STORAGE_ACCOUNT_NAME:
        logging.error("AZURE_STORAGE_ACCOUNT_NAME n'est pas configurée.")
        return
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential()
    )

    # 1. Obtenir l'état actuel
    training_state = get_training_state(blob_service_client)

    last_training_count = training_state.get("last_training_click_count", 0)
    # 2. Compter le nombre actuel d'interactions
    clicks_parquet_blob_name = CLICKS_BLOB_NAME.replace(".csv", ".parquet")
    try:
        clicks_blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=clicks_parquet_blob_name
        )
        clicks_data = clicks_blob_client.download_blob(timeout=120).readall()
        clicks_df = pd.read_parquet(BytesIO(clicks_data))
        current_click_count = len(clicks_df)
    except Exception as e:
        logging.error(f"Impossible de lire le fichier des clics : {e}")
        return

    logging.info(
        f"Nombre de clics actuel : {current_click_count}. " f"Dernier entraînement à : {last_training_count} clics."
    )
    # 3. Vérifier si le seuil est atteint
    # On vérifie si le nombre de clics a dépassé le prochain multiple du seuil
    next_threshold = (last_training_count // RETRAIN_THRESHOLD_INCREMENT + 1) * RETRAIN_THRESHOLD_INCREMENT
    if current_click_count >= next_threshold:
        logging.info(f"Seuil de {next_threshold} clics dépassé. Démarrage du ré-entraînement.")

        try:
            update_retraining_status(blob_service_client, "in_progress")
            # 4. Exécuter le script d'entraînement. Il sauvegarde le modèle directement.
            metrics = train_and_save_model(
                # connect_str=CONNECT_STR, # Removed, using Managed Identity
                container_name=CONTAINER_NAME,
                clicks_blob=CLICKS_BLOB_NAME,
                articles_blob=METADATA_BLOB_NAME,
                embeddings_blob=EMBEDDINGS_BLOB_NAME,
                model_output_blob=MODEL_BLOB_NAME,
            )
            # The train_and_save_model function now handles saving the model to blob storage directly.
            # So, we don't need to re-upload it here.
            logging.info("Modèle entraîné et sauvegardé avec succès sur Azure.")

            # 6. Mettre à jour l'état d'entraînement
            save_training_state(blob_service_client, current_click_count)

            # 7. Enregistrer les métriques et le statut de cet entraînement
            try:
                log_training_run(blob_service_client, metrics, current_click_count)
                update_retraining_status(blob_service_client, "idle", {"last_run_metrics": metrics})  # type: ignore
            except Exception as log_e:
                logging.error(f"Erreur lors de la sauvegarde du log d'entraînement : {log_e}")
                # Même si le logging échoue, on met à jour le statut principal
                update_retraining_status(
                    blob_service_client,
                    "succeeded_with_log_error",
                    {"last_run_metrics": metrics, "log_error": str(log_e)},
                )

            logging.info(f"État d'entraînement mis à jour à {current_click_count} clics.")
        except Exception as e:
            update_retraining_status(blob_service_client, "failed", {"error": str(e)})
            logging.error(f"Une erreur est survenue pendant le ré-entraînement : {e}")
    else:
        logging.info("Le seuil de ré-entraînement n'est pas encore atteint.")
