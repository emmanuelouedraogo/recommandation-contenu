# /home/epikaizo/Desktop/recommandation-contenu/retrain_trigger_function.py

import azure.functions as func
import logging
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from io import StringIO, BytesIO
from azure.core.exceptions import ResourceNotFoundError  # type: ignore
from datetime import datetime, timezone

# Importer les scripts de modélisation
from reco_model_script import train_and_save_model  # noqa: F401

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
INTERACTIONS_LOG_BLOB_NAME = "interactions/new_interactions_log.csv"  # Log des nouvelles interactions

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
    append_blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=TRAINING_LOG_BLOB_NAME)

    # S'assurer que le blob existe et est bien un Append Blob
    try:
        append_blob_client.get_blob_properties()
    except Exception:  # Si le blob n'existe pas, le créer
        append_blob_client.create_blob()
        # Ajouter l'en-tête CSV au nouveau fichier
        header = "timestamp,click_count," + ",".join(metrics.keys()) + "\n"
        append_blob_client.append_block(header.encode("utf-8"))

    metrics_values_str = ",".join(map(str, metrics.values()))
    new_log_entry = f"{pd.Timestamp.now()},{click_count},{metrics_values_str}\n"
    append_blob_client.append_block(new_log_entry.encode("utf-8"))


def _process_new_interactions(blob_service_client: BlobServiceClient) -> int:
    """
    Traite le log des nouvelles interactions, le fusionne avec le fichier de clics principal,
    et archive le log traité. Retourne le nombre total de clics après la fusion.
    """
    interactions_log_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=INTERACTIONS_LOG_BLOB_NAME
    )
    clicks_parquet_blob_name = CLICKS_BLOB_NAME.replace(".csv", ".parquet")
    clicks_parquet_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=clicks_parquet_blob_name
    )

    try:
        # 1. Charger les clics existants
        downloader = clicks_parquet_client.download_blob(timeout=120)
        existing_clicks_df = pd.read_parquet(BytesIO(downloader.readall()))
        current_click_count = len(existing_clicks_df)

        # 2. Charger les nouvelles interactions
        new_interactions_data = interactions_log_client.download_blob().readall()
        new_interactions_df = pd.read_csv(StringIO(new_interactions_data.decode("utf-8")))

        if not new_interactions_df.empty:
            logging.info(f"Traitement de {len(new_interactions_df)} nouvelles interactions.")
            # 3. Fusionner les données
            updated_clicks_df = pd.concat([existing_clicks_df, new_interactions_df], ignore_index=True)

            # 4. Sauvegarder le DataFrame mis à jour
            output_parquet = BytesIO()
            updated_clicks_df.to_parquet(output_parquet, index=False)
            clicks_parquet_client.upload_blob(output_parquet.getvalue(), overwrite=True)
            logging.info(f"Mise à jour de {clicks_parquet_blob_name} avec les nouvelles interactions.")

            # 5. Archiver le log traité au lieu de le supprimer
            archive_blob_name = f"archive/interactions/{datetime.now(timezone.utc).isoformat()}.csv"
            archive_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=archive_blob_name)
            archive_client.start_copy_from_url(interactions_log_client.url)
            interactions_log_client.delete_blob()
            logging.info(f"Log des interactions archivé dans {archive_blob_name} et vidé.")
            return len(updated_clicks_df)
    except ResourceNotFoundError:
        logging.info("Aucun nouveau log d'interactions à traiter.")
    except (pd.errors.ParserError, ValueError) as e:
        logging.error(f"Erreur de parsing des nouvelles interactions, le fichier est peut-être corrompu: {e}", exc_info=True)
        # Ne pas continuer si les données sont corrompues pour éviter d'écraser les clics existants
        raise  # Propage l'erreur pour arrêter le traitement
    except Exception as e:
        logging.error(f"Erreur inattendue lors du traitement des nouvelles interactions : {e}", exc_info=True)
        raise  # Propage l'erreur

    # Retourne le compte existant si aucune nouvelle interaction n'a été traitée
    return current_click_count


@retrain_app.schedule(schedule="0 * * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False)
def timer_trigger_retrain(myTimer: func.TimerRequest) -> None:
    logging.info("Déclencheur de ré-entraînement activé.")
    if not STORAGE_ACCOUNT_NAME:
        logging.error("AZURE_STORAGE_ACCOUNT_NAME n'est pas configurée.")
        return
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
        # Augmenter les timeouts pour les opérations sur les blobs volumineux
        connection_timeout=20,
        read_timeout=120,
    )

    # 1. Obtenir l'état actuel
    training_state = get_training_state(blob_service_client)

    last_training_count = training_state.get("last_training_click_count", 0)
    # 2. Compter le nombre actuel d'interactions
    try:
        # --- NOUVEAU : Traiter les nouvelles interactions avant de vérifier le seuil ---
        current_click_count = _process_new_interactions(blob_service_client)
    except Exception as e:
        logging.error(f"Échec du traitement des interactions. Arrêt du déclencheur. Erreur: {e}")
        update_retraining_status(blob_service_client, "failed", {"error": f"Interaction processing failed: {e}"})
        return

    log_msg = f"Nombre de clics actuel : {current_click_count}. Dernier entraînement à : {last_training_count} clics."
    logging.info(log_msg)
    # 3. Vérifier si le seuil est atteint
    # On vérifie si le nombre de clics a dépassé le prochain multiple du seuil
    next_threshold = (last_training_count // RETRAIN_THRESHOLD_INCREMENT + 1) * RETRAIN_THRESHOLD_INCREMENT
    if current_click_count >= next_threshold:
        logging.info(f"Seuil de {next_threshold} clics dépassé. Démarrage du ré-entraînement.")

        try:
            update_retraining_status(blob_service_client, "in_progress")
            # 4. Exécuter le script d'entraînement. Il sauvegarde le modèle directement.
            metrics = train_and_save_model(container_name=CONTAINER_NAME,
                                           clicks_blob=CLICKS_BLOB_NAME,
                                           articles_blob=METADATA_BLOB_NAME,
                                           embeddings_blob=EMBEDDINGS_BLOB_NAME,
                                           model_output_blob=MODEL_BLOB_NAME)
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
