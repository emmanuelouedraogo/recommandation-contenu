# /home/epikaizo/Desktop/recommandation-contenu/reco_model_script.py (version modifiée)

import pandas as pd
import joblib
import os
import logging
from datetime import datetime  # type: ignore
from azure.storage.blob import BlobServiceClient  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from models import HybridRecommender  # Assurez-vous que vos classes de modèles sont importables
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

if not STORAGE_ACCOUNT_NAME:
    logging.error("AZURE_STORAGE_ACCOUNT_NAME is not set. Unable to proceed.")
    exit(1)


def load_pickle_from_blob(container_name: str, blob_name: str):
    """Charge un objet Python depuis un blob Pickle."""
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential()
    )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob(timeout=120)
    
    # Charger directement depuis le flux d'octets en mémoire pour plus de robustesse
    blob_bytes = downloader.readall()
    return joblib.load(BytesIO(blob_bytes))


def evaluate_precision_at_k(model, test_df, k=10):
    """Calcule la Précision@k moyenne pour les utilisateurs du jeu de test."""
    logging.info(f"Évaluation de la Précision@{k}...")

    # Grouper les articles réellement cliqués par utilisateur dans le jeu de test
    user_ground_truth = test_df.groupby("user_id")["article_id"].apply(list).to_dict()

    precisions = []
    test_users = list(user_ground_truth.keys())

    for user_id in test_users:
        # Obtenir les k meilleures recommandations
        recommendations_df = model.recommend_items(user_id, topn=k)
        recommended_items = recommendations_df["article_id"].tolist()

        # Calculer le nombre de "hits"
        hits = len(set(recommended_items) & set(user_ground_truth[user_id]))
        precisions.append(hits / k)

    return sum(precisions) / len(precisions) if precisions else 0


def log_training_metrics(container_name, metrics, click_count):
    """Enregistre les métriques d'entraînement dans un fichier CSV sur Azure Blob Storage."""
    log_blob_name = "logs/training_log.csv"
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential()
    )
    append_blob_client = blob_service_client.get_blob_client(container=container_name, blob=log_blob_name)

    log_entry = {  # type: ignore
        "timestamp": datetime.utcnow().isoformat(),
        "precision_at_10": metrics.get("precision_at_10", 0),
        "click_count": click_count,
    }

    # S'assurer que le blob existe et est bien un Append Blob
    try:
        append_blob_client.get_blob_properties()
    except Exception:  # Si le blob n'existe pas, le créer
        append_blob_client.create_blob()
        # Ajouter l'en-tête CSV au nouveau fichier
        header = "timestamp,precision_at_10,click_count\n"
        append_blob_client.append_block(header.encode("utf-8"))

    # Ajouter la nouvelle entrée au blob
    new_log_line = f"{log_entry['timestamp']},{log_entry['precision_at_10']},{log_entry['click_count']}\n"
    append_blob_client.append_block(new_log_line.encode("utf-8"))
    logging.info(f"Métrique d'entraînement enregistrée dans {log_blob_name} via Append Blob.")


def load_df_from_parquet_blob(container_name: str, blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob Parquet."""
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential()
    )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob(timeout=120)
    return pd.read_parquet(downloader.readall())


def train_and_save_model(container_name, clicks_blob, articles_blob, embeddings_blob, model_output_blob):
    """Fonction principale pour entraîner et sauvegarder le modèle."""

    # 1. Charger toutes les données nécessaires depuis Azure (maintenant en Parquet)
    logging.info("Chargement des données...")
    try:
        # Assurez-vous que les noms de blobs sont corrects pour les fichiers Parquet
        clicks_parquet_blob = clicks_blob.replace(".csv", ".parquet")
        articles_parquet_blob = articles_blob.replace(".csv", ".parquet")
        i2vec_parquet_blob = embeddings_blob.replace(".pickle", "_i2vec.parquet")
        dic_ri_parquet_blob = embeddings_blob.replace(".pickle", "_dic_ri.parquet")
        dic_ir_parquet_blob = embeddings_blob.replace(".pickle", "_dic_ir.parquet")

        clicks_df = load_df_from_parquet_blob(container_name, clicks_parquet_blob)
        articles_df = load_df_from_parquet_blob(container_name, articles_parquet_blob)

        # Reconstruire les objets nécessaires à partir des fichiers Parquet
        i2vec = load_df_from_parquet_blob(container_name, i2vec_parquet_blob).values
        dic_ri = dict(load_df_from_parquet_blob(container_name, dic_ri_parquet_blob).itertuples(index=False, name=None))
        dic_ir = dict(load_df_from_parquet_blob(container_name, dic_ir_parquet_blob).itertuples(index=False, name=None))
    except Exception as e:
        logging.error(f"Échec du chargement des données d'entraînement depuis Azure. Erreur : {e}", exc_info=True)
        raise  # Propage l'exception pour que la fonction appelante puisse la gérer

    if clicks_df.empty or len(clicks_df) < 10:
        raise ValueError("Pas assez de données de clics pour l'entraînement.")

    train_df, test_df = train_test_split(clicks_df, test_size=0.2, random_state=42)  # noqa

    # 2. Entraîner le modèle hybride
    logging.info("Entraînement du modèle hybride...")
    hybrid_model = HybridRecommender(
        data_map=train_df, i2vec=i2vec, dic_ri=dic_ri, dic_ir=dic_ir, items_df=articles_df, cf_weight=0.5, cb_weight=0.5
    )
    hybrid_model.fit()

    # 3. Évaluer le modèle
    precision = evaluate_precision_at_k(hybrid_model, test_df, k=10)
    logging.info(f"Précision@10 obtenue : {precision:.4f}")
    metrics = {"precision_at_10": precision}

    # Enregistrer les métriques.
    # Note: La fonction log_training_metrics utilise maintenant DefaultAzureCredential.
    log_training_metrics(container_name, metrics, len(clicks_df))

    # 4. Sauvegarder le modèle entraîné dans un fichier local temporaire
    local_model_path = os.path.join("/tmp", os.path.basename(model_output_blob))
    joblib.dump(hybrid_model, local_model_path)
    logging.info(f"Modèle sauvegardé localement dans '{local_model_path}'.")

    # 5. Uploader le modèle sur Azure en utilisant DefaultAzureCredential
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential()
    )
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_output_blob)
    with open(local_model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.info(f"Modèle uploadé avec succès sur Azure Blob Storage : {model_output_blob}")

    return metrics
