# /home/epikaizo/Desktop/recommandation-contenu/reco_model_script.py (version modifiée)

import pandas as pd
import joblib
import os
import logging # type: ignore
from datetime import datetime, timezone # type: ignore
from azure.storage.blob import BlobServiceClient # type: ignore
from azure.identity import DefaultAzureCredential # type: ignore
from io import StringIO
from models import HybridRecommender  # Assurez-vous que vos classes de modèles sont importables
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

if not STORAGE_ACCOUNT_NAME:
    logging.error("AZURE_STORAGE_ACCOUNT_NAME n'est pas définie. Impossible de procéder.")
    exit(1)


def load_df_from_parquet_blob(container_name: str, blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob Parquet."""
    blob_service_client = BlobServiceClient(account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential())
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob(timeout=120)
    return pd.read_parquet(downloader.readall())


def load_csv_from_blob(container_name: str, blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob CSV."""
    blob_service_client = BlobServiceClient(account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential())
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob(encoding="utf-8", timeout=120)
    return pd.read_csv(StringIO(downloader.readall()))


def load_pickle_from_blob(container_name: str, blob_name: str):
    """Charge un objet Python depuis un blob Pickle."""
    blob_service_client = BlobServiceClient(account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential())
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob(timeout=120)
    # Azure Functions /tmp directory is suitable for temporary files
    # Ensure the directory exists
    os.makedirs("/tmp", exist_ok=True)
    local_path = os.path.join("/tmp", os.path.basename(blob_name))
    try:
        with open(local_path, "wb") as f:
            f.write(downloader.readall())
        return joblib.load(local_path)
    return None


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
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=log_blob_name)

    log_entry = { # type: ignore
        "timestamp": datetime.utcnow().isoformat(),
        "precision_at_10": metrics.get("precision_at_10", 0),
        "click_count": click_count,
    }

    try:
        downloader = blob_client.download_blob()
        log_df = pd.read_csv(StringIO(downloader.readall().decode("utf-8"))) # Log is still CSV
    except Exception:
        log_df = pd.DataFrame(columns=log_entry.keys())

    updated_log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    blob_client.upload_blob(updated_log_df.to_csv(index=False), overwrite=True)
    logging.info(f"Métrique d'entraînement enregistrée dans {log_blob_name}")


def train_and_save_model(container_name, clicks_blob, articles_blob, embeddings_blob, model_output_blob):
    """Fonction principale pour entraîner et sauvegarder le modèle."""

    # 1. Charger toutes les données nécessaires depuis Azure (maintenant en Parquet)
    logging.info("Chargement des données...")
    # Assurez-vous que les noms de blobs sont corrects pour les fichiers Parquet
    clicks_parquet_blob = clicks_blob.replace(".csv", ".parquet")
    articles_parquet_blob = articles_blob.replace(".csv", ".parquet")
    i2vec_parquet_blob = embeddings_blob.replace(".pickle", "_i2vec.parquet")
    dic_ri_parquet_blob = embeddings_blob.replace(".pickle", "_dic_ri.parquet")
    dic_ir_parquet_blob = embeddings_blob.replace(".pickle", "_dic_ir.parquet")

    clicks_df = load_df_from_parquet_blob(container_name, clicks_parquet_blob)
    articles_df = load_df_from_parquet_blob(container_name, articles_parquet_blob)
    
    # Charger les composants des embeddings depuis leurs fichiers Parquet respectifs
    i2vec_df = load_df_from_parquet_blob(container_name, i2vec_parquet_blob)
    dic_ri_df = load_df_from_parquet_blob(container_name, dic_ri_parquet_blob)
    dic_ir_df = load_df_from_parquet_blob(container_name, dic_ir_parquet_blob)

    # Reconstruire les objets nécessaires
    i2vec = i2vec_df.values # Convert DataFrame back to NumPy array
    dic_ri = dict(zip(dic_ri_df["article_id"], dic_ri_df["index"]))
    dic_ir = dict(zip(dic_ir_df["index"], dic_ir_df["article_id"]))

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

    # Enregistrer les métriques
    log_training_metrics(connect_str, container_name, metrics, len(clicks_df))

    # 4. Sauvegarder le modèle entraîné dans un fichier local temporaire
    local_model_path = os.path.join("/tmp", os.path.basename(model_output_blob))
    joblib.dump(hybrid_model, local_model_path)
    logging.info(f"Modèle sauvegardé localement dans '{local_model_path}'.")

    # 5. Uploader le modèle sur Azure
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_output_blob)
    with open(local_model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.info(f"Modèle uploadé avec succès sur Azure Blob Storage : {model_output_blob}")
