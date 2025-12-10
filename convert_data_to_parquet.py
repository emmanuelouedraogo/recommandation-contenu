import os
import pandas as pd
import joblib
import logging
import numpy as np
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
# Le script s'attend à ce que les fichiers de données soient à la racine du projet.
LOCAL_DATA_DIR = "data"  # Dossier de sortie pour les fichiers Parquet

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


def save_df_as_parquet_local(df: pd.DataFrame, file_name_prefix: str):
    """Sauvegarde un DataFrame en tant que fichier Parquet localement."""
    output_path = os.path.join(LOCAL_DATA_DIR, f"{file_name_prefix}.parquet")
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"DataFrame sauvegardé au format Parquet : {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de {output_path}: {e}")


def convert_csv_to_parquet(local_csv_path: str):
    """Convertit un fichier CSV local en Parquet."""
    logger.info(f"Conversion de {local_csv_path} en Parquet...")
    try:
        df = pd.read_csv(local_csv_path)
        file_name_prefix = os.path.basename(local_csv_path).replace(".csv", "")
        save_df_as_parquet_local(df, file_name_prefix)
        logger.info(f"Conversion de {local_csv_path} terminée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de {local_csv_path}: {e}")


def convert_embeddings_pickle_to_parquet(local_pickle_path: str):
    """Convertit le contenu du fichier pickle d'embeddings en fichiers Parquet locaux, en gérant les différents types de données."""
    logger.info(f"Conversion de {local_pickle_path} en Parquet...")
    try:
        with open(local_pickle_path, "rb") as f:
            embeddings_data = joblib.load(f)

        # Assurer que i2vec_data est un tableau NumPy 2D
        i2vec_data = embeddings_data["i2vec"]
        
        # Convertir en tableau NumPy si ce n'est pas déjà le cas
        if not isinstance(i2vec_data, np.ndarray):
            i2vec_data = np.array(i2vec_data)

        # Si c'est un tableau 1D (par exemple, un seul embedding), le remodeler en 2D
        if i2vec_data.ndim == 1:
            i2vec_data = i2vec_data.reshape(1, -1)
            
        save_df_as_parquet_local(pd.DataFrame(i2vec_data), "articles_i2vec") # type: ignore
        save_df_as_parquet_local(pd.DataFrame(list(embeddings_data["dic_ri"].items()), columns=["article_id", "index"]), "articles_dic_ri")
        save_df_as_parquet_local(pd.DataFrame(list(embeddings_data["dic_ir"].items()), columns=["index", "article_id"]), "articles_dic_ir")
        logger.info(f"Conversion de {local_pickle_path} terminée avec succès en plusieurs fichiers Parquet.")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de {local_pickle_path}: {e}")


if __name__ == "__main__":
    convert_csv_to_parquet("clicks_sample.csv")
    convert_csv_to_parquet("articles_metadata.csv")
    convert_embeddings_pickle_to_parquet("articles_embeddings.pickle")
    logger.info("Processus de conversion des données terminé.")