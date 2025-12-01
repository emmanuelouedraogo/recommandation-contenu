# /home/epikaizo/Desktop/recommandation-contenu/reco_model_script.py (version modifiée)

import pandas as pd
import joblib
import os
from azure.storage.blob import BlobServiceClient
from io import StringIO
from models import HybridRecommender # Assurez-vous que vos classes de modèles sont importables
from sklearn.model_selection import train_test_split

def load_data_from_azure(connect_str, container_name, blob_name):
    """Charge un fichier depuis Azure Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloader = blob_client.download_blob()
    
    if blob_name.endswith('.csv'):
        return pd.read_csv(StringIO(downloader.readall().decode('utf-8')))
    elif blob_name.endswith('.pickle'):
        local_path = f"/tmp/{os.path.basename(blob_name)}"
        with open(local_path, "wb") as f:
            f.write(downloader.readall())
        return joblib.load(local_path)
    return None

def evaluate_precision_at_k(model, test_df, k=10):
    """Calcule la Précision@k moyenne pour les utilisateurs du jeu de test."""
    print(f"Évaluation de la Précision@{k}...")
    
    # Grouper les articles réellement cliqués par utilisateur dans le jeu de test
    user_ground_truth = test_df.groupby('user_id')['article_id'].apply(list).to_dict()
    
    precisions = []
    test_users = list(user_ground_truth.keys())

    for user_id in test_users:
        # Obtenir les k meilleures recommandations
        recommendations_df = model.recommend_items(user_id, topn=k)
        recommended_items = recommendations_df['article_id'].tolist()
        
        # Calculer le nombre de "hits"
        hits = len(set(recommended_items) & set(user_ground_truth[user_id]))
        precisions.append(hits / k)
        
    return sum(precisions) / len(precisions) if precisions else 0

def train_and_save_model(connect_str, container_name, clicks_blob, articles_blob, embeddings_blob):
    """Fonction principale pour entraîner et sauvegarder le modèle."""
    
    # 1. Charger toutes les données nécessaires depuis Azure
    print("Chargement des données...")
    clicks_df = load_data_from_azure(connect_str, container_name, clicks_blob)
    articles_df = load_data_from_azure(connect_str, container_name, articles_blob)
    embeddings_data = load_data_from_azure(connect_str, container_name, embeddings_blob)
    
    if clicks_df.empty or len(clicks_df) < 10:
        raise ValueError("Pas assez de données de clics pour l'entraînement.")

    train_df, test_df = train_test_split(clicks_df, test_size=0.2, random_state=42)

    # Extraire les données du pickle d'embeddings
    i2vec = embeddings_data['i2vec']
    dic_ri = embeddings_data['dic_ri']
    dic_ir = embeddings_data['dic_ir']
    
    # 2. Entraîner le modèle hybride
    print("Entraînement du modèle hybride...")
    hybrid_model = HybridRecommender(
        data_map=train_df, # Entraîner uniquement sur le jeu d'entraînement
        i2vec=i2vec,
        dic_ri=dic_ri,
        dic_ir=dic_ir,
        items_df=articles_df,
        cf_weight=0.5,
        cb_weight=0.5
    )
    hybrid_model.fit()
    
    # 3. Évaluer le modèle
    precision = evaluate_precision_at_k(hybrid_model, test_df, k=10)
    print(f"Précision@10 obtenue : {precision:.4f}")
    metrics = {"precision_at_10": precision}

    # 4. Sauvegarder le modèle entraîné dans un fichier local temporaire
    local_model_path = "/tmp/hybrid_recommender_pipeline.pkl"
    joblib.dump(hybrid_model, local_model_path)
    print(f"Modèle sauvegardé localement dans '{local_model_path}'.")
    
    return local_model_path, metrics

# ... (le reste de votre script si vous avez une exécution via __main__)
