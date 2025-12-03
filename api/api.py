import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition du modèle de données pour les requêtes
class User(BaseModel):
    user_id: int

app = FastAPI(
    title="API de Recommandation de Contenu",
    description="Une API pour obtenir des recommandations d'articles pour les utilisateurs.",
    version="1.0.0"
)

# --- Chargement du modèle au démarrage de l'application ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "save", "hybrid_recommender_pipeline.pkl")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
model = None

@app.on_event("startup")
def load_model():
    global model
    
    # S'assurer que le répertoire du modèle existe
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Vérification/Création du répertoire : '{model_dir}'")

    logger.info(f"Chargement du modèle depuis : {MODEL_PATH}")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = joblib.load(f)
        logger.info("Modèle chargé avec succès.")
    except FileNotFoundError:
        logger.error(f"Erreur: Le fichier modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}")
        raise RuntimeError(f"Fichier modèle non trouvé: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors du chargement du modèle : {e}")
        raise RuntimeError(f"Erreur au chargement du modèle: {e}")

@app.get("/health/", summary="Vérifier l'état de santé de l'API")
def health_check():
    return {"status": "ok"}

@app.post("/recommendations/", summary="Obtenir des recommandations d'articles", response_model=list)
def get_recommendations(user: User):
    """
    Prend un `user_id` en entrée et retourne une liste de 5 articles recommandés.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé. Veuillez réessayer plus tard.")

    logger.info(f"Génération de recommandations pour l'utilisateur : {user.user_id}")
    recommendations_df = model.recommend_items(user.user_id, topn=5)
    
    return recommendations_df.to_dict(orient='records')