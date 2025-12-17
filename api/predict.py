import os
import pandas as pd
from pathlib import Path
import joblib
import logging
from models import HybridRecommender, ContentBasedTimeDecayRecommender  # Classes nécessaires pour le unpickling


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration des chemins ---

try:
    # Méthode robuste lorsque le code est exécuté comme un script.
    # __file__ est le chemin du script courant.
    # .parent.parent remonte de deux niveaux pour atteindre la racine du projet.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Solution de repli pour les environnements interactifs (ex: notebooks)
    # où __file__ n'est pas défini. On suppose que le notebook est exécuté
    # depuis la racine du projet.
    PROJECT_ROOT = Path.cwd()

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# S'assurer que les répertoires pour les modèles et les données existent
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_PATH = MODELS_DIR / "hybrid_recommender_pipeline.pkl"
ARTICLES_PATH = DATA_DIR / "articles_metadata.csv"
CLICKS_PATH = DATA_DIR / "clicks_sample.csv"


def load_pipeline(path: str):
    """
    Charge le pipeline de recommandation depuis le chemin spécifié.
    """
    try:
        pipeline = joblib.load(Path(path))
        logger.info(f"Pipeline chargé avec succès depuis : {Path(path).resolve()}")
        return pipeline
    except FileNotFoundError:
        logger.error(f"Erreur : Le fichier du pipeline est introuvable à l'adresse : {Path(path).resolve()}")
        raise
    except Exception as e:
        logger.error(f"Une erreur est survenue lors du chargement du pipeline : {e}")
        raise


def _generate_recommendations_logic(
    user_id: int, pipeline, articles_df: pd.DataFrame, clicks_df: pd.DataFrame, top_n: int = 10
):
    """
    Génère des recommandations pour un utilisateur donné.

    Args:
        user_id (int): L'identifiant de l'utilisateur.
        pipeline: Le pipeline de recommandation entraîné.
        articles_df (pd.DataFrame): DataFrame des métadonnées des articles.
        clicks_df (pd.DataFrame): DataFrame des clics des utilisateurs.
        top_n (int): Le nombre de recommandations à retourner.

    Returns:
        list: Une liste de dictionnaires, chaque dictionnaire représentant un article recommandé.
    """
    # 1. Vérifier que le pipeline est un objet valide avant de l'utiliser.
    if not hasattr(pipeline, "recommend_items"):
        logger.error(f"L'objet pipeline fourni n'a pas de méthode 'recommend_items'. Type de l'objet: {type(pipeline)}")
        return []

    try:
        # Utiliser la méthode recommend_items du modèle hybride, qui est la bonne façon d'obtenir des recommandations.
        recommendations_df = pipeline.recommend_items(uid=user_id, topn=top_n)

        # 2. Valider le type de la sortie du modèle.
        if not isinstance(recommendations_df, pd.DataFrame):
            logger.error(f"La méthode 'recommend_items' a retourné un type inattendu : {type(recommendations_df)}")
            return []

        if recommendations_df.empty:
            logger.warning(
                f"Aucune recommandation générée pour l'utilisateur {user_id} (cold start ou pas d'articles pertinents). Retour d'une liste vide."
            )
            return []

        # 3. Valider la présence des colonnes nécessaires avant de les utiliser.
        required_cols = {"article_id", "final_score"}
        if not required_cols.issubset(recommendations_df.columns):
            logger.error(
                f"Le DataFrame de recommandations ne contient pas les colonnes requises {required_cols}. Colonnes présentes : {list(recommendations_df.columns)}"
            )
            return []

        recommendations_df = recommendations_df.rename(columns={"final_score": "score"})
        return recommendations_df[["article_id", "score"]].to_dict(orient="records")

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction avec le pipeline pour l'utilisateur {user_id}: {e}", exc_info=True)
        logger.warning(f"Utilisation de la stratégie de fallback (articles populaires) pour l'utilisateur {user_id}.")

        try:
            if clicks_df.empty:
                logger.warning("Le DataFrame de clics est vide, impossible de fournir des recommandations populaires.")
                return []

            # Calculer les articles les plus populaires en comptant les occurrences de chaque article_id
            click_counts = clicks_df["click_article_id"].value_counts()
            popular_articles = click_counts.nlargest(top_n).reset_index()
            popular_articles.columns = [
                "article_id",
                "score",
            ]  # Renommer les colonnes pour correspondre au format de sortie

            return popular_articles.to_dict(orient="records")
        except Exception as fallback_e:
            logger.error(
                f"Erreur critique lors de la tentative de fallback pour l'utilisateur {user_id}: {fallback_e}",
                exc_info=True,
            )
            return []


class Recommender:
    """
    Classe qui encapsule le pipeline de recommandation et les données nécessaires.
    Les données (articles, clics) sont chargées une seule fois lors de l'initialisation.
    """

    def __init__(self, pipeline, articles_df: pd.DataFrame, clicks_df: pd.DataFrame):
        """Initialise le Recommender avec des objets déjà chargés en mémoire."""
        self.pipeline = pipeline
        self.articles_df = articles_df
        self.clicks_df = clicks_df
        self._validate_data()

    def _validate_data(self):
        logger.info("Validation des DataFrames chargés...")
        if "article_id" not in self.articles_df.columns:
            raise ValueError("La colonne 'article_id' est manquante dans le fichier d'articles.")
        if "user_id" not in self.clicks_df.columns or "click_article_id" not in self.clicks_df.columns:
            raise ValueError("Les colonnes 'user_id' ou 'click_article_id' sont manquantes dans le fichier de clics.")
        logger.info("Validation des données réussie.")

    def generate_recommendations(self, user_id: int, top_n: int = 10):
        """Génère des recommandations en utilisant les données pré-chargées."""
        if self.pipeline is None or self.articles_df is None or self.clicks_df is None:
            logger.error("Le Recommender n'est pas correctement initialisé (données ou pipeline manquants).")
            return []

        return _generate_recommendations_logic(
            user_id=user_id, pipeline=self.pipeline, articles_df=self.articles_df, clicks_df=self.clicks_df, top_n=top_n
        )


if __name__ == "__main__":
    # --- Exemple d'utilisation ---
    # Ceci s'exécute uniquement lorsque le script est appelé directement.
    logger.info("--- Début du script de prédiction en mode standalone ---")

    # 1. Initialiser le Recommender (charge le modèle et les données une seule fois)
    # Pour l'exécution locale, nous chargeons manuellement les artefacts
    pipeline_obj = load_pipeline(PIPELINE_PATH)
    articles_obj = pd.read_csv(ARTICLES_PATH)
    clicks_obj = pd.read_csv(CLICKS_PATH, nrows=100000)

    recommender = Recommender(pipeline=pipeline_obj, articles_df=articles_obj, clicks_df=clicks_obj)

    sample_user_id = 3  # Un utilisateur exemple
    # 2. Générer des recommandations (utilise les données en mémoire)
    recommendations = recommender.generate_recommendations(user_id=sample_user_id, top_n=5)

    logger.info(f"\n--- Top 5 Recommandations pour l'utilisateur {sample_user_id} ---")
    for reco in recommendations:
        logger.info(f"  - Article ID: {reco['article_id']}, Score prédit: {reco['score']:.4f}")
