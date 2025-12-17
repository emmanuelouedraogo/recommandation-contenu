import unittest
import pandas as pd
from unittest.mock import MagicMock
import sys
import os

# Ajouter le répertoire racine au path pour permettre l'import de 'api.predict'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestGenerateRecommendations(unittest.TestCase):

    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer des données de test
        self.articles_data = {"article_id": [10, 20, 30, 40, 50]}
        self.articles_df = pd.DataFrame(self.articles_data)

        self.clicks_data = {"user_id": [1, 1, 2, 2, 1], "click_article_id": [10, 20, 10, 30, 30]}
        self.clicks_df = pd.DataFrame(self.clicks_data)

        # Créer un faux pipeline qui simule l'interface de HybridRecommender
        self.mock_pipeline = MagicMock()

    def test_recommendations_standard(self):
        """
        Teste le cas nominal où des recommandations sont générées pour un utilisateur.
        """
        # Configurer le mock du pipeline pour retourner un DataFrame de recommandations
        reco_df = pd.DataFrame({"article_id": [40, 50], "final_score": [0.9, 0.8]})
        self.mock_pipeline.recommend_items.return_value = reco_df

        from api.predict import Recommender

        # Instancier le Recommender avec les données de test en mémoire
        recommender = Recommender(pipeline=self.mock_pipeline, articles_df=self.articles_df, clicks_df=self.clicks_df)

        user_id = 1
        recommendations = recommender.generate_recommendations(user_id=user_id, top_n=5)

        # Vérifications
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["article_id"], 40)  # L'article 40 a le score le plus élevé
        self.assertAlmostEqual(recommendations[0]["score"], 0.9)

        # Vérifier que la méthode de recommandation du pipeline a été appelée correctement
        self.mock_pipeline.recommend_items.assert_called_once_with(uid=user_id, topn=5)

    def test_fallback_on_prediction_error(self):
        """
        Teste que la stratégie de fallback (articles populaires) est utilisée si le pipeline lève une exception.
        """
        # Configurer le mock pour qu'il lève une exception lors de l'appel
        self.mock_pipeline.recommend_items.side_effect = Exception("Erreur de prédiction simulée")
        from api.predict import Recommender

        recommender = Recommender(pipeline=self.mock_pipeline, articles_df=self.articles_df, clicks_df=self.clicks_df)

        recommendations = recommender.generate_recommendations(user_id=1, top_n=2)

        # Vérifications du fallback
        # Dans nos données de test, les articles 10 et 30 sont les plus populaires (2 clics chacun)
        self.assertEqual(len(recommendations), 2)
        recommended_ids = {rec["article_id"] for rec in recommendations}
        self.assertSetEqual(recommended_ids, {10, 30})

    def test_invalid_pipeline_object(self):
        """
        Teste que la fonction retourne une liste vide si l'objet pipeline est invalide (pas de méthode 'recommend_items').
        """
        invalid_pipeline = object()  # Un objet simple qui n'a pas la méthode requise
        from api.predict import _generate_recommendations_logic

        recommendations = _generate_recommendations_logic(
            user_id=1,
            pipeline=invalid_pipeline,
            articles_df=self.articles_df,
            clicks_df=self.clicks_df,
        )
        self.assertEqual(recommendations, [])

    def test_model_returns_non_dataframe(self):
        """
        Teste que la fonction retourne une liste vide si le pipeline retourne un type incorrect (pas un DataFrame).
        """
        self.mock_pipeline.recommend_items.return_value = None  # Retourne None au lieu d'un DataFrame
        from api.predict import _generate_recommendations_logic

        recommendations = _generate_recommendations_logic(
            user_id=1,
            pipeline=self.mock_pipeline,
            articles_df=self.articles_df,
            clicks_df=self.clicks_df,
        )
        self.assertEqual(recommendations, [])

    def test_model_returns_df_with_missing_columns(self):
        """
        Teste que la fonction retourne une liste vide si le DataFrame retourné n'a pas les colonnes requises.
        """
        # Retourne un DataFrame avec des noms de colonnes incorrects
        invalid_reco_df = pd.DataFrame({"id": [40], "prediction": [0.9]})
        self.mock_pipeline.recommend_items.return_value = invalid_reco_df
        from api.predict import _generate_recommendations_logic

        recommendations = _generate_recommendations_logic(
            user_id=1,
            pipeline=self.mock_pipeline,
            articles_df=self.articles_df,
            clicks_df=self.clicks_df,
        )
        self.assertEqual(recommendations, [])


if __name__ == "__main__":
    unittest.main()
