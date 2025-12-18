import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from surprise import Reader, Dataset, SVDpp, SVD, KNNWithMeans

"""
## **5. Définition des Classes de Modèles**

Nous définissons une structure de classes standard pour tous nos modèles. Chaque classe doit avoir :
- `get_model_name()`: Renvoie le nom du modèle.
- `fit()`: Entraîne le modèle sur les données d'entraînement.
- `recommend_items(uid, topn)`: Génère des recommandations pour un
  utilisateur donné.
"""


# ### **Modèles Basés sur la Popularité**
class PopularityFiltRecommender:  # Modèle le plus simple
    MODEL_NAME = "Popularity-Filtering"

    def __init__(self, data_map: pd.DataFrame):
        self.train_user_interact: pd.DataFrame = data_map
        self.raw_reco: Optional[pd.DataFrame] = None

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(self) -> None:
        # Calcule la popularité de chaque article en sommant le nombre de clics ('nb').
        # 'nb' agit comme un "rating": plus il est élevé, plus l'article
        # est populaire.  # noqa: E501
        self.raw_reco = self.train_user_interact.groupby("article_id")["nb"].sum()
        self.raw_reco = self.raw_reco.sort_values(ascending=False).reset_index()

    def recommend_items(self, uid: Any, topn: int = 5) -> pd.DataFrame:
        # Recommande simplement les N articles les plus populaires,
        # quelle que soit l'utilisateur  # noqa: E501
        return self.raw_reco.head(topn) if topn > 0 else self.raw_reco  # type: ignore


class RecentPopularityRecommender:  # Popularity model sensitive to time
    MODEL_NAME = "Recent-Popularity"

    def __init__(self, data_map: pd.DataFrame):
        self.train_user_interact: pd.DataFrame = data_map
        self.raw_reco: Optional[pd.DataFrame] = None

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(self) -> None:
        # Calcule la popularité uniquement sur les interactions les plus
        # récentes (dernier quartile)
        quantile_75 = self.train_user_interact["click_timestamp"].quantile(0.75)
        recent_clicks = self.train_user_interact[self.train_user_interact["click_timestamp"] >= quantile_75]
        # La popularité est calculée en sommant le "rating" implicite ('nb')
        # des clics récents.
        self.raw_reco = recent_clicks.groupby("article_id")["nb"].sum()
        self.raw_reco = self.raw_reco.sort_values(ascending=False).reset_index()

    def recommend_items(self, uid: Any, topn: int = 5) -> pd.DataFrame:
        # Recommande les articles qui sont devenus populaires récemment
        return self.raw_reco.head(topn) if topn > 0 else self.raw_reco  # type: ignore


class PopularityByCategoryRecommender:  # Custom popularity model by category
    MODEL_NAME = "Popularity-By-Category"

    def __init__(self, data_map: pd.DataFrame, items_df: pd.DataFrame):
        self.train_user_interact: pd.DataFrame = data_map
        self.items_df: pd.DataFrame = items_df
        self.category_popularity: Optional[pd.DataFrame] = None

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(self) -> None:
        # Pre-calculates the popularity of each item within its category
        merged_df = self.train_user_interact.merge(self.items_df[["article_id", "category_id"]], on="article_id")
        # La popularité par catégorie est aussi basée sur la somme des
        # "ratings" ('nb').
        self.category_popularity = (
            merged_df.groupby(["category_id", "article_id"])["nb"].sum().reset_index()
        )  # noqa: E501
        self.category_popularity = self.category_popularity.sort_values("nb", ascending=False)

    def recommend_items(self, uid: Any, topn: int = 5) -> pd.DataFrame:
        # 1. Trouver les catégories préférées de l'utilisateur
        user_clicks = self.train_user_interact[self.train_user_interact["user_id"] == uid]
        user_items = user_clicks.merge(self.items_df[["article_id", "category_id"]], on="article_id")

        if user_items.empty:
            # For new users (cold start), fall back to global popularity
            pop_reco = self.category_popularity.groupby("article_id")["nb"].sum()
            pop_reco = pop_reco.sort_values(ascending=False).reset_index()
            return pop_reco.head(topn)  # noqa: E501

        # 2. Identifier les 3 catégories les plus lues par l'utilisateur
        top_categories = user_items["category_id"].value_counts().index[:3]  # Top 3 categories

        # 3. Recommend the most popular articles from these categories  # noqa: E501
        recs = self.category_popularity[self.category_popularity["category_id"].isin(top_categories)]  # noqa: E501

        # 4. Filter out already seen articles and return the top N
        items_to_ignore = set(user_clicks["article_id"])
        recs = recs[~recs["article_id"].isin(items_to_ignore)]

        recs = recs.drop_duplicates(subset="article_id").sort_values("nb", ascending=False)
        return recs.head(topn) if topn > 0 else recs


# ### **Modèles Basés sur le Contenu (Content-Based)**
class ContentBasedRecommender:  # Content-based model, full user profile
    MODEL_NAME = "Content-Based"

    def __init__(self, data_map: pd.DataFrame, i2vec: np.ndarray, dic_ri: Dict[Any, int], dic_ir: Dict[int, Any]):
        self.dic_ir: Dict[int, Any] = dic_ir
        self.dic_ri: Dict[Any, int] = dic_ri
        self.items_embedding: np.ndarray = i2vec
        self.train_user_interact: pd.DataFrame = data_map
        self.user_profiles: Dict[Any, np.ndarray] = {}

    def _build_users_profile(
        self, uid: Any, click_df: pd.DataFrame, emb_matrix: np.ndarray, dic_ri: Dict[Any, int]
    ) -> np.ndarray:
        """Builds a user profile as the weighted average of the embeddings of
        the articles they have read."""
        click_uid_df = click_df.loc[click_df.user_id == uid]
        # Récupère les embeddings des articles lus
        user_item_profiles = np.array([emb_matrix[dic_ri[iid]] for iid in click_uid_df["article_id"] if iid in dic_ri])
        if len(user_item_profiles) == 0:
            return np.zeros((1, self.items_embedding.shape[1]))

        # RATING CONSIDERATION: The number of clicks ('nb') is used as a
        # weight (rating). An item clicked multiple times will have more
        # influence on the user's profile.
        user_item_strengths = np.array(click_uid_df["nb"]).reshape(-1, 1)
        # Calcule la moyenne des embeddings des articles, pondérée par ce
        # "rating"
        sum_of_strengths = np.sum(user_item_strengths)
        if sum_of_strengths == 0:
            return np.zeros((1, self.items_embedding.shape[1]))

        weighted_avg = np.sum(user_item_profiles * user_item_strengths, axis=0)
        user_item_strengths_weighted_avg = weighted_avg / sum_of_strengths
        # Normalize the profile to have a norm of 1 (important for cosine
        # similarity)
        return preprocessing.normalize(user_item_strengths_weighted_avg.reshape(1, -1))  # noqa: E501  # noqa: E501

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(self) -> None:
        # Construit le profil pour chaque utilisateur unique dans les données
        # d'entraînement
        for uid in tqdm(self.train_user_interact.user_id.unique(), desc="Building User Profiles"):
            self.user_profiles[uid] = self._build_users_profile(
                uid, self.train_user_interact, self.items_embedding, self.dic_ri
            )

    def recommend_items(self, uid: Any, topn: int = 10, items_to_filter: Optional[List[Any]] = None) -> pd.DataFrame:
        if uid not in self.user_profiles:
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])

        # 1. Calcule la similarité cosinus entre le profil de l'utilisateur et
        # TOUS les articles
        cosine_similarities = cosine_similarity(self.user_profiles[uid], self.items_embedding)
        # 2. Trie les articles par similarité décroissante
        similar_indices = cosine_similarities.argsort().flatten()[::-1]
        similar_items = [(self.dic_ir[i], cosine_similarities[0, i]) for i in similar_indices if i in self.dic_ir]

        # 3. Filtre les articles que l'utilisateur a déjà lus
        items_to_ignore = set(
            self.train_user_interact.loc[self.train_user_interact.user_id == uid, "article_id"]  # noqa: E501
        )
        reco = pd.DataFrame(similar_items, columns=["article_id", "cb_cosine_with_profile"])  # noqa: E501
        reco = reco[~reco.article_id.isin(items_to_ignore)]

        # Filtre optionnel pour l'évaluation
        if items_to_filter is not None:
            reco = reco[reco["article_id"].isin(items_to_filter)]

        return reco.head(topn) if topn > 0 else reco


class ContentBasedLastClickRecommender(ContentBasedRecommender):  # Based on short-term interest
    MODEL_NAME = "Content-Based-Last-Click"

    def fit(self) -> None:
        pass  # Pas de fit global, le profil est calculé à la volée

    def recommend_items(self, uid: Any, topn: int = 10, items_to_filter: Optional[List[Any]] = None) -> pd.DataFrame:
        user_clicks = self.train_user_interact[self.train_user_interact.user_id == uid]
        if user_clicks.empty:
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])

        # 1. Trouve le dernier article sur lequel l'utilisateur a cliqué  # noqa: E501
        last_click = user_clicks.sort_values("click_timestamp", ascending=False).iloc[0]
        last_article_id = last_click["article_id"]
        # 2. Utilise l'embedding de cet article comme un "profil utilisateur temporaire"
        last_article_inner_id = self.dic_ri.get(last_article_id)
        if last_article_inner_id is None or last_article_inner_id >= len(self.items_embedding):
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])  # noqa: E501

        user_profile = self.items_embedding[last_article_inner_id].reshape(1, -1)
        # 3. Recommande des articles similaires à ce dernier article
        cosine_similarities = cosine_similarity(user_profile, self.items_embedding)
        similar_indices = cosine_similarities.argsort().flatten()[::-1]
        similar_items = [(self.dic_ir[i], cosine_similarities[0, i]) for i in similar_indices if i in self.dic_ir]

        # 4. Filtre et retourne le top N
        items_to_ignore = set(user_clicks.article_id)
        reco_df = pd.DataFrame(similar_items, columns=["article_id", "cb_cosine_with_profile"])  # noqa: E501
        reco = reco_df[~reco_df.article_id.isin(items_to_ignore)]

        # Filtre optionnel pour l'évaluation
        if items_to_filter is not None:
            reco = reco[reco["article_id"].isin(items_to_filter)]

        return reco.head(topn) if topn > 0 else reco


class ContentBasedMostInteractedItemRecommender(ContentBasedRecommender):  # Based on "favorite" item
    MODEL_NAME = "Content-Based-Most-Interacted"

    def fit(self) -> None:
        pass  # Pas de fit global

    def recommend_items(self, uid: Any, topn: int = 10, items_to_filter: Optional[List[Any]] = None) -> pd.DataFrame:
        user_clicks = self.train_user_interact[self.train_user_interact.user_id == uid]
        if user_clicks.empty:
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])

        # 1. Trouve l'article avec lequel l'utilisateur a le plus interagi  # noqa: E501
        most_interacted = user_clicks.sort_values("nb", ascending=False).iloc[0]
        fav_article_id = most_interacted["article_id"]

        # 2. Utilise l'embedding de cet article comme profil
        fav_article_inner_id = self.dic_ri.get(fav_article_id)
        if fav_article_inner_id is None or fav_article_inner_id >= len(self.items_embedding):
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])  # noqa: E501

        user_profile = self.items_embedding[fav_article_inner_id].reshape(1, -1)
        # 3. Recommande des articles similaires à cet article "préféré"
        cosine_similarities = cosine_similarity(user_profile, self.items_embedding)
        similar_indices = cosine_similarities.argsort().flatten()[::-1]
        similar_items = [(self.dic_ir[i], cosine_similarities[0, i]) for i in similar_indices if i in self.dic_ir]

        # 4. Filtre et retourne le top N
        items_to_ignore = set(user_clicks.article_id)
        reco_df = pd.DataFrame(similar_items, columns=["article_id", "cb_cosine_with_profile"])
        reco = reco_df[~reco_df.article_id.isin(items_to_ignore)]

        # Filtre optionnel pour l'évaluation
        if items_to_filter is not None:
            reco = reco[reco["article_id"].isin(items_to_filter)]

        return reco.head(topn) if topn > 0 else reco


class ContentBasedTimeDecayRecommender(ContentBasedRecommender):
    MODEL_NAME = "Content-Based-Time-Decay"

    def __init__(
        self,
        data_map: pd.DataFrame,
        i2vec: np.ndarray,
        dic_ri: Dict[Any, int],
        dic_ir: Dict[int, Any],
        decay_rate: float = 0.05,
    ):
        super().__init__(data_map, i2vec, dic_ri, dic_ir)
        self.decay_rate: float = decay_rate  # Lambda (λ) pour la décroissance
        self.MODEL_NAME = f"Content-Based-Time-Decay(λ={decay_rate})"

    def _build_users_profile(
        self, uid: Any, click_df: pd.DataFrame, emb_matrix: np.ndarray, dic_ri: Dict[Any, int]
    ) -> np.ndarray:
        """Surcharge la méthode pour inclure la décroissance temporelle."""
        # Filtrer en amont pour ne garder que les articles présents dans le
        # dictionnaire d'embeddings
        click_uid_df = click_df.loc[(click_df.user_id == uid) & (click_df.article_id.isin(dic_ri))]

        if click_uid_df.empty:
            return np.zeros((1, self.items_embedding.shape[1]))

        # Calcul de la décroissance temporelle
        latest_timestamp = click_uid_df["click_timestamp"].max()
        time_diff = latest_timestamp - click_uid_df["click_timestamp"]
        time_diff_days = time_diff / (3600 * 24)
        time_decay_weight = np.exp(-self.decay_rate * time_diff_days.values)

        # RATING CONSIDERATION: The final weight is a combination of the
        # 'nb' rating and time decay. A recent interaction with a high
        # rating will have the most weight.
        final_weights = (click_uid_df["nb"] * time_decay_weight).values.reshape(-1, 1)
        user_item_profiles = np.array([emb_matrix[dic_ri[iid]] for iid in click_uid_df["article_id"]])  # noqa: E501
        sum_of_weights = np.sum(final_weights)
        if sum_of_weights == 0 or len(user_item_profiles) == 0:
            return np.zeros((1, self.items_embedding.shape[1]))

        weighted_avg = np.sum(user_item_profiles * final_weights, axis=0)
        weighted_avg_profile = weighted_avg / sum_of_weights
        return preprocessing.normalize(weighted_avg_profile.reshape(1, -1))


# ### **Modèles de Filtrage Collaboratif (avec `surprise`)**
# Ces modèles utilisent les interactions explicites (ici, le nombre de clics
# `nb` comme un "rating") pour trouver des similarités.


class CollabFiltRecommender:  # Base model for Surprise algorithms
    MODEL_NAME = "Collaborative-Filtering-SVDpp"

    def __init__(self, data_map: pd.DataFrame):
        self.train_user_interact: pd.DataFrame = data_map
        self.algo: Optional[Any] = None

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(self) -> None:
        # 1. Prépare les données pour la bibliothèque Surprise
        # L'échelle est indicative, les vraies valeurs peuvent dépasser 5.
        reader = Reader(rating_scale=(0, 5))
        # PRISE EN COMPTE DU RATING : La colonne 'nb' est explicitement passée
        # comme étant le "rating". Surprise va interpréter un 'nb' élevé comme
        # un signal d'intérêt fort.
        data = Dataset.load_from_df(self.train_user_interact[["user_id", "article_id", "nb"]], reader)
        trainset = data.build_full_trainset()
        # 2. Entraîne l'algorithme SVD++, une version améliorée de SVD qui
        # prend en compte les interactions implicites.
        self.algo = SVDpp(n_epochs=20, lr_all=0.007, reg_all=0.1, random_state=42)
        self.algo.fit(trainset)

    def recommend_items(self, uid: Any, topn: int = 5) -> pd.DataFrame:
        # 1. Identifie les articles que l'utilisateur n'a pas encore vus
        iid_to_ignore = set(
            self.train_user_interact.loc[self.train_user_interact.user_id == uid, "article_id"]  # noqa: E501
        )
        all_items = set(self.train_user_interact.article_id)
        items2pred_ids = all_items - iid_to_ignore

        # 2. Prédit un score pour chaque article non vu
        predictions = [self.algo.predict(uid=uid, iid=iid) for iid in items2pred_ids]

        # 3. Trie les prédictions et retourne le top N
        recs = [(pred.iid, pred.est) for pred in predictions]
        recommendations_df = pd.DataFrame(recs, columns=["article_id", "pred"]).sort_values(  # noqa: E501
            by="pred", ascending=False
        )

        return recommendations_df.head(topn) if topn > 0 else recommendations_df


class CollabFiltSVDRecommender(CollabFiltRecommender):  # Classic matrix factorization
    MODEL_NAME = "Collaborative-Filtering-SVD"

    def fit(self) -> None:
        reader = Reader(rating_scale=(0, 5))
        # PRISE EN COMPTE DU RATING : Comme pour SVDpp, 'nb' est utilisé comme
        # le rating.
        data = Dataset.load_from_df(self.train_user_interact[["user_id", "article_id", "nb"]], reader)  # noqa: E501
        trainset = data.build_full_trainset()
        # Utilise l'algorithme SVD, un standard de la factorisation de matrice.
        self.algo = SVD(n_epochs=20, lr_all=0.005, reg_all=0.1, random_state=42)
        self.algo.fit(trainset)


class CollabFiltKNNRecommender(CollabFiltRecommender):  # Neighbor-based approach
    MODEL_NAME = "Collaborative-Filtering-KNN"

    def fit(self) -> None:
        reader = Reader(rating_scale=(0, 5))
        # PRISE EN COMPTE DU RATING : 'nb' est également utilisé ici pour
        # calculer les similarités entre articles.
        data = Dataset.load_from_df(self.train_user_interact[["user_id", "article_id", "nb"]], reader)  # noqa: E501
        trainset = data.build_full_trainset()
        # Utilise une approche basée sur les voisins (item-based) : un article
        # est recommandé s'il est similaire à d'autres articles que
        # l'utilisateur a aimés.
        sim_options = {"name": "cosine", "user_based": False}  # Item-based
        self.algo = KNNWithMeans(sim_options=sim_options)
        self.algo.fit(trainset)


"""
### **Modèle Hybride**
L'objectif du modèle hybride est de combiner les forces de plusieurs approches
pour pallier leurs faiblesses respectives. Ici, nous combinons :
- **Filtrage Collaboratif (CF) :** Très bon pour capturer les goûts subtils
  et la sérendipité à partir des interactions collectives (ici, SVDpp).
- **Filtrage Basé sur le Contenu (CB) :** Excellent pour gérer le problème du
  "cold start" (nouveaux articles) et pour recommander des articles de niche.

La combinaison se fait par une **somme pondérée des scores normalisés** des
deux modèles.
"""


class HybridRecommender:  # Combines scores from multiple models
    MODEL_NAME = "Hybrid-Filtering"

    def __init__(
        self,
        data_map: pd.DataFrame,
        i2vec: np.ndarray,
        dic_ri: Dict[Any, int],
        dic_ir: Dict[int, Any],
        items_df: pd.DataFrame,
        cf_weight: float = 0.5,
        cb_weight: float = 0.5,
    ):
        self.train_user_interact: pd.DataFrame = data_map
        self.dic_ir: Dict[int, Any] = dic_ir
        self.dic_ri: Dict[Any, int] = dic_ri
        self.items_embedding: np.ndarray = i2vec
        self.items_df: pd.DataFrame = items_df
        self.cf_weight: float = cf_weight
        self.cb_weight: float = cb_weight
        self.MODEL_NAME = f"Hybrid-W(cf={cf_weight:.2f},cb={cb_weight:.2f})"
        self.cf_model: Optional[Any] = None
        self.cb_model: Optional[Any] = None
        self.pf_model: Optional[Any] = None

    def get_model_name(self) -> str:
        return self.MODEL_NAME

    def fit(
        self, cf_model: Optional[Any] = None, cb_model: Optional[Any] = None, pf_model: Optional[Any] = None
    ) -> None:
        # Entraîne chaque sous-modèle. On peut aussi passer des modèles
        # pré-entraînés pour l'optimisation.
        self.cf_model = cf_model or CollabFiltRecommender(self.train_user_interact)  # Utilise SVDpp par défaut
        self.cf_model.fit()

        self.cb_model = cb_model or ContentBasedTimeDecayRecommender(
            self.train_user_interact,
            self.items_embedding,
            self.dic_ri,
            self.dic_ir,
        )
        self.cb_model.fit()

        self.pf_model = pf_model or PopularityFiltRecommender(self.train_user_interact)
        self.pf_model.fit()

    def recommend_items(self, uid: Any, topn: int = 10) -> pd.DataFrame:
        # Handle cold start users (users not in the training data)
        if uid not in self.train_user_interact["user_id"].unique():
            # Return an empty DataFrame with the expected columns for
            # consistency
            return pd.DataFrame(
                columns=[
                    "article_id",
                    "norm_score_cf",
                    "norm_score_cb",
                    "final_score",
                ]
            )

        items_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id == uid, "article_id"])

        # 1. Obtenir les recommandations complètes de chaque sous-modèle
        reco_cf = self.cf_model.recommend_items(uid, topn=0)
        reco_cb = self.cb_model.recommend_items(uid, topn=0)

        # 2. Normaliser les scores de chaque modèle (entre 0 et 1) pour les
        # rendre comparables.
        # La normalisation Min-Max est une technique courante pour cela.
        if not reco_cf.empty:
            min_pred, max_pred = reco_cf["pred"].min(), reco_cf["pred"].max()
            reco_cf["norm_score"] = (reco_cf["pred"] - min_pred) / (max_pred - min_pred + 1e-5)
        else:
            # Ensure columns exist even if empty
            reco_cf = pd.DataFrame(columns=["article_id", "norm_score"])

        if not reco_cb.empty:
            min_cos = reco_cb["cb_cosine_with_profile"].min()
            max_cos = reco_cb["cb_cosine_with_profile"].max()
            norm_expr = reco_cb["cb_cosine_with_profile"] - min_cos
            norm_denom = max_cos - min_cos + 1e-5
            reco_cb["norm_score"] = norm_expr / norm_denom
        else:
            # Ensure columns exist even if empty
            reco_cb = pd.DataFrame(columns=["article_id", "norm_score"])

        # 3. Fusionner les recommandations sur l'ID de l'article
        reco = pd.merge(
            reco_cf[["article_id", "norm_score"]],
            reco_cb[["article_id", "norm_score"]],
            on="article_id",
            how="outer",
            suffixes=("_cf", "_cb"),
        )
        reco.fillna(0, inplace=True)

        # 4. Calculer le score final comme une somme pondérée et filtrer les
        # articles déjà vus
        reco["final_score"] = (self.cf_weight * reco["norm_score_cf"]) + (self.cb_weight * reco["norm_score_cb"])
        # 5. Trier par score final et filtrer les articles déjà vus
        reco = reco.sort_values("final_score", ascending=False)

        reco = reco[~reco["article_id"].isin(items_to_ignore)]

        return reco.head(topn) if topn > 0 else reco


# ## **6. Définition du Cadre d'Évaluation (`ModelEvaluator`)**
class ModelEvaluator:
    def __init__(
        self,
        data_map: pd.DataFrame,
        data_map_train: pd.DataFrame,
        data_map_test: pd.DataFrame,
        items_df: pd.DataFrame,
        i2vec: np.ndarray,
        dic_ri: Dict[Any, int],
    ):
        self.full_user_interact: pd.DataFrame = data_map
        self.train_user_interact: pd.DataFrame = data_map_train
        self.test_user_interact: pd.DataFrame = data_map_test
        self.full_items: pd.DataFrame = items_df
        self.i2vec: np.ndarray = i2vec  # Nécessaire pour la métrique de diversité
        self.dic_ri: Dict[Any, int] = dic_ri  # Nécessaire pour la métrique de diversité

    def get_random_sample(self, uid: Any, sample_size: int = 100, seed: int = 42) -> set:
        # Récupère tous les articles sur lesquels l'utilisateur a interagi
        # dans l'ensemble de données complet
        user_interacted_items = set(
            self.full_user_interact.loc[self.full_user_interact["user_id"] == uid]["article_id"]
        )
        # Les candidats pour l'échantillonnage négatif sont tous les articles
        # du catalogue filtré moins ceux déjà vus
        all_possible_items = set(self.full_items["article_id"])
        non_clicked_items = all_possible_items - user_interacted_items
        random.seed(seed)
        # S'assure de ne pas essayer de sampler plus d'éléments qu'il n'en existe
        sample_size = min(len(non_clicked_items), sample_size)
        non_clicked_items_sample = set(random.sample(list(non_clicked_items), sample_size))
        return non_clicked_items_sample

    def _verify_hit_top_n(self, iid: Any, recommended_items: Any, topn: int) -> Tuple[int, int]:
        """Vérifie si un item est dans le top-N d'une liste de recommandations."""
        if isinstance(recommended_items, pd.Series):
            recommended_items = recommended_items.values

        try:
            index = next(i for i, c in enumerate(recommended_items) if c == iid)
        except StopIteration:
            index = -1
        hit = int(index != -1 and index < topn)
        return hit, index

    def _calculate_intra_list_similarity(self, recommended_items_ids: List[Any]) -> float:
        """Calcule la diversité d'une liste de recommandations
        (Intra-List Similarity)."""
        if len(recommended_items_ids) < 2:
            return 0.0

        reco_indices = [self.dic_ri[iid] for iid in recommended_items_ids if iid in self.dic_ri]
        if len(reco_indices) < 2:
            return 0.0

        reco_embeddings = self.i2vec[reco_indices]
        similarity_matrix = cosine_similarity(reco_embeddings)
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        return np.mean(similarity_matrix[upper_triangle_indices])

    def evaluate_model_for_user(self, model: Any, uid: Any) -> Dict[str, Any]:
        """Évalue les recommandations pour un seul utilisateur."""
        all_available_item_ids = set(self.full_items["article_id"])
        user_clicks_in_test = self.test_user_interact["user_id"] == uid
        items_are_available = self.test_user_interact["article_id"].isin(all_available_item_ids)
        loc_filter = user_clicks_in_test & items_are_available
        uid_test_clicked_items = set(self.test_user_interact.article_id.loc[loc_filter])  # noqa: E501
        uid_test_nb_clicked_items = len(uid_test_clicked_items)

        if uid_test_nb_clicked_items == 0:
            return {
                "hits@5_count": 0,
                "hits@10_count": 0,
                "interacted_count": 0,
                "recall@5": 0,
                "recall@10": 0,
                "reciprocal_rank": 0,
                "intra_list_similarity": 0,
            }  # noqa: E501

        hits_at_5_count = 0
        hits_at_10_count = 0
        sum_reciprocal_rank = 0

        for iid in uid_test_clicked_items:
            uid_non_clicked_items_sample = self.get_random_sample(
                uid, sample_size=100, seed=int(iid) % (2**32)  # noqa: E501
            )
            items_to_rank = list(uid_non_clicked_items_sample.union({iid}))

            uid_reco = model.recommend_items(uid, topn=0)
            filt_uid_reco = uid_reco[uid_reco["article_id"].isin(items_to_rank)]

            hit_at_5, rank = self._verify_hit_top_n(iid, filt_uid_reco["article_id"], 5)  # noqa: F841
            hits_at_5_count += hit_at_5
            hit_at_10, rank = self._verify_hit_top_n(iid, filt_uid_reco["article_id"], 10)
            hits_at_10_count += hit_at_10
            if rank != -1:
                sum_reciprocal_rank += 1 / (rank + 1)

        general_recos = model.recommend_items(uid, topn=10)
        top_10_reco_ids = general_recos["article_id"].tolist()
        intra_list_similarity = self._calculate_intra_list_similarity(top_10_reco_ids)
        # The recall is calculated over all test items for the user,
        # which is correct.
        # The reciprocal rank is also correctly averaged.
        recall_at_5 = hits_at_5_count / float(uid_test_nb_clicked_items)
        recall_at_10 = hits_at_10_count / float(uid_test_nb_clicked_items)
        avg_reciprocal_rank = sum_reciprocal_rank / uid_test_nb_clicked_items
        return {
            "hits@5_count": hits_at_5_count,
            "hits@10_count": hits_at_10_count,
            "interacted_count": uid_test_nb_clicked_items,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            "reciprocal_rank": avg_reciprocal_rank,
            "intra_list_similarity": intra_list_similarity,
        }

    def evaluate_model(self, model: Any, breaknb: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Évalue un modèle sur un échantillon d'utilisateurs du jeu de test."""
        people_metrics = []

        # On ne peut évaluer que les utilisateurs présents dans le jeu de test.
        test_user_ids = set(self.test_user_interact.user_id.unique())

        # Pour les modèles 'implicit', on ne peut évaluer que les utilisateurs
        # qu'ils ont vus pendant l'entraînement.
        if hasattr(model, "user_inv_map"):
            model_users = set(model.user_inv_map.keys())
            test_user_ids = test_user_ids.intersection(model_users)
        list_uid = list(test_user_ids)
        random.shuffle(list_uid)

        # Boucle sur un échantillon d'utilisateurs pour accélérer l'évaluation
        for idx, uid in enumerate(tqdm(list_uid, desc=f"Evaluating {model.get_model_name()}")):  # noqa: E501
            if idx >= breaknb:
                break
            person_metrics = self.evaluate_model_for_user(model, uid)
            person_metrics["uid"] = uid
            people_metrics.append(person_metrics)

        results = pd.DataFrame(people_metrics).sort_values("interacted_count", ascending=False)

        # Agrégation des métriques au niveau global
        total_hits_5 = results["hits@5_count"].sum()
        total_hits_10 = results["hits@10_count"].sum()
        total_interacted = float(results["interacted_count"].sum())
        global_recall_at_5 = total_hits_5 / total_interacted
        global_recall_at_10 = total_hits_10 / total_interacted
        global_mrr = results["reciprocal_rank"].mean()
        global_ils = results["intra_list_similarity"].mean()

        global_metrics = {
            "modelName": model.get_model_name(),
            "recall@5": global_recall_at_5,
            "recall@10": global_recall_at_10,
            "mrr": global_mrr,
            "diversity_ils": global_ils,
        }
        return global_metrics, results
