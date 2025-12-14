import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVDpp
from tqdm import tqdm


class PopularityFiltRecommender:
    MODEL_NAME = "Popularity-Filtering"

    def __init__(self, data_map):
        self.train_user_interact = data_map
        self.raw_reco = None

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self):
        self.raw_reco = (
            self.train_user_interact.groupby("article_id")["nb"].sum().sort_values(ascending=False).reset_index()
        )

    def recommend_items(self, uid, topn=5):
        return self.raw_reco.head(topn) if topn > 0 else self.raw_reco


class ContentBasedRecommender:
    MODEL_NAME = "Content-Based"

    def __init__(self, data_map, i2vec, dic_ri, dic_ir):
        self.dic_ir = dic_ir
        self.dic_ri = dic_ri
        self.items_embedding = i2vec
        self.train_user_interact = data_map
        self.user_profiles = {}

    def _build_users_profile(self, uid, click_df, emb_matrix, dic_ri):
        click_uid_df = click_df.loc[click_df.user_id == uid]
        user_item_profiles = np.array([emb_matrix[dic_ri[iid]] for iid in click_uid_df["article_id"] if iid in dic_ri])
        if len(user_item_profiles) == 0:
            return np.zeros((1, self.items_embedding.shape[1]))
        user_item_strengths = np.array(click_uid_df["nb"]).reshape(-1, 1)
        sum_of_strengths = np.sum(user_item_strengths)
        if sum_of_strengths == 0:
            return np.zeros((1, self.items_embedding.shape[1]))
        user_item_strengths_weighted_avg = np.sum(user_item_profiles * user_item_strengths, axis=0) / sum_of_strengths
        return preprocessing.normalize(user_item_strengths_weighted_avg.reshape(1, -1))

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self):
        for uid in tqdm(self.train_user_interact.user_id.unique(), desc="Building User Profiles"):
            self.user_profiles[uid] = self._build_users_profile(
                uid, self.train_user_interact, self.items_embedding, self.dic_ri
            )

    def recommend_items(self, uid, topn=10, items_to_filter=None):
        if uid not in self.user_profiles:
            return pd.DataFrame(columns=["article_id", "cb_cosine_with_profile"])
        cosine_similarities = cosine_similarity(self.user_profiles[uid], self.items_embedding)
        similar_indices = cosine_similarities.argsort().flatten()[::-1]
        similar_items = [(self.dic_ir[i], cosine_similarities[0, i]) for i in similar_indices if i in self.dic_ir]
        items_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id == uid].article_id)
        reco = pd.DataFrame(similar_items, columns=["article_id", "cb_cosine_with_profile"])
        reco = reco[~reco.article_id.isin(items_to_ignore)]
        if items_to_filter is not None:
            reco = reco[reco["article_id"].isin(items_to_filter)]
        return reco.head(topn) if topn > 0 else reco


class ContentBasedTimeDecayRecommender(ContentBasedRecommender):
    MODEL_NAME = "Content-Based-Time-Decay"

    def __init__(self, data_map, i2vec, dic_ri, dic_ir, decay_rate=0.05):
        super().__init__(data_map, i2vec, dic_ri, dic_ir)
        self.decay_rate = decay_rate
        self.MODEL_NAME = f"Content-Based-Time-Decay(λ={decay_rate})"

    def _build_users_profile(self, uid, click_df, emb_matrix, dic_ri):
        click_uid_df = click_df.loc[click_df.user_id == uid]
        if click_uid_df.empty:
            return np.zeros((1, self.items_embedding.shape[1]))
        latest_timestamp = click_uid_df["click_timestamp"].max()
        time_diff_days = (latest_timestamp - click_uid_df["click_timestamp"]) / (3600 * 24)
        time_decay_weight = np.exp(-self.decay_rate * time_diff_days)
        final_weights = (click_uid_df["nb"] * time_decay_weight).values.reshape(-1, 1)
        user_item_profiles = np.array([emb_matrix[dic_ri[iid]] for iid in click_uid_df["article_id"] if iid in dic_ri])
        sum_of_weights = np.sum(final_weights)
        if sum_of_weights == 0:
            return np.zeros((1, self.items_embedding.shape[1]))
        weighted_avg_profile = np.sum(user_item_profiles * final_weights, axis=0) / sum_of_weights
        return preprocessing.normalize(weighted_avg_profile.reshape(1, -1))


class CollabFiltRecommender:
    MODEL_NAME = "Collaborative-Filtering-SVDpp"

    def __init__(self, data_map):
        self.train_user_interact = data_map
        self.algo = None

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self):
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(self.train_user_interact[["user_id", "article_id", "nb"]], reader)
        trainset = data.build_full_trainset()
        self.algo = SVDpp(n_epochs=20, lr_all=0.007, reg_all=0.1, random_state=42)
        self.algo.fit(trainset)

    def recommend_items(self, uid, topn=5):
        iid_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id == uid].article_id)
        all_items = set(self.train_user_interact.article_id)
        items2pred_ids = all_items - iid_to_ignore
        try:
            # Vérifier si l'utilisateur est connu du modèle
            self.algo.trainset.to_inner_uid(uid)  # Lève une ValueError si l'utilisateur est inconnu
        except ValueError:
            # L'utilisateur est inconnu (cold start), retourner un DataFrame vide
            return pd.DataFrame(columns=["article_id", "pred"])

        predictions = [self.algo.predict(uid=uid, iid=iid) for iid in items2pred_ids]
        recs = [(pred.iid, pred.est) for pred in predictions]
        recommendations_df = pd.DataFrame(recs, columns=["article_id", "pred"]).sort_values(by="pred", ascending=False)
        return recommendations_df.head(topn) if topn > 0 else recommendations_df


class HybridRecommender:
    MODEL_NAME = "Hybrid-Filtering"

    def __init__(self, data_map, i2vec, dic_ri, dic_ir, items_df, cf_weight=0.5, cb_weight=0.5):
        self.train_user_interact = data_map
        self.dic_ir = dic_ir
        self.dic_ri = dic_ri
        self.items_embedding = i2vec
        self.items_df = items_df
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.MODEL_NAME = f"Hybrid-W(cf={cf_weight:.2f},cb={cb_weight:.2f})"
        self.cf_model = None
        self.cb_model = None
        self.pf_model = None

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, cf_model=None, cb_model=None, pf_model=None):
        self.cf_model = cf_model or CollabFiltRecommender(self.train_user_interact)
        self.cf_model.fit()
        self.cb_model = cb_model or ContentBasedTimeDecayRecommender(
            self.train_user_interact, self.items_embedding, self.dic_ri, self.dic_ir
        )
        self.cb_model.fit()
        self.pf_model = pf_model or PopularityFiltRecommender(self.train_user_interact)
        self.pf_model.fit()

    def recommend_items(self, uid, topn=10):
        # Gérer les nouveaux utilisateurs (cold start)
        if uid not in self.train_user_interact["user_id"].unique():
            # Pour un nouvel utilisateur, retourner les articles les plus populaires
            return self.pf_model.recommend_items(uid, topn=topn)

        items_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id == uid, "article_id"])

        # 1. Obtenir les recommandations de chaque sous-modèle
        reco_cf = self.cf_model.recommend_items(uid, topn=0)
        reco_cb = self.cb_model.recommend_items(uid, topn=0)

        # 2. Normaliser les scores
        if not reco_cf.empty:
            reco_cf["norm_score"] = (reco_cf["pred"] - reco_cf["pred"].min()) / (
                reco_cf["pred"].max() - reco_cf["pred"].min() + 1e-5
            )
        else:
            reco_cf = pd.DataFrame(columns=["article_id", "norm_score"])

        if not reco_cb.empty:
            reco_cb["norm_score"] = (reco_cb["cb_cosine_with_profile"] - reco_cb["cb_cosine_with_profile"].min()) / (
                reco_cb["cb_cosine_with_profile"].max() - reco_cb["cb_cosine_with_profile"].min() + 1e-5
            )
        else:
            reco_cb = pd.DataFrame(columns=["article_id", "norm_score"])

        # 3. Fusionner les recommandations
        reco = pd.merge(
            reco_cf[["article_id", "norm_score"]],
            reco_cb[["article_id", "norm_score"]],
            on="article_id",
            how="outer",
            suffixes=("_cf", "_cb"),
        )
        reco.fillna(0, inplace=True)

        # 4. Calculer le score final pondéré
        reco["final_score"] = (self.cf_weight * reco["norm_score_cf"]) + (self.cb_weight * reco["norm_score_cb"])

        # 5. Trier et filtrer les articles déjà vus
        reco = reco.sort_values("final_score", ascending=False)
        reco = reco[~reco["article_id"].isin(items_to_ignore)]

        return reco.head(topn) if topn > 0 else reco
