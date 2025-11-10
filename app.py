# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, SVDpp, KNNWithMeans

# --- Définition des Classes de Modèles (inspiré de reco_model_script.py) ---
# NOTE: Pour que pickle.load() fonctionne, les définitions de classes doivent être disponibles.
# Idéalement, ces classes seraient dans un fichier models.py importé ici et dans le script d'entraînement.

class PopularityFiltRecommender:
    def __init__(self, data_map):
        self.train_user_interact = data_map
        self.raw_reco = None
    def fit(self):
        self.raw_reco = self.train_user_interact.groupby('article_id')['nb'].sum().sort_values(ascending=False).reset_index()
    def recommend_items(self, uid, topn=5):
        return self.raw_reco.head(topn) if topn > 0 else self.raw_reco

class ContentBasedRecommender:
    def __init__(self, data_map, i2vec, dic_ri, dic_ir):
        self.dic_ir = dic_ir
        self.dic_ri = dic_ri
        self.items_embedding = i2vec
        self.train_user_interact = data_map
        self.user_profiles = {}
    def _build_users_profile(self, uid, click_df, emb_matrix, dic_ri):
        # Implémentation de la construction de profil (peut être vide si non utilisée directement)
        pass
    def fit(self):
        pass
    def recommend_items(self, uid, topn=10, items_to_filter=None):
        # Implémentation de la recommandation (peut être vide si non utilisée directement)
        return pd.DataFrame(columns=['article_id'])

class ContentBasedTimeDecayRecommender(ContentBasedRecommender):
    def __init__(self, data_map, i2vec, dic_ri, dic_ir, decay_rate=0.05):
        super().__init__(data_map, i2vec, dic_ri, dic_ir)
        self.decay_rate = decay_rate
    def _build_users_profile(self, uid, click_df, emb_matrix, dic_ri):
        click_uid_df = click_df.loc[click_df.user_id == uid]
        if click_uid_df.empty: return np.zeros((1, self.items_embedding.shape[1]))
        latest_timestamp = click_uid_df['click_timestamp'].max()
        time_diff_days = (latest_timestamp - click_uid_df['click_timestamp']) / (3600 * 24)
        time_decay_weight = np.exp(-self.decay_rate * time_diff_days)
        final_weights = (click_uid_df['nb'] * time_decay_weight).values.reshape(-1, 1)
        user_item_profiles = np.array([emb_matrix[dic_ri[iid]] for iid in click_uid_df['article_id'] if iid in dic_ri])
        sum_of_weights = np.sum(final_weights)
        if sum_of_weights == 0: return np.zeros((1, self.items_embedding.shape[1]))
        weighted_avg_profile = np.sum(user_item_profiles * final_weights, axis=0) / sum_of_weights
        return preprocessing.normalize(weighted_avg_profile.reshape(1, -1))
    def fit(self):
        for uid in self.train_user_interact.user_id.unique():
            self.user_profiles[uid] = self._build_users_profile(uid, self.train_user_interact, self.items_embedding, self.dic_ri)
    def recommend_items(self, uid, topn=10, items_to_filter=None):
        if uid not in self.user_profiles: return pd.DataFrame(columns=['article_id', 'cb_cosine_with_profile'])
        cosine_similarities = cosine_similarity(self.user_profiles[uid], self.items_embedding)
        similar_indices = cosine_similarities.argsort().flatten()[::-1]
        similar_items = [(self.dic_ir[i], cosine_similarities[0,i]) for i in similar_indices if i in self.dic_ir]
        items_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id==uid].article_id)
        reco = pd.DataFrame(similar_items, columns=['article_id', 'cb_cosine_with_profile'])
        reco = reco[~reco.article_id.isin(items_to_ignore)]
        return reco.head(topn) if topn > 0 else reco

class CollabFiltRecommender:
    def __init__(self, data_map):
        self.train_user_interact = data_map
        self.algo = None
    def fit(self):
        reader = Reader(rating_scale=(0,5))
        data = Dataset.load_from_df(self.train_user_interact[['user_id','article_id','nb']], reader)
        trainset = data.build_full_trainset()
        self.algo = SVDpp(n_epochs=20, lr_all=0.007, reg_all=0.1, random_state=42)
        self.algo.fit(trainset)
    def recommend_items(self, uid, topn=5):
        iid_to_ignore=set(self.train_user_interact.loc[self.train_user_interact.user_id==uid].article_id)
        all_items = set(self.train_user_interact.article_id)
        items2pred_ids = all_items - iid_to_ignore
        predictions = [self.algo.predict(uid=uid, iid=iid) for iid in items2pred_ids]
        recs = [(pred.iid, pred.est) for pred in predictions]
        recommendations_df = pd.DataFrame(recs, columns=['article_id', 'pred']).sort_values(by='pred', ascending=False)
        return recommendations_df.head(topn) if topn > 0 else recommendations_df

class HybridRecommender:
    def __init__(self, data_map, i2vec, dic_ri, dic_ir, items_df, cf_weight=0.5, cb_weight=0.5):
        self.train_user_interact = data_map; self.dic_ir = dic_ir; self.dic_ri = dic_ri
        self.items_embedding = i2vec; self.items_df = items_df; self.cf_weight = cf_weight
        self.cb_weight = cb_weight; self.cf_model = None; self.cb_model = None; self.pf_model = None
    def fit(self, cf_model=None, cb_model=None, pf_model=None):
        self.cf_model = cf_model; self.cb_model = cb_model; self.pf_model = pf_model
    def recommend_items(self, uid, topn=10):
        if uid not in self.train_user_interact['user_id'].unique(): return self.pf_model.recommend_items(uid, topn)
        items_to_ignore = set(self.train_user_interact.loc[self.train_user_interact.user_id == uid, 'article_id'])
        reco_cf = self.cf_model.recommend_items(uid, topn=0); reco_cb = self.cb_model.recommend_items(uid, topn=0)
        if not reco_cf.empty: reco_cf['norm_score'] = (reco_cf['pred'] - reco_cf['pred'].min()) / (reco_cf['pred'].max() - reco_cf['pred'].min() + 1e-5)
        else: reco_cf = pd.DataFrame(columns=['article_id', 'norm_score'])
        if not reco_cb.empty: reco_cb['norm_score'] = (reco_cb['cb_cosine_with_profile'] - reco_cb['cb_cosine_with_profile'].min()) / (reco_cb['cb_cosine_with_profile'].max() - reco_cb['cb_cosine_with_profile'].min() + 1e-5)
        else: reco_cb = pd.DataFrame(columns=['article_id', 'norm_score'])
        reco = pd.merge(reco_cf[['article_id', 'norm_score']], reco_cb[['article_id', 'norm_score']], on='article_id', how='outer', suffixes=('_cf', '_cb'))
        reco.fillna(0, inplace=True); reco['final_score'] = (self.cf_weight * reco['norm_score_cf']) + (self.cb_weight * reco['norm_score_cb'])
        reco = reco.sort_values('final_score', ascending=False); reco = reco[~reco['article_id'].isin(items_to_ignore)]
        return reco.head(topn) if topn > 0 else reco

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recommandation de Contenu",
    page_icon="📚",
    layout="wide"
)

# --- Fonctions de Chargement des Données ---

@st.cache_resource
def load_csv_data(file_path, columns):
    """Charge un fichier CSV ou en crée un nouveau. Mise en cache."""
    # S'assure que le dossier parent existe
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        return df
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(model_path):
    """Charge le modèle de recommandation sauvegardé. Mise en cache."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            return model
    return None

def save_csv_data(df, file_path):
    """Sauvegarde un DataFrame dans un fichier CSV."""
    df.to_csv(file_path, index=False)

# --- Initialisation et Chargement ---

USERS_FILE = 'data/users.csv'
ARTICLES_FILE = 'data/articles_metadata.csv'
MODEL_FILE = 'hybrid_recommender_pipeline.pkl'

# Chargement des données
users_df = load_csv_data(USERS_FILE, columns=['user_id'])
articles_df = load_csv_data(ARTICLES_FILE, columns=['article_id', 'title', 'content', 'category_id', 'created_at_ts'])
model = load_model(MODEL_FILE)

# --- Fonctions du Système de Recommandation ---

def get_recommendations(user_id):
    """
    Orchestre le type de recommandation à fournir.
    """
    if model is None:
        st.error("Le modèle de recommandation n'a pas pu être chargé. Veuillez vérifier le fichier `save/hybrid_recommender_pipeline.pkl`.")
        return None

    if user_id not in users_df['user_id'].unique():
        st.error("Cet identifiant utilisateur n'existe pas. Veuillez créer un compte.")
        return None

    # Vérifie si l'utilisateur est un "cold start" (présent dans users_df mais pas dans les données d'entraînement du modèle)
    is_cold_start = user_id not in model.train_user_interact['user_id'].unique()

    # Cas 1 : Utilisateur "Cold Start" (aucun historique de consultation dans le modèle)
    if is_cold_start:
        st.success("Nouvel utilisateur ! Voici les 5 articles les plus populaires :")
        # On se rabat sur le modèle de popularité (pf_model) qui est une composante du pipeline hybride.
        if hasattr(model, 'pf_model') and model.pf_model is not None:
            reco_ids_df = model.pf_model.recommend_items(user_id, topn=5)
            reco_ids = reco_ids_df['article_id'].tolist()
        else:
            st.warning("Le sous-modèle de popularité n'est pas disponible. Impossible de générer des recommandations.")
            return pd.DataFrame()
    # Cas 2 : Utilisateur avec un historique
    else:
        st.success(f"Bienvenue, utilisateur {user_id} ! Voici vos recommandations personnalisées :")
        # Utilise la méthode principale du modèle hybride pour obtenir des recommandations personnalisées.
        reco_ids_df = model.recommend_items(user_id, topn=5)
        reco_ids = reco_ids_df['article_id'].tolist()

    # Récupère les détails des articles recommandés
    if reco_ids:
        recommendations = articles_df[articles_df['article_id'].isin(reco_ids)].copy()
        # Conserve l'ordre des recommandations
        recommendations['sort_order'] = recommendations['article_id'].apply(lambda x: reco_ids.index(x))
        return recommendations.sort_values('sort_order')
    return pd.DataFrame()

# --- Interface Streamlit ---

st.title("📚 Système de Recommandation de Contenu")

# Menu dans la barre latérale
menu = ["Recommandations", "Créer un compte", "Ajouter un article"]
choice = st.sidebar.selectbox("Menu", menu)

if model is None:
    st.error("Le modèle de recommandation n'est pas disponible. L'application ne peut pas fonctionner.")
    st.stop()


if choice == "Recommandations":
    st.header("Obtenez vos recommandations")
    
    # Affiche la liste des utilisateurs pour faciliter le test
    if not users_df.empty:
        st.write("Utilisateurs existants :")
        st.dataframe(users_df, use_container_width=True)

    user_id_input = st.text_input("Entrez votre identifiant utilisateur :")

    if st.button("Obtenir mes recommandations"):
        if user_id_input:
            try:
                # Tente de convertir en int si vos IDs sont numériques
                user_id = int(user_id_input)
                recommendations = get_recommendations(user_id)
                if recommendations is not None and not recommendations.empty:
                    for _, row in recommendations.iterrows():
                        with st.container():
                            st.subheader(f"{row['title']} (Article ID: {row['article_id']})")
                            st.write(row['content'][:200] + "...") # Affiche un aperçu
                            st.divider()
                elif recommendations is not None:
                     st.warning("Il n'y a pas assez d'articles à recommander pour le moment.")

            except ValueError:
                st.error("L'identifiant utilisateur doit être un nombre.")
        else:
            st.warning("Veuillez entrer un identifiant utilisateur.")

elif choice == "Créer un compte":
    st.header("Créez votre compte")
    
    if st.button("Créer un nouvel identifiant"):
        # Génère un nouvel ID unique (plus robuste qu'un simple incrément)
        new_user_id = int(users_df['user_id'].max() + 1) if not users_df.empty else 1
        
        # Ajoute au DataFrame et sauvegarde
        new_user_df = pd.DataFrame([{'user_id': new_user_id}])
        users_df = pd.concat([users_df, new_user_df], ignore_index=True)
        save_csv_data(users_df, USERS_FILE)
        
        st.success(f"Votre nouveau compte a été créé avec succès ! Votre identifiant est :")
        st.code(new_user_id, language='text')
        st.info("Vous pouvez maintenant utiliser cet identifiant dans la section 'Recommandations'.")

elif choice == "Ajouter un article":
    st.header("Ajouter un nouvel article ou livre")

    with st.form(key="article_form", clear_on_submit=True):
        article_title = st.text_input("Titre de l'article/livre")
        article_category = st.number_input("ID de la catégorie", min_value=0, step=1)
        article_content = st.text_area("Contenu")
        submit_button = st.form_submit_button(label="Ajouter à la base de données")

        if submit_button:
            if article_title and article_content:
                # Génère un ID unique pour l'article
                new_article_id = int(articles_df['article_id'].max() + 1) if not articles_df.empty else 1
                
                new_article = pd.DataFrame([{
                    'article_id': new_article_id,
                    'title': article_title,
                    'content': article_content,
                    'category_id': article_category,
                    'created_at_ts': int(pd.Timestamp.now().timestamp())
                }])
                
                articles_df = pd.concat([articles_df, new_article], ignore_index=True)
                save_csv_data(articles_df, ARTICLES_FILE)
                
                st.success(f"L'article '{article_title}' a été ajouté avec succès !")
            else:
                st.warning("Veuillez remplir le titre et le contenu.")

    st.divider()
    st.subheader("Articles actuels dans la base de données")
    st.dataframe(articles_df, use_container_width=True)
