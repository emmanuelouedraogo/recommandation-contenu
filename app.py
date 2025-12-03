# app.py

import streamlit as st
import pandas as pd
import os
import requests
from azure.storage.blob import BlobServiceClient
from io import StringIO

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recommandation de Contenu",
    page_icon="📚",
    layout="wide"
)

# --- Constantes ---
AZURE_CONNECTION_STRING = st.secrets.get("AZURE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = "reco-data"
USERS_BLOB_NAME = "users.csv"
ARTICLES_BLOB_NAME = "articles_metadata.csv" # Nom du blob pour les articles
CLICKS_BLOB_NAME = "clicks_sample.csv" # Nom du blob pour les interactions
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"

# Utilise l'URL de l'API depuis les secrets, crucial pour le déploiement.
# Pas de valeur par défaut pour forcer la configuration en production.
API_URL = st.secrets.get("API_URL")

# --- Fonctions de Chargement des Données ---

@st.cache_resource
def get_blob_service_client():
    """Crée un client de service blob. Mis en cache pour la performance."""
    if not AZURE_CONNECTION_STRING:
        st.error("La chaîne de connexion Azure n'est pas configurée dans les secrets !")
        return None
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

def load_df_from_blob(blob_name):
    """Charge un DataFrame depuis un blob CSV."""
    blob_service_client = get_blob_service_client()
    if not blob_service_client: return pd.DataFrame()
    
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    try:
        downloader = blob_client.download_blob(encoding='utf-8')
        blob_data = downloader.readall()
        return pd.read_csv(StringIO(blob_data))
    except Exception as e:
        st.warning(f"Le blob '{blob_name}' est vide ou n'existe pas. Un nouveau sera créé. Erreur: {e}")
        return pd.DataFrame() # Retourne un DF vide pour gérer tous les cas

def save_df_to_blob(df, blob_name):
    """Sauvegarde un DataFrame dans un blob CSV."""
    blob_service_client = get_blob_service_client()

    output = StringIO()
    df.to_csv(output, index=False)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    blob_client.upload_blob(output.getvalue(), overwrite=True)

# --- Initialisation et Chargement ---
# Chargement des données
users_df = load_df_from_blob(USERS_BLOB_NAME)
articles_df = load_df_from_blob(ARTICLES_BLOB_NAME)

def add_interaction(user_id, article_id, rating):
    """Charge les interactions, ajoute une nouvelle note et sauvegarde."""
    clicks_df = load_df_from_blob(CLICKS_BLOB_NAME)
    
    new_interaction = pd.DataFrame([{
        'user_id': user_id,
        'article_id': article_id,
        'click_timestamp': int(pd.Timestamp.now().timestamp()),
        'nb': rating # 'nb' est utilisé comme score d'interaction dans les modèles
    }])
    
    updated_clicks_df = pd.concat([clicks_df, new_interaction], ignore_index=True)
    save_df_to_blob(updated_clicks_df, CLICKS_BLOB_NAME)
    st.toast(f"Merci pour votre note de {rating}/5 !", icon="⭐")

def update_interaction(user_id, article_id, new_rating):
    """Met à jour la note la plus récente pour un article donné par un utilisateur."""
    clicks_df = load_df_from_blob(CLICKS_BLOB_NAME)
    
    # Trouve l'index de la dernière interaction pour ce couple utilisateur/article
    user_article_interactions = clicks_df[(clicks_df['user_id'] == user_id) & (clicks_df['article_id'] == article_id)]
    if not user_article_interactions.empty:
        latest_interaction_index = user_article_interactions['click_timestamp'].idxmax()
        
        # Met à jour la note et le timestamp
        clicks_df.loc[latest_interaction_index, 'nb'] = new_rating
        clicks_df.loc[latest_interaction_index, 'click_timestamp'] = int(pd.Timestamp.now().timestamp())
        save_df_to_blob(clicks_df, CLICKS_BLOB_NAME)
        st.toast(f"Votre note a été mise à jour à {new_rating}/5 !", icon="👍")

# --- Fonctions du Système de Recommandation ---

def get_recommendations(user_id):
    """
    Appelle l'API FastAPI pour obtenir les recommandations.
    """
    if not API_URL:
        st.error("L'URL de l'API n'est pas configurée dans les secrets Streamlit (API_URL).")
        return None

    if user_id not in users_df['user_id'].unique():
        st.error("Cet identifiant utilisateur n'existe pas. Veuillez créer un compte.")
        return None
    
    with st.spinner('Recherche de vos recommandations...'):
        try:
            # L'API Azure Function déployée attend une requête GET avec un paramètre d'URL.
            # L'URL de l'endpoint est "/api/recommend".
            headers = {'Accept': 'application/json'}
            response = requests.get(f"{API_URL}/api/recommend", params={"user_id": user_id}, headers=headers, timeout=20)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            
            data = response.json()
            
            # L'API renvoie une liste de dictionnaires que nous convertissons en DataFrame.
            return pd.DataFrame(data)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter le service de recommandation. Vérifiez que l'URL de l'API est correcte et que le service est démarré. (Erreur: {e})")
            return None

# --- Interface Streamlit ---

st.title("📚 Système de Recommandation de Contenu")

# Menu dans la barre latérale
menu = ["Recommandations", "Mon Historique", "Performance du Modèle", "Créer un compte", "Ajouter un article"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Recommandations":
    st.header("Obtenez vos recommandations")
    
    # Affiche la liste des utilisateurs pour faciliter le test
    if not users_df.empty:
        st.info("Utilisateurs existants (pour tester) :")
        st.dataframe(users_df, width='stretch')

    user_id_input = st.text_input("Entrez votre identifiant utilisateur :")

    if st.button("Obtenir mes recommandations"):
        if user_id_input:
            try:
                # Tente de convertir en int si vos IDs sont numériques
                user_id = int(user_id_input)
                recommendations = get_recommendations(user_id)
                
                if recommendations is not None and not recommendations.empty:
                    # Fusionner avec les métadonnées pour obtenir les titres et contenus
                    reco_details = recommendations.merge(articles_df, on='article_id', how='left')
                    
                    st.success(f"Bienvenue, Utilisateur {user_id} ! Voici vos recommandations personnalisées :")
                    
                    # CORRECTION: Itérer sur `reco_details` pour avoir accès aux titres et contenus.
                    for _, row in reco_details.iterrows():
                        with st.container():
                            # Utiliser les données fusionnées.
                            st.subheader(f"{row.get('title', 'Titre inconnu')}")
                            st.caption(f"Score de recommandation : {row.get('final_score', 0):.2f} | ID Article : {row['article_id']}")
                            st.write(str(row.get('content', 'Contenu non disponible.'))[:250] + "...") # Affiche un aperçu
                            
                            # Section pour noter l'article
                            rating = st.slider("Notez cet article :", 1, 5, 3, key=f"rating_{row['article_id']}")
                            if st.button("Envoyer ma note", key=f"btn_{row['article_id']}"):
                                add_interaction(user_id, row['article_id'], rating)
                            st.divider()
                elif recommendations is not None:
                     st.warning("Il n'y a pas assez d'articles à recommander pour le moment.")

            except ValueError:
                st.error("L'identifiant utilisateur doit être un nombre.")
        else:
            st.warning("Veuillez entrer un identifiant utilisateur.")

elif choice == "Mon Historique":
    st.header("Historique de vos notations")

    user_id_history = st.text_input("Entrez votre identifiant utilisateur pour voir votre historique :", key="history_user_id")

    if st.button("Afficher mon historique", key="history_btn"):
        if user_id_history:
            try:
                user_id = int(user_id_history)
                
                if user_id not in users_df['user_id'].unique():
                    st.error("Cet identifiant utilisateur n'existe pas.")
                else:
                    clicks_df = load_df_from_blob(CLICKS_BLOB_NAME)
                    
                    if clicks_df.empty:
                        st.warning("Aucune notation n'a encore été enregistrée dans le système.")
                    else:
                        user_history_df = clicks_df[clicks_df['user_id'] == user_id]
                        
                        if user_history_df.empty:
                            st.info("Vous n'avez encore noté aucun article.")
                        else:
                            # S'assurer que chaque article n'apparaît qu'une fois (le plus récent)
                            user_history_df = user_history_df.sort_values('click_timestamp').drop_duplicates(subset=['user_id', 'article_id'], keep='last')
                            history_details = user_history_df.merge(articles_df, on='article_id', how='left')
                            history_details = history_details.sort_values(by='click_timestamp', ascending=False)
                            
                            st.subheader(f"Articles que vous avez notés, Utilisateur {user_id} :")
                            for _, row in history_details.iterrows():
                                col1, col2 = st.columns([3, 2])
                                with col1:
                                    st.markdown(f"**{row.get('title', 'Titre inconnu')}**")
                                    st.caption(f"Dernière modification : {pd.to_datetime(row['click_timestamp'], unit='s').strftime('%Y-%m-%d %H:%M')}")
                                with col2:
                                    new_rating = st.number_input("Votre note", min_value=1, max_value=5, value=int(row.get('nb', 0)), key=f"update_rating_{row['article_id']}")
                                    if st.button("Modifier la note", key=f"update_btn_{row['article_id']}"):
                                        update_interaction(user_id, row['article_id'], new_rating)
                                st.divider()
            except ValueError:
                st.error("L'identifiant utilisateur doit être un nombre.")

elif choice == "Performance du Modèle":
    st.header("Historique et Performance des Entraînements")

    log_df = load_df_from_blob(TRAINING_LOG_BLOB_NAME)

    if log_df.empty:
        st.info("Aucun historique d'entraînement n'a encore été enregistré.")
    else:
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        log_df = log_df.sort_values('timestamp').reset_index(drop=True)

        st.subheader("Évolution de la Précision@10")
        st.line_chart(log_df, x='timestamp', y='precision_at_10')

        st.subheader("Précision@10 en fonction du nombre d'interactions")
        st.line_chart(log_df, x='click_count', y='precision_at_10')

        st.subheader("Détail des entraînements")
        st.dataframe(log_df, width='stretch')

elif choice == "Créer un compte":
    st.header("Créez votre compte")
    
    if st.button("Créer un nouvel identifiant"):
        # Génère un nouvel ID unique (plus robuste qu'un simple incrément)
        new_user_id = (int(users_df['user_id'].max()) + 1) if not users_df.empty else 1
        
        # Ajoute au DataFrame et sauvegarde
        new_user_df = pd.DataFrame([{'user_id': new_user_id}])
        users_df = pd.concat([users_df, new_user_df], ignore_index=True)
        save_df_to_blob(users_df, USERS_BLOB_NAME)
        
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

        if submit_button and article_title and article_content:
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
                # Note: La sauvegarde d'un nouvel article ne mettra pas à jour le fichier public via l'URL.
                # Il faudrait un mécanisme de ré-upload pour cela.
                
                st.success(f"L'article '{article_title}' a été ajouté avec succès !")
            else:
                st.warning("Veuillez remplir le titre et le contenu.")
    
    st.divider()
    st.subheader("Articles actuels dans la base de données")
    st.dataframe(articles_df, width='stretch')
