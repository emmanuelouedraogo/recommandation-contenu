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
USERS_BLOB_NAME = 'users.csv'
URL_ARTICLES = "https://recoappstorage123.blob.core.windows.net/reco-data/articles_metadata.csv"
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

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
        downloader = blob_client.download_blob(max_connections=1, encoding='utf-8')
        blob_data = downloader.readall()
        return pd.read_csv(StringIO(blob_data))
    except Exception as e:
        st.warning(f"Le blob '{blob_name}' est vide ou n'existe pas. Un nouveau sera créé. Erreur: {e}")
        return pd.DataFrame(columns=['user_id'] if 'user' in blob_name else ['article_id', 'title', 'content', 'category_id', 'created_at_ts'])

@st.cache_data
def load_articles_from_url(url):
    """Charge le DataFrame des articles depuis une URL publique."""
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Impossible de charger les articles depuis l'URL. Erreur: {e}")
        return pd.DataFrame()

def save_df_to_blob(df, blob_name):
    """Sauvegarde un DataFrame dans un blob CSV."""
    blob_service_client = get_blob_service_client()
    if not blob_service_client: return

    output = StringIO()
    df.to_csv(output, index=False)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    blob_client.upload_blob(output.getvalue(), overwrite=True)

# --- Initialisation et Chargement ---
# Chargement des données
users_df = load_df_from_blob(USERS_BLOB_NAME)
articles_df = load_articles_from_url(URL_ARTICLES)

# --- Fonctions du Système de Recommandation ---

def get_recommendations(user_id):
    """
    Appelle l'API FastAPI pour obtenir les recommandations.
    """
    if user_id not in users_df['user_id'].unique():
        st.error("Cet identifiant utilisateur n'existe pas. Veuillez créer un compte.")
        return None
    
    with st.spinner('Recherche de vos recommandations...'):
        try:
            # L'API attend une requête POST avec un corps JSON
            response = requests.post(f"{API_URL}/recommendations/", json={'user_id': user_id}, timeout=10)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            
            data = response.json()
            
            # L'API renvoie directement une liste de dictionnaires
            return pd.DataFrame(data)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter le service de recommandation. Veuillez réessayer plus tard. (Erreur: {e})")
            return None

# --- Interface Streamlit ---

st.title("📚 Système de Recommandation de Contenu")

# Menu dans la barre latérale
menu = ["Recommandations", "Créer un compte", "Ajouter un article"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Recommandations":
    st.header("Obtenez vos recommandations")
    
    # Affiche la liste des utilisateurs pour faciliter le test
    if not users_df.empty:
        st.info("Utilisateurs existants (pour tester) :")
        st.dataframe(users_df, use_container_width=True)

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
                    
                    for _, row in recommendations.iterrows():
                        with st.container():
                            # Utiliser les données fusionnées si disponibles, sinon juste l'ID
                            st.subheader(f"{row.get('title', 'Titre inconnu')} (Article ID: {row['article_id']})")
                            st.write(str(row.get('content', ''))[:200] + "...") # Affiche un aperçu
                            st.markdown("---")
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
    st.dataframe(articles_df, use_container_width=True)
