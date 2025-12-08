import streamlit as st
import pandas as pd
import os
import requests
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from io import StringIO
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
import logging
# --- Configuration de la Page ---
# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recommandation de Contenu",
    page_icon="📚",
    layout="wide"
)

# --- Configuration du Logger ---
# Créer un logger pour suivre les événements de l'application.
# En production, logger vers stdout/stderr est la meilleure pratique.
# Azure App Service collecte ces logs automatiquement.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Autres constantes
AZURE_CONTAINER_NAME = "reco-data"
USERS_BLOB_NAME = "users.csv"
ARTICLES_BLOB_NAME = "articles_metadata.csv"  # Nom du blob pour les articles
CLICKS_BLOB_NAME = "clicks_sample.csv"        # Nom du blob pour les interactions
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"
STORAGE_SECRET_NAME = "STORAGE-CONNECTION-STRING" # Le nom du secret pour la chaîne de connexion
API_URL_SECRET_NAME = "API-URL"                   # Le nom du secret pour l'URL de l'API

# --- Gestion des Secrets via Azure Key Vault ---
class ConfigError(Exception):
    """Exception personnalisée pour les erreurs de configuration."""
    pass

@st.cache_data(show_spinner=False)
def recuperer_secret_depuis_key_vault(vault_url: str, secret_name: str) -> str:
    """Récupère un secret depuis Azure Key Vault en utilisant l'identité managée."""
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        retrieved_secret = client.get_secret(secret_name)
        logger.info(f"Secret '{secret_name}' récupéré avec succès depuis Key Vault.")
        return retrieved_secret.value
    except Exception as e:
        logger.critical(f"Échec de la récupération du secret depuis Key Vault. URL: {vault_url}. Erreur: {e}")
        raise ConfigError(f"Impossible de récupérer le secret '{secret_name}' depuis Azure Key Vault.") from e

# L'URL du Key Vault doit être stockée dans les secrets Streamlit ou en variable d'environnement
KEY_VAULT_URL = st.secrets.get("KEY_VAULT_URL") or os.environ.get("KEY_VAULT_URL")

if not KEY_VAULT_URL:
    st.error("Le secret 'KEY_VAULT_URL' n'est pas configuré. Ajoutez-le à .streamlit/secrets.toml ou en variable d'environnement.")
    st.stop()

try:
    # Récupération dynamique de la chaîne de connexion
    AZURE_CONNECTION_STRING = recuperer_secret_depuis_key_vault(KEY_VAULT_URL, STORAGE_SECRET_NAME)
    # Récupération dynamique de l'URL de l'API depuis le Key Vault
    API_URL = recuperer_secret_depuis_key_vault(KEY_VAULT_URL, API_URL_SECRET_NAME)
    # S'assurer que l'URL ne se termine pas par un slash pour éviter les doubles slashes
    API_URL = API_URL.strip().rstrip('/')
except ConfigError as e:
    st.error(f"Erreur de configuration critique : {e}")
    st.info("Veuillez vérifier les permissions de l'identité managée de l'App Service sur le Key Vault et la présence des secrets.")
    st.stop()

# ==============================================================================
# --- Fonctions de Chargement des Données ---
# ==============================================================================
@st.cache_resource(ttl=3600)
def recuperer_client_blob_service(conn_str: str) -> BlobServiceClient:
    """Crée un client de service blob. Mis en cache pour la performance."""
    if not conn_str:
        st.error("La chaîne de connexion Azure n'est pas configurée dans les secrets !")
        st.stop()
    return BlobServiceClient.from_connection_string(conn_str)

# Initialiser le client une seule fois
blob_service_client = recuperer_client_blob_service(AZURE_CONNECTION_STRING)

@st.cache_data(ttl=3600) # Cache les données pendant 1 heure
def charger_df_depuis_blob(blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob CSV en utilisant le client global."""
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    try:
        downloader = blob_client.download_blob(encoding='utf-8')
        blob_data = downloader.readall()
        return pd.read_csv(StringIO(blob_data))
    except ServiceRequestError as e:
        st.error(f"Erreur de connexion au stockage Azure. Vérifiez votre connexion internet et la chaîne de connexion. Erreur: {e}")
        return pd.DataFrame()
    except ResourceNotFoundError:
        st.warning(f"Le blob '{blob_name}' n'a pas été trouvé. Un nouveau sera créé si nécessaire.")
        return pd.DataFrame()

def sauvegarder_df_vers_blob(df: pd.DataFrame, blob_name: str) -> bool:
    """Sauvegarde un DataFrame dans un blob CSV en utilisant le client global."""

    output = StringIO()
    df.to_csv(output, index=False)
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    try:
        blob_client.upload_blob(output.getvalue(), overwrite=True)
        return True
    except Exception as e:
        error_msg = f"Échec de la sauvegarde des données dans '{blob_name}'. Erreur: {e}"
        st.error(error_msg)
        logger.error(error_msg)
        return False

# ==============================================================================
# --- Logique de l'application ---
# ==============================================================================
def ajouter_interaction(user_id, article_id, rating):
    """Ajoute une nouvelle interaction (note) et la sauvegarde."""
    clicks_df = charger_df_depuis_blob(CLICKS_BLOB_NAME)
    
    new_interaction = pd.DataFrame([{
        'user_id': user_id,
        'article_id': article_id,
        'click_timestamp': int(pd.Timestamp.now().timestamp()),
        'nb': rating
    }])
    
    updated_clicks_df = pd.concat([clicks_df, new_interaction], ignore_index=True)
    
    if sauvegarder_df_vers_blob(updated_clicks_df, CLICKS_BLOB_NAME):
        # Invalide le cache pour que la prochaine lecture récupère les données à jour
        st.cache_data.clear()
        st.toast(f"Merci pour votre note de {rating}/5 !", icon="⭐")

def mettre_a_jour_interaction(user_id, article_id, new_rating):
    """Met à jour la note la plus récente pour un article donné par un utilisateur."""
    clicks_df = charger_df_depuis_blob(CLICKS_BLOB_NAME)
    
    # Trouve l'index de la dernière interaction pour ce couple utilisateur/article
    user_article_interactions = clicks_df[(clicks_df['user_id'] == user_id) & (clicks_df['article_id'] == article_id)]
    if not user_article_interactions.empty:
        latest_interaction_index = user_article_interactions['click_timestamp'].idxmax()
        
        # Met à jour la note et le timestamp
        clicks_df.loc[latest_interaction_index, 'nb'] = new_rating
        clicks_df.loc[latest_interaction_index, 'click_timestamp'] = int(pd.Timestamp.now().timestamp())
        
        if sauvegarder_df_vers_blob(clicks_df, CLICKS_BLOB_NAME):
            # Invalide le cache pour que l'historique se mette à jour
            st.cache_data.clear()
            st.toast(f"Votre note a été mise à jour à {new_rating}/5 !", icon="👍")

# ==============================================================================
# --- Fonctions du Système de Recommandation (API) ---
# ==============================================================================
def obtenir_recommandations(user_id):
    """
    Appelle l'API FastAPI pour obtenir les recommandations.
    """
    logger.info(f"Début de la récupération des recommandations pour user_id: {user_id}")
    users_df = charger_df_depuis_blob(USERS_BLOB_NAME)
    if users_df.empty:
        error_msg = "Impossible de vérifier l'utilisateur. Le fichier des utilisateurs est vide ou inaccessible."
        st.error(error_msg)
        logger.warning(f"Échec de la vérification pour user_id {user_id}: {error_msg}")
        return None
    if user_id not in users_df['user_id'].unique():
        st.error(f"L'identifiant utilisateur '{user_id}' n'existe pas. Veuillez créer un compte.") # Message pour l'UI
        return None
    
    with st.spinner('Recherche de vos recommandations...'):
        try:
            # L'API Azure Function déployée attend une requête GET avec un paramètre d'URL.
            # L'URL de l'endpoint est "/api/recommend".
            headers = {'Accept': 'application/json'}
            response = requests.get(f"{API_URL}/api/recommend", params={"user_id": user_id}, headers=headers, timeout=20)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            
            try:
                data = response.json()
                # L'API renvoie une liste de dictionnaires que nous convertissons en DataFrame.
                logger.info(f"Recommandations reçues avec succès pour user_id: {user_id}. Nombre de recos: {len(data)}")
                return pd.DataFrame(data)
            except requests.exceptions.JSONDecodeError:
                error_msg = f"Le service de recommandation a renvoyé une réponse invalide. Statut: {response.status_code}, Contenu: {response.text}"
                st.error(error_msg)
                logger.error(f"Erreur de décodage JSON pour user_id {user_id}. {error_msg}")
                return None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Impossible de contacter le service de recommandation. Vérifiez que l'URL de l'API est correcte et que le service est démarré. (Erreur: {e})"
            st.error(error_msg)
            logger.critical(f"Échec de l'appel API pour user_id {user_id}. {error_msg}")
            return None
        except Exception as e:
            error_msg = f"Une erreur inattendue est survenue: {e}"
            st.error(error_msg)
            logger.error(f"Erreur inattendue lors de la récupération des recommandations pour user_id {user_id}. {error_msg}")
            return None

def afficher_page_recommandations():
    """Affiche la page des recommandations."""
    st.header("Obtenez vos recommandations")
    
    # Affiche la liste des utilisateurs pour faciliter le test
    users_df_display = charger_df_depuis_blob(USERS_BLOB_NAME)
    if not users_df_display.empty:
        st.info("Utilisateurs existants (pour les tests) :")
        st.dataframe(users_df_display, width='stretch')

    if st.session_state.user_id is None:
        st.info("Veuillez vous connecter via la barre latérale pour obtenir vos recommandations.")
    else:
        user_id = st.session_state.user_id
        recommendations = obtenir_recommandations(user_id)
        
        if recommendations is not None and not recommendations.empty:
            articles_df = charger_df_depuis_blob(ARTICLES_BLOB_NAME)
            reco_details = recommendations.merge(articles_df, on='article_id', how='left')
            
            st.success(f"Bienvenue, Utilisateur {user_id} ! Voici vos recommandations personnalisées :")
            
            for _, row in reco_details.iterrows():
                with st.container():
                    st.subheader(f"{row.get('title', 'Titre inconnu')}")
                    st.caption(f"Score de recommandation : {row.get('final_score', 0):.2f} | ID Article : {row['article_id']}")
                    st.write(str(row.get('content', 'Contenu non disponible.'))[:250] + "...")
                    
                    rating = st.slider("Notez cet article :", 1, 5, 3, key=f"rating_{row['article_id']}")
                    if st.button("Envoyer ma note", key=f"btn_{row['article_id']}"):
                        ajouter_interaction(user_id, row['article_id'], rating)
                    st.divider()
        elif recommendations is not None:
             st.warning("Il n'y a pas assez d'articles à recommander pour le moment.")

def afficher_page_historique():
    """Affiche la page de l'historique des notations."""
    st.header("Historique de vos notations")
    
    if st.session_state.user_id is None:
        st.info("Veuillez vous connecter via la barre latérale pour voir votre historique.")
    else:
        user_id = st.session_state.user_id
        clicks_df = charger_df_depuis_blob(CLICKS_BLOB_NAME)
        
        if clicks_df.empty:
            st.warning("Aucune notation n'a encore été enregistrée dans le système.")
        else:
            user_history_df = clicks_df[clicks_df['user_id'] == user_id]
            
            if user_history_df.empty:
                st.info("Vous n'avez encore noté aucun article.")
            else:
                # --- Correction et Validation ---
                required_cols = ['user_id', 'article_id', 'click_timestamp', 'nb']
                if not all(col in user_history_df.columns for col in required_cols):
                    st.error("Le fichier d'historique (clicks_sample.csv) est mal formaté. Il manque des colonnes attendues (ex: 'user_id', 'article_id').")
                    logger.error(f"Colonnes manquantes dans clicks_sample.csv. Colonnes trouvées : {user_history_df.columns.tolist()}")
                    st.stop()

                user_history_df = user_history_df.sort_values('click_timestamp').drop_duplicates(subset=['user_id', 'article_id'], keep='last')
                articles_df_history = charger_df_depuis_blob(ARTICLES_BLOB_NAME)
                history_details = user_history_df.merge(articles_df_history, on='article_id', how='left').fillna({'title': 'Titre inconnu'})
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
                            mettre_a_jour_interaction(user_id, row['article_id'], new_rating)
                    st.divider()

def afficher_page_performance():
    """Affiche la page de performance du modèle."""
    st.header("Historique et Performance des Entraînements")

    log_df = charger_df_depuis_blob(TRAINING_LOG_BLOB_NAME)

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

def afficher_page_creation_compte():
    """Affiche la page de création de compte."""
    st.header("Créez votre compte")
    
    if st.button("Créer un nouvel identifiant"):
        current_users_df = charger_df_depuis_blob(USERS_BLOB_NAME)
        # Génère un nouvel ID unique (plus robuste qu'un simple incrément)
        if current_users_df.empty:
            new_user_id = 1
        else:
            new_user_id = int(current_users_df['user_id'].max() if not current_users_df.empty else 0) + 1
            while new_user_id in current_users_df['user_id'].values:
                new_user_id += 1 # Assure l'unicité même si des IDs ont été supprimés
        
        # Ajoute au DataFrame et sauvegarde
        new_user_df = pd.DataFrame([{'user_id': new_user_id}])
        updated_users_df = pd.concat([current_users_df, new_user_df], ignore_index=True)
        
        if sauvegarder_df_vers_blob(updated_users_df, USERS_BLOB_NAME):
            st.cache_data.clear()
            st.success(f"Votre nouveau compte a été créé avec succès ! Votre identifiant est :")
            st.code(new_user_id, language='text')
            st.info("Vous pouvez maintenant utiliser cet identifiant dans la section 'Recommandations'.")

def afficher_page_ajout_article():
    """Affiche la page d'ajout d'article."""
    st.header("Ajouter un nouvel article ou livre")

    with st.form(key="article_form", clear_on_submit=True):
        article_title = st.text_input("Titre de l'article/livre")
        article_category = st.number_input("ID de la catégorie", min_value=0, step=1)
        article_content = st.text_area("Contenu")
        submit_button = st.form_submit_button(label="Ajouter à la base de données")

        if submit_button and article_title and article_content:
            current_articles_df = charger_df_depuis_blob(ARTICLES_BLOB_NAME)
            # Génère un ID unique pour l'article
            new_article_id = int(current_articles_df['article_id'].max() + 1) if not current_articles_df.empty else 1
            
            new_article = pd.DataFrame([{
                'article_id': new_article_id,
                'title': article_title,
                'content': article_content,
                'category_id': article_category,
                'created_at_ts': int(pd.Timestamp.now().timestamp())
            }])
            
            updated_articles_df = pd.concat([current_articles_df, new_article], ignore_index=True)
            if sauvegarder_df_vers_blob(updated_articles_df, ARTICLES_BLOB_NAME):
                # Invalide le cache pour que la liste des articles soit mise à jour
                st.cache_data.clear()
                st.success(f"L'article '{article_title}' a été ajouté avec succès !")
    
    st.divider()
    st.subheader("Articles actuels dans la base de données")
    # Recharger les données pour afficher le nouvel article
    st.dataframe(charger_df_depuis_blob(ARTICLES_BLOB_NAME), width='stretch')

# ==============================================================================
# --- Interface Streamlit ---
# ==============================================================================
st.title("📚 Système de Recommandation de Contenu")

# --- Gestion de la session utilisateur ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Menu dans la barre latérale
st.sidebar.title("Navigation")
menu = ["Recommandations", "Mon Historique", "Performance du Modèle", "Créer un compte", "Ajouter un article"]
choice = st.sidebar.selectbox("Menu", menu)

st.sidebar.divider()

# --- Section de connexion dans la barre latérale ---
if st.session_state.user_id is None:
    st.sidebar.header("Connexion")
    login_user_id = st.sidebar.text_input("Entrez votre identifiant utilisateur", key="login_input")
    if st.sidebar.button("Se connecter"):
        if login_user_id:
            try:
                user_id_to_check = int(login_user_id)
                users_df = charger_df_depuis_blob(USERS_BLOB_NAME)
                if user_id_to_check in users_df['user_id'].unique():
                    st.session_state.user_id = user_id_to_check
                    st.rerun() # Recharge la page pour refléter l'état connecté
                else:
                    st.sidebar.error("Cet utilisateur n'existe pas.")
            except ValueError:
                st.sidebar.error("L'ID doit être un nombre.")
else:
    st.sidebar.success(f"Connecté en tant que : **{st.session_state.user_id}**")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.user_id = None
        st.rerun()

# --- Routeur de page principal ---
if choice == "Recommandations":
    afficher_page_recommandations()
elif choice == "Mon Historique":
    afficher_page_historique()
elif choice == "Performance du Modèle":
    afficher_page_performance()
elif choice == "Créer un compte":
    afficher_page_creation_compte()
elif choice == "Ajouter un article":
    afficher_page_ajout_article()
