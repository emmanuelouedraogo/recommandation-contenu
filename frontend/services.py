import streamlit as st
import pandas as pd
import requests
from azure.storage.blob import BlobServiceClient
from io import StringIO
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
import logging

logger = logging.getLogger(__name__)

# --- Constantes de l'Application ---
AZURE_CONTAINER_NAME = "reco-data"
USERS_BLOB_NAME = "users.csv"
ARTICLES_BLOB_NAME = "articles_metadata.csv"
CLICKS_BLOB_NAME = "clicks_sample.csv"
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"


# ==============================================================================
# --- Services de Données (Azure Blob Storage) ---
# ==============================================================================

@st.cache_resource(ttl=3600)
def recuperer_client_blob_service() -> BlobServiceClient:
    """Crée un client de service blob en utilisant la chaîne de connexion des secrets."""
    connect_str = st.secrets.get("STORAGE_CONNECTION_STRING")
    if not connect_str:
        st.error("La chaîne de connexion au stockage ('STORAGE_CONNECTION_STRING') n'est pas configurée !")
        st.stop()
    return BlobServiceClient.from_connection_string(connect_str)


@st.cache_data(ttl=3600)
def charger_df_depuis_blob(blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob CSV."""
    blob_client = recuperer_client_blob_service().get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    try:
        downloader = blob_client.download_blob(encoding='utf-8')
        blob_data = downloader.readall()
        df = pd.read_csv(StringIO(blob_data))
        return df
    except ServiceRequestError as e:
        st.error(f"Erreur de connexion au stockage Azure. Vérifiez votre connexion internet et la chaîne de connexion. Erreur: {e}")
        return pd.DataFrame()
    except ResourceNotFoundError:
        st.warning(f"Le blob '{blob_name}' n'a pas été trouvé. Un nouveau sera créé si nécessaire.")
        return pd.DataFrame()


def sauvegarder_df_vers_blob(df: pd.DataFrame, blob_name: str) -> bool:
    """Sauvegarde un DataFrame dans un blob CSV."""
    output = StringIO()
    df.to_csv(output, index=False)
    blob_client = recuperer_client_blob_service().get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    try:
        blob_client.upload_blob(output.getvalue(), overwrite=True)
        return True
    except Exception as e:
        error_msg = f"Échec de la sauvegarde des données dans '{blob_name}'. Erreur: {e}"
        st.error(error_msg)
        logger.error(error_msg)
        return False


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
        st.cache_data.clear()
        st.toast(f"Merci pour votre note de {rating}/5 !", icon="⭐")  # noqa

def mettre_a_jour_interaction(user_id, article_id, new_rating):
    """Met à jour la note la plus récente pour un article donné par un utilisateur."""
    clicks_df = charger_df_depuis_blob(CLICKS_BLOB_NAME)
    user_article_interactions = clicks_df[(clicks_df['user_id'] == user_id) & (clicks_df['article_id'] == article_id)]
    if not user_article_interactions.empty:
        latest_interaction_index = user_article_interactions['click_timestamp'].idxmax()
        clicks_df.loc[latest_interaction_index, 'nb'] = new_rating
        clicks_df.loc[latest_interaction_index, 'click_timestamp'] = int(pd.Timestamp.now().timestamp())
        if sauvegarder_df_vers_blob(clicks_df, CLICKS_BLOB_NAME):
            st.cache_data.clear()  # noqa
            st.toast(f"Votre note a été mise à jour à {new_rating}/5 !", icon="👍")  # noqa


# ==============================================================================
# --- Services API (Recommandation) ---
# ==============================================================================

def obtenir_recommandations(api_url, user_id):
    """Appelle l'API pour obtenir les recommandations."""
    logger.info(f"Début de la récupération des recommandations pour user_id: {user_id}")
    users_df = charger_df_depuis_blob(USERS_BLOB_NAME)
    if users_df.empty:
        st.error("Impossible de vérifier l'utilisateur. Le fichier des utilisateurs est vide ou inaccessible.")
        return None
    if user_id not in users_df['user_id'].unique():
        st.error(f"L'identifiant utilisateur '{user_id}' n'existe pas. Veuillez créer un compte.")
        return None

    with st.spinner('Recherche de vos recommandations...'):
        try:
            headers = {'Accept': 'application/json'}
            response = requests.get(
                f"{api_url}/api/recommend",
                params={"user_id": user_id},
                headers=headers,
                timeout=20
            )
            response.raise_for_status()
            try:
                data = response.json()
                logger.info(f"Recommandations reçues avec succès pour user_id: {user_id}. Nombre: {len(data)}")
                return pd.DataFrame(data)
            except requests.exceptions.JSONDecodeError:
                st.error(f"Le service de recommandation a renvoyé une réponse invalide. Statut: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter le service de recommandation. (Erreur: {e})")
            return None
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue: {e}")
            return None