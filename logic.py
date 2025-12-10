import pandas as pd
from io import StringIO, BytesIO
import requests
import os
import logging
import time
from azure.storage.blob import BlobServiceClient  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from functools import lru_cache, wraps
from azure.core.exceptions import ResourceNotFoundError

# --- Configuration ---
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
if not STORAGE_ACCOUNT_NAME:
    logging.error("AZURE_STORAGE_ACCOUNT_NAME n'est pas définie. Impossible de procéder.")
AZURE_CONTAINER_NAME = "reco-data"
USERS_BLOB_NAME = "users.csv"
ARTICLES_BLOB_NAME = "articles_metadata.csv"
CLICKS_BLOB_NAME = "clicks_sample.csv"
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"
CACHE_TTL_SECONDS = 600  # 10 minutes
logger = logging.getLogger(__name__)


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    Décorateur de cache avec une durée de vie (Time To Live).
    """

    def wrapper_cache(func):
        # Applique le cache LRU à la fonction
        cached_func = lru_cache(maxsize=maxsize)(func)

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Utilise un timestamp arrondi pour créer des "fenêtres" de cache
            # et des arguments de la fonction pour la cle de cache.

            now = time.time()
            ttl_hash = round(now / seconds)
            cache_key = (ttl_hash,) + args + tuple(sorted(kwargs.items()))
            if cache_key in cached_func.cache_info().keys():
                logger.info(f"Cache HIT for {func.__name__} with key {cache_key}")
            else:
                logger.info(f"Cache MISS for {func.__name__} with key {cache_key}. Executing function.")
            return cached_func(ttl_hash, *args, **kwargs)

        wrapped_func.cache_clear = cached_func.cache_clear
        return wrapped_func

    return wrapper_cache


@lru_cache(maxsize=1)
def recuperer_client_blob_service() -> BlobServiceClient:
    """Crée un client de service blob en utilisant la chaîne de connexion."""
    # Utilise DefaultAzureCredential pour l'authentification (Managed Identity en priorité)
    return BlobServiceClient(account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net", credential=DefaultAzureCredential())  # type: ignore


@timed_lru_cache(seconds=600)
def charger_df_depuis_blob(ttl_hash, blob_service_client: BlobServiceClient, blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob CSV."""
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
    # Adapter le nom pour la version Parquet.
    parquet_blob_name = blob_name.replace(".csv", ".parquet")
    parquet_blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=parquet_blob_name)

    try:
        # Essayer de charger le Parquet en priorité. Parquet attend un flux binaire.
        downloader = parquet_blob_client.download_blob()
        df = pd.read_parquet(downloader.readall())
        logger.info(f"Chargé depuis le format Parquet optimisé : {parquet_blob_name}")
        return df
    except ResourceNotFoundError:
        # Si le Parquet n'existe pas, se rabattre sur le CSV
        logger.warning(f"Blob Parquet non trouvé, tentative de chargement du CSV : {blob_name}")
        downloader = blob_client.download_blob()
        blob_data = downloader.readall()
        df = pd.read_csv(StringIO(blob_data))
        # Renomme la colonne 'click_article_id' en 'article_id' si elle existe, pour la cohérence
        if "click_article_id" in df.columns:
            df.rename(columns={"click_article_id": "article_id"}, inplace=True)

        # Assurez-vous que 'article_id' est de type int pour les merges futurs
        df["article_id"] = df["article_id"].astype(int)

        return df


def sauvegarder_df_vers_blob(blob_service_client: BlobServiceClient, df: pd.DataFrame, blob_name: str):
    """Sauvegarde un DataFrame dans un blob CSV."""
    # Sauvegarder au format Parquet pour la performance
    parquet_blob_name = blob_name.replace(".csv", ".parquet")
    output_parquet = BytesIO()
    df.to_parquet(output_parquet, index=False)
    parquet_blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=parquet_blob_name)
    parquet_blob_client.upload_blob(output_parquet.getvalue(), overwrite=True)
    logger.info(f"DataFrame sauvegardé au format Parquet : {parquet_blob_name}")


def obtenir_recommandations_pour_utilisateur(
    api_url: str, user_id: int, country_filter: str = None, device_filter: str = None
):
    """
    Vérifie l'utilisateur et appelle l'API externe pour obtenir des recommandations.
    Retourne un dictionnaire avec les résultats ou une erreur.
    """
    blob_service_client = recuperer_client_blob_service()
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)
    if clicks_df.empty or not clicks_df["user_id"].isin([user_id]).any():
        return {"error": f"L'utilisateur {user_id} n'existe pas."}

    try:  # type: ignore
        headers = {"Accept": "application/json"}
        response = requests.get(f"{api_url}/api/recommend", params={"user_id": user_id}, headers=headers, timeout=20)
        response.raise_for_status()
        recos = response.json()

        # Enrichir les recommandations avec les détails des articles
        if recos:
            articles_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=ARTICLES_BLOB_NAME)
            recos_df = pd.DataFrame(recos)
            reco_details = recos_df.merge(articles_df, on="article_id", how="left")

            # Appliquer les filtres si fournis
            if country_filter or device_filter:
                article_context = _get_article_context(
                    ttl_hash=round(time.time() / CACHE_TTL_SECONDS), clicks_df=clicks_df
                )  # type: ignore
                reco_details = reco_details.merge(article_context, on="article_id", how="left")

                if country_filter:
                    reco_details = reco_details[
                        reco_details["unique_countries"].apply(
                            lambda x: country_filter in x if isinstance(x, list) else False
                        )
                    ]
                if device_filter:
                    reco_details = reco_details[
                        reco_details["unique_devices"].apply(
                            lambda x: device_filter in x if isinstance(x, list) else False
                        )
                    ]
            return reco_details.to_dict(orient="records")
        return []
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 404:
            logger.error(f"L'endpoint de recommandation n'a pas été trouvé (404) pour user_id {user_id}: {e}")
            return {"error": "Le point de terminaison de l'API de recommandation est introuvable."}
        logger.error(f"Erreur API pour user_id {user_id}: {e}")
        return {"error": "Le service de recommandation est indisponible."}
    except Exception as e:
        logger.error(f"Erreur inattendue pour user_id {user_id}: {e}")
        return {"error": "Une erreur inattendue est survenue."}


def obtenir_historique_utilisateur(user_id: int):
    """Récupère l'historique des notations pour un utilisateur."""
    blob_service_client = recuperer_client_blob_service()
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)
    if clicks_df.empty:
        return []

    user_history = clicks_df[clicks_df["user_id"] == user_id]
    if user_history.empty:
        return []
    articles_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=ARTICLES_BLOB_NAME)
    history_details = user_history.merge(articles_df, on="article_id", how="left").fillna({"title": "Titre inconnu"})
    history_details = history_details.sort_values(by="click_timestamp", ascending=False)

    return history_details.to_dict(orient="records")


def ajouter_ou_mettre_a_jour_interaction(user_id: int, article_id: int, rating: int):
    """Ajoute ou met à jour une notation."""
    blob_service_client = recuperer_client_blob_service()
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)

    # Vérifie s'il existe une interaction pour ce couple user/article
    user_article_interactions = clicks_df[(clicks_df["user_id"] == user_id) & (clicks_df["article_id"] == article_id)]

    if not user_article_interactions.empty:
        # Mise à jour : trouve l'index de la dernière interaction
        latest_interaction_index = user_article_interactions["click_timestamp"].idxmax()
        clicks_df.loc[latest_interaction_index, "nb"] = rating
        clicks_df.loc[latest_interaction_index, "click_timestamp"] = int(pd.Timestamp.now().timestamp())
    else:
        # Ajout : crée une nouvelle ligne
        new_interaction = pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "article_id": article_id,
                    "click_timestamp": int(pd.Timestamp.now().timestamp()),
                    "nb": rating,
                }
            ]
        )
        clicks_df = pd.concat([clicks_df, new_interaction], ignore_index=True)

    sauvegarder_df_vers_blob(blob_service_client, clicks_df, CLICKS_BLOB_NAME)


def creer_nouvel_utilisateur():
    """Crée un nouvel utilisateur avec un ID unique."""
    blob_service_client = recuperer_client_blob_service()
    users_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=USERS_BLOB_NAME)

    if users_df.empty:
        new_user_id = 1
    else:
        new_user_id = int(users_df["user_id"].max()) + 1
    new_user_df = pd.DataFrame([{"user_id": new_user_id}])
    updated_users_df = pd.concat([users_df, new_user_df], ignore_index=True)
    sauvegarder_df_vers_blob(blob_service_client, updated_users_df, USERS_BLOB_NAME)
    return new_user_id


def obtenir_utilisateurs():
    """Récupère la liste de tous les utilisateurs uniques à partir des clics."""
    blob_service_client = recuperer_client_blob_service()
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)
    if clicks_df.empty:
        return []
    unique_users = clicks_df["user_id"].unique()
    users_df = pd.DataFrame(unique_users, columns=["user_id"])
    return users_df.to_dict(orient="records")


def obtenir_contexte_utilisateur(user_id: int):
    """Récupère le pays et le groupe d'appareils du dernier clic de l'utilisateur."""
    blob_service_client = recuperer_client_blob_service()
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)

    if clicks_df.empty or user_id not in clicks_df["user_id"].unique():
        return None

    user_clicks = clicks_df[clicks_df["user_id"] == user_id]
    if user_clicks.empty:
        return None

    # Trouver le clic le plus récent
    latest_click = user_clicks.sort_values(by="click_timestamp", ascending=False).iloc[0]

    return {
        "country": latest_click.get("click_country", "Inconnu"),
        "deviceGroup": latest_click.get("click_deviceGroup", "Inconnu"),
    }


def creer_nouvel_article(title: str, content: str, category_id: int) -> int:
    """Crée un nouvel article et le sauvegarde dans le blob."""
    blob_service_client = recuperer_client_blob_service()
    articles_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=ARTICLES_BLOB_NAME)

    if articles_df.empty:
        new_article_id = 1
    else:
        # Assure que 'article_id' est numérique pour max()
        articles_df["article_id"] = pd.to_numeric(articles_df["article_id"], errors="coerce").fillna(0)
        new_article_id = int(articles_df["article_id"].max()) + 1

    new_article = pd.DataFrame(
        [
            {
                "article_id": new_article_id,
                "title": title,
                "content": content,
                "category_id": category_id,
                "created_at_ts": int(pd.Timestamp.now().timestamp()),
            }
        ]
    )
    updated_articles_df = pd.concat([articles_df, new_article], ignore_index=True)
    sauvegarder_df_vers_blob(blob_service_client, updated_articles_df, ARTICLES_BLOB_NAME)
    return new_article_id


def obtenir_performance_modele():
    """Récupère les logs de performance de l'entraînement du modèle."""
    blob_service_client = recuperer_client_blob_service()
    performance_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=TRAINING_LOG_BLOB_NAME)
    if performance_df.empty:
        return []
    return performance_df.to_dict(orient="records")


def obtenir_tendances_globales_clics():
    """
    Récupère et agrège les données de clics pour les tendances globales par pays et par groupe d'appareils."""
    blob_service_client = recuperer_client_blob_service()  # type: ignore
    clicks_df = charger_df_depuis_blob(blob_service_client=blob_service_client, blob_name=CLICKS_BLOB_NAME)

    if clicks_df.empty:
        return {"clicks_by_country": [], "clicks_by_device": []}

    # Agrégation par pays
    clicks_by_country = clicks_df["click_country"].value_counts().reset_index()
    clicks_by_country.columns = ["country", "count"]

    # Agrégation par groupe d'appareils
    clicks_by_device = clicks_df["click_deviceGroup"].value_counts().reset_index()
    clicks_by_device.columns = ["deviceGroup", "count"]

    return {
        "clicks_by_country": clicks_by_country.to_dict(orient="records"),
        "clicks_by_device": clicks_by_device.to_dict(orient="records"),
    }


@timed_lru_cache(seconds=600)
def _get_article_context(ttl_hash, clicks_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
    """Calcule et cache le contexte des articles (pays/appareils uniques) à partir des clics."""
    if clicks_df.empty:
        return pd.DataFrame(columns=["article_id", "unique_countries", "unique_devices"])

    article_context = (
        clicks_df.groupby("article_id")
        .agg(
            unique_countries=("click_country", lambda x: list(x.unique())),
            unique_devices=("click_deviceGroup", lambda x: list(x.unique())),
        )
        .reset_index()
    )
    logger.info(f"Calculé et mis en cache le contexte des articles à partir de {len(clicks_df)} clics.")
    return article_context
