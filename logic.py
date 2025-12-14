import pandas as pd
from io import StringIO, BytesIO
import requests
import os
import logging
import time
from datetime import timezone
from azure.storage.blob import BlobServiceClient  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from functools import lru_cache, wraps
from azure.core.exceptions import ResourceNotFoundError

# --- Configuration ---
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
if not STORAGE_ACCOUNT_NAME:
    raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable is not set. Application cannot start.")
API_RECO_URL = os.getenv("API_URL")
if not API_RECO_URL:
    raise ValueError("API_URL environment variable is not set. Application cannot start.")
AZURE_CONTAINER_NAME = "reco-data"
USERS_BLOB_NAME = "users.csv"  # Le nom du fichier reste le même
ARTICLES_BLOB_NAME = "articles_metadata.csv"
CLICKS_BLOB_NAME = "clicks_sample.csv"
TRAINING_LOG_BLOB_NAME = "logs/training_log.csv"
INTERACTIONS_LOG_BLOB_NAME = "interactions/new_interactions_log.csv"  # For appending new interactions
STATUS_BLOB_NAME = "status/retraining_status.json"  # Fichier pour le statut en direct
CACHE_TTL_SECONDS = 600  # 10 minutes
logger = logging.getLogger(__name__)


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    Décorateur de cache avec une durée de vie (Time To Live).
    """

    def wrapper_cache(func):
        # Applique le cache LRU à la fonction
        @lru_cache(maxsize=maxsize)
        def cached_func_with_ttl(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Utilise un timestamp arrondi pour créer des "fenêtres" de cache
            now = time.time()
            ttl_hash = round(now / seconds)
            return cached_func_with_ttl(ttl_hash, *args, **kwargs)

        wrapped_func.cache_clear = cached_func_with_ttl.cache_clear
        return wrapped_func

    return wrapper_cache


@lru_cache(maxsize=1)
def recuperer_client_blob_service() -> BlobServiceClient:
    """Crée un client de service blob en utilisant la chaîne de connexion."""
    # Utilise DefaultAzureCredential pour l'authentification (Managed Identity en priorité).
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())  # type: ignore


@timed_lru_cache(seconds=600)
def charger_df_depuis_blob(blob_name: str) -> pd.DataFrame:
    """Charge un DataFrame depuis un blob CSV."""
    blob_service_client = recuperer_client_blob_service()
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
        blob_data = downloader.readall().decode("utf-8")
        df = pd.read_csv(StringIO(blob_data))
        # Renomme la colonne 'click_article_id' en 'article_id' si elle existe, pour la cohérence
        if "click_article_id" in df.columns:
            df.rename(columns={"click_article_id": "article_id"}, inplace=True)

        # Assurez-vous que 'article_id' est de type int pour les merges futurs
        df["article_id"] = df["article_id"].astype(int)

        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du blob {blob_name}: {e}")
        return pd.DataFrame()


def sauvegarder_df_vers_blob(df: pd.DataFrame, blob_name: str):
    """Sauvegarde un DataFrame dans un blob au format Parquet."""
    blob_service_client = recuperer_client_blob_service()
    parquet_blob_name = blob_name.replace(".csv", ".parquet")
    output_parquet = BytesIO()
    df.to_parquet(output_parquet, index=False)
    parquet_blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=parquet_blob_name)
    parquet_blob_client.upload_blob(output_parquet.getvalue(), overwrite=True)
    logger.info(f"DataFrame sauvegardé au format Parquet : {parquet_blob_name}")


def obtenir_recommandations_pour_utilisateur(user_id: int, country_filter: str = None, device_filter: str = None):
    """
    Vérifie l'utilisateur et appelle l'API externe pour obtenir des recommandations.
    Retourne un dictionnaire avec les résultats ou une erreur.
    """  # type: ignore
    # La vérification de l'existence de l'utilisateur est maintenant gérée par la liste unifiée
    all_users = [user["user_id"] for user in obtenir_utilisateurs()]
    if user_id not in all_users:
        return {"error": f"L'utilisateur {user_id} n'existe pas."}

    try:
        headers = {"Accept": "application/json"}
        response = requests.get(
            f"{API_RECO_URL}/api/recommend", params={"user_id": user_id}, headers=headers, timeout=20
        )
        response.raise_for_status()
        recos = response.json()

        # Enrichir les recommandations avec les détails des articles
        if recos:
            articles_df = charger_df_depuis_blob(blob_name=ARTICLES_BLOB_NAME)
            recos_df = pd.DataFrame(recos)
            reco_details = recos_df.merge(articles_df, on="article_id", how="left")

            # Appliquer les filtres si fournis
            if country_filter or device_filter:
                article_context = _get_article_context()  # type: ignore
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
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)
    if clicks_df.empty:
        return []

    user_history = clicks_df[clicks_df["user_id"] == user_id]
    if user_history.empty:
        return []
    articles_df = charger_df_depuis_blob(blob_name=ARTICLES_BLOB_NAME)
    history_details = user_history.merge(articles_df, on="article_id", how="left").fillna({"title": "Titre inconnu"})
    history_details = history_details.sort_values(by="click_timestamp", ascending=False)

    return history_details.to_dict(orient="records")


def ajouter_ou_mettre_a_jour_interaction(user_id: int, article_id: int, rating: int):
    """Ajoute ou met à jour une notation."""
    # --- Refactoring for Performance: Use an Append-Only Log ---
    # Instead of reading and rewriting the entire clicks dataset for each interaction,
    # we append new interactions to a log file. This is much faster and more scalable.
    # A separate batch process (like the existing retrain trigger) can then merge these logs.
    blob_service_client = recuperer_client_blob_service()
    append_blob_client = blob_service_client.get_blob_client(
        container=AZURE_CONTAINER_NAME, blob=INTERACTIONS_LOG_BLOB_NAME
    )

    # Create the append blob with a header if it doesn't exist
    try:
        append_blob_client.get_blob_properties()
    except ResourceNotFoundError:
        logger.info(f"Creating new interaction log blob: {INTERACTIONS_LOG_BLOB_NAME}")
        append_blob_client.create_blob(metadata={"blob_type": "AppendBlob"})
        header = "user_id,article_id,rating,timestamp\n"
        append_blob_client.append_block(header.encode("utf-8"))

    # Append the new interaction as a single line of CSV
    timestamp = int(pd.Timestamp.now().timestamp())
    log_entry = f"{user_id},{article_id},{rating},{timestamp}\n"
    append_blob_client.append_block(log_entry.encode("utf-8"))
    logger.info(f"Logged new interaction for user {user_id}, article {article_id}.")


def creer_nouvel_utilisateur():
    """Crée un nouvel utilisateur avec un ID unique."""
    # Utilise la liste des utilisateurs existants (clics + users.csv) comme source de vérité
    existing_users = obtenir_utilisateurs()
    if not existing_users:
        new_user_id = 1
    else:
        max_id = max(user["user_id"] for user in existing_users)
        new_user_id = max_id + 1

    # Ajoute le nouvel utilisateur au fichier users.csv pour persistance
    users_df = charger_df_depuis_blob(blob_name=USERS_BLOB_NAME)
    new_user_df = pd.DataFrame([{"user_id": new_user_id, "date_creation": pd.Timestamp.now(timezone.utc).isoformat()}])
    updated_users_df = pd.concat([users_df, new_user_df], ignore_index=True) if not users_df.empty else new_user_df
    sauvegarder_df_vers_blob(updated_users_df, USERS_BLOB_NAME)

    # Invalider le cache des utilisateurs pour que le nouveau soit visible immédiatement
    obtenir_utilisateurs.cache_clear()
    return new_user_id


def supprimer_utilisateur(user_id: int):
    """
    Désactive un utilisateur en le marquant comme 'deleted' (suppression douce).
    Cette opération est réversible et préserve l'historique des interactions.
    """
    users_df = charger_df_depuis_blob(blob_name=USERS_BLOB_NAME)

    if users_df.empty:
        logger.warning(
            f"Tentative de suppression de l'utilisateur {user_id}, mais le fichier des utilisateurs est vide."
        )
        return False

    if "status" not in users_df.columns:
        users_df["status"] = "active"

    if user_id in users_df["user_id"].values:
        # Marquer l'utilisateur comme supprimé
        users_df.loc[users_df["user_id"] == user_id, "status"] = "deleted"
        sauvegarder_df_vers_blob(users_df, USERS_BLOB_NAME)
        logger.info(f"Utilisateur {user_id} marqué comme 'deleted' (soft delete).")

        # Invalider le cache pour que la liste des utilisateurs soit mise à jour
        obtenir_utilisateurs.cache_clear()
        return True
    else:
        # Si l'utilisateur n'est pas dans users.csv, il est peut-être implicite via clicks_sample.csv
        # Dans ce cas, on l'ajoute à users.csv avec le statut 'deleted'
        logger.warning(f"Utilisateur {user_id} non trouvé dans {USERS_BLOB_NAME}. Ajout avec le statut 'deleted'.")
        new_deleted_user = pd.DataFrame(
            [{"user_id": user_id, "status": "deleted", "date_creation": pd.Timestamp.now(timezone.utc).isoformat()}]
        )
        updated_df = pd.concat([users_df, new_deleted_user], ignore_index=True)
        sauvegarder_df_vers_blob(updated_df, USERS_BLOB_NAME)
        obtenir_utilisateurs.cache_clear()
        return True


def reactiver_utilisateur(user_id: int):
    """
    Réactive un utilisateur qui a été marqué comme 'deleted'.
    """
    users_df = charger_df_depuis_blob(blob_name=USERS_BLOB_NAME)

    if users_df.empty:
        return False

    # Si la colonne status n'existe pas, aucun utilisateur ne peut être réactivé.
    if "status" not in users_df.columns:
        return False

    # Vérifier si l'utilisateur existe et est bien 'deleted'
    user_mask = (users_df["user_id"] == user_id) & (users_df["status"] == "deleted")
    if user_mask.any():
        users_df.loc[user_mask, "status"] = "active"
        sauvegarder_df_vers_blob(users_df, USERS_BLOB_NAME)
        logger.info(f"Utilisateur {user_id} a été réactivé.")
        obtenir_utilisateurs.cache_clear()
        return True

    return False


@timed_lru_cache(seconds=300)  # Cache de 5 minutes
def obtenir_utilisateurs():
    """Récupère une liste unifiée d'utilisateurs depuis les clics et le fichier des utilisateurs."""
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)
    users_df = charger_df_depuis_blob(blob_name=USERS_BLOB_NAME)

    # Unifier les IDs des deux sources
    click_user_ids = set(clicks_df["user_id"].unique()) if not clicks_df.empty else set()
    manual_user_ids = set(users_df["user_id"].unique()) if not users_df.empty else set()
    all_user_ids = sorted(list(click_user_ids.union(manual_user_ids)))

    # Filtrer les utilisateurs supprimés
    if not users_df.empty and "status" in users_df.columns:
        deleted_users = set(users_df[users_df["status"] == "deleted"]["user_id"])
        active_user_ids = [uid for uid in all_user_ids if uid not in deleted_users]
    else:
        active_user_ids = all_user_ids

    return [{"user_id": uid} for uid in active_user_ids]


def obtenir_tous_les_utilisateurs_avec_statut():
    """
    Récupère la liste de tous les utilisateurs (actifs et supprimés) avec leur statut.
    Destiné à la page d'administration.
    """
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)
    users_df = charger_df_depuis_blob(blob_name=USERS_BLOB_NAME)

    # Unifier les IDs des deux sources
    click_user_ids = set(clicks_df["user_id"].unique()) if not clicks_df.empty else set()
    manual_user_ids = set(users_df["user_id"].unique()) if not users_df.empty else set()
    all_user_ids = sorted(list(click_user_ids.union(manual_user_ids)))

    # Créer un dictionnaire de statuts à partir de users.df
    status_map = {}
    if not users_df.empty and "status" in users_df.columns:
        # S'assurer qu'il n'y a pas de doublons d'ID utilisateur pour éviter les erreurs
        # lors de la création de la série. On garde la dernière occurrence.
        users_status_df = users_df.drop_duplicates(subset=["user_id"], keep="last")
        status_map = pd.Series(users_status_df.status.values, index=users_status_df.user_id).to_dict()

    # Construire la liste finale avec le statut (par défaut 'active')
    result = [{"user_id": uid, "status": status_map.get(uid, "active")} for uid in all_user_ids]
    return result


def obtenir_contexte_utilisateur(user_id: int):
    """Récupère le pays et le groupe d'appareils du dernier clic de l'utilisateur."""
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)

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
    articles_df = charger_df_depuis_blob(blob_name=ARTICLES_BLOB_NAME)

    if articles_df.empty:
        new_article_id = 1  # Démarrer à 1 si aucun article n'existe
    else:
        # Assurer que 'article_id' est numérique pour max()
        articles_df["article_id"] = pd.to_numeric(articles_df["article_id"], errors="coerce").fillna(0).astype(int)
        new_article_id = int(articles_df["article_id"].max()) + 1

    # Calculer le nombre de mots à partir du contenu
    words_count = len(content.split())

    new_article = pd.DataFrame(
        [
            {
                "article_id": new_article_id,
                "category_id": category_id,
                "created_at_ts": int(pd.Timestamp.now().timestamp()),
                "publisher_id": 0,  # publisher_id n'est pas fourni par le front, on met une valeur par défaut
                "words_count": words_count,
                # Note: 'title' et 'content' ne sont pas dans le schéma fourni, mais sont conservés pour l'affichage.
                "title": title,
            }
        ]
    )
    updated_articles_df = (
        pd.concat([articles_df, new_article], ignore_index=True) if not articles_df.empty else new_article
    )
    sauvegarder_df_vers_blob(updated_articles_df, ARTICLES_BLOB_NAME)
    return new_article_id


def obtenir_statut_reentrainement():
    """Récupère le statut actuel du processus de ré-entraînement."""
    blob_service_client = recuperer_client_blob_service()
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=STATUS_BLOB_NAME)
    try:
        downloader = blob_client.download_blob()
        status_data = downloader.readall()
        return pd.read_json(StringIO(status_data.decode("utf-8")), typ="series").to_dict()
    except ResourceNotFoundError:
        # Si le fichier n'existe pas, retourner un statut par défaut
        return {"status": "idle", "last_update": "N/A"}
    except Exception as e:
        logger.error(f"Impossible de lire le statut de ré-entraînement : {e}")
        return {"status": "unknown", "error": str(e)}


def obtenir_performance_modele():
    """Récupère les logs de performance de l'entraînement du modèle."""
    performance_df = charger_df_depuis_blob(blob_name=TRAINING_LOG_BLOB_NAME)
    if performance_df.empty:
        return []
    return performance_df.to_dict(orient="records")


@timed_lru_cache(seconds=900)  # Cache de 15 minutes
def obtenir_tendances_globales_clics():
    """Récupère et agrège les données de clics pour les tendances globales par pays et par groupe d'appareils."""
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)

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
def _get_article_context() -> pd.DataFrame:  # type: ignore
    """Calcule et cache le contexte des articles (pays/appareils uniques) à partir des clics."""
    clicks_df = charger_df_depuis_blob(blob_name=CLICKS_BLOB_NAME)
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
