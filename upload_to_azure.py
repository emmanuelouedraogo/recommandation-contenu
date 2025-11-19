import os
import streamlit as st
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# --- Configuration ---
# Récupérez ces informations depuis votre compte Azure
# Il est recommandé d'utiliser des variables d'environnement ou les secrets Streamlit
# pour ne pas les écrire en dur dans le code.
AZURE_CONNECTION_STRING = st.secrets.get("AZURE_CONNECTION_STRING", "VOTRE_CHAINE_DE_CONNEXION_AZURE_ICI")
AZURE_CONTAINER_NAME = "reco-data"

# Chemin vers votre dossier de données local
LOCAL_DATA_PATH = "data"

def upload_file_to_blob(blob_service_client, container_name, local_file_path, blob_name):
    """
    Charge un fichier local vers un blob Azure.

    :param blob_service_client: Le client du service blob.
    :param container_name: Le nom du conteneur.
    :param local_file_path: Le chemin complet du fichier local.
    :param blob_name: Le nom/chemin du blob dans le conteneur.
    """
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        print(f"Chargement de '{local_file_path}' vers le blob '{blob_name}'...")
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print("Chargement terminé.")
    except Exception as e:
        print(f"Erreur lors du chargement de {local_file_path}: {e}")

def main():
    """
    Fonction principale pour charger tous les fichiers de données vers Azure.
    """
    if AZURE_CONNECTION_STRING == "VOTRE_CHAINE_DE_CONNEXION_AZURE_ICI":
        print("ERREUR : Veuillez configurer votre AZURE_CONNECTION_STRING.")
        return

    # 1. Créer le client de service blob
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    # 2. Charger les fichiers principaux
    # Fichier de métadonnées des articles
    upload_file_to_blob(blob_service_client, AZURE_CONTAINER_NAME,
                        os.path.join(LOCAL_DATA_PATH, 'articles_metadata.csv'),
                        'articles_metadata.csv')

    # Fichier d'embeddings
    upload_file_to_blob(blob_service_client, AZURE_CONTAINER_NAME,
                        os.path.join(LOCAL_DATA_PATH, 'articles_embeddings.pickle'),
                        'articles_embeddings.pickle')
    
    # Fichier de clics (échantillon)
    # Note: Le nom local est 'clicks_sample.csv' dans le dossier 'data'
    clicks_sample_path = os.path.join(LOCAL_DATA_PATH, 'clicks_sample.csv')
    if os.path.exists(clicks_sample_path):
        upload_file_to_blob(blob_service_client, AZURE_CONTAINER_NAME,
                            clicks_sample_path,
                            'clicks_sample.csv')
    else:
        print(f"AVERTISSEMENT : Le fichier '{clicks_sample_path}' n'a pas été trouvé.")

    print("\n--- Processus de chargement terminé ! ---")

if __name__ == "__main__":
    main()

```

### Comment utiliser ce script :

#1.  **Placez** ce fichier `upload_to_azure.py` à la racine de votre projet `recommandation-contenu/`.
#2.  **Installez `tqdm`** si ce n'est pas déjà fait :
#   ```bash
#   pip install tqdm
   ```
#3.  **Configurez votre chaîne de connexion Azure**. Le script essaiera de la lire depuis les secrets de Streamlit. Si vous l'exécutez localement, vous pouvez soit la définir comme variable d'environnement (`AZURE_CONNECTION_STRING`), soit la remplacer temporairement dans le script.
#4.  **Exécutez le script** depuis votre terminal :
#   ```bash
#   python upload_to_azure.py
   ```

#Le script va maintenant parcourir vos fichiers locaux et les envoyer vers votre conteneur Azure, prêt à être utilisé par vos autres scripts.

#<!--
#[PROMPT_SUGGESTION]Maintenant, modifie le script `reco_model_script.py` pour qu'il lise les données depuis Azure au lieu du système de fichiers local.[/PROMPT_SUGGESTION]
#[PROMPT_SUGGESTION]Comment puis-je exécuter ce script de chargement automatiquement à intervalles réguliers en utilisant les services Azure ?[/PROMPT_SUGGESTION]
