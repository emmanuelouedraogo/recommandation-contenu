# API de Recommandation de Contenu - Azure Functions
   
Ce projet déploie une API de recommandation de contenu en tant qu'application "serverless" sur Azure Functions.

L'API expose un endpoint `/api/recommend` qui prend un `user_id` en paramètre et retourne une liste de recommandations générées par un modèle de machine learning. Le modèle est automatiquement téléchargé depuis Azure Blob Storage au démarrage de la fonction.

## Architecture

- **Hébergement** : Azure Functions (Plan Consommation)
- **Langage** : Python 3.10
- **Déploiement** : CI/CD avec GitHub Actions
- **Stockage du modèle** : Azure Blob Storage

## Prérequis

- Python 3.10+
- Azure Functions Core Tools
- Un compte Azure avec un abonnement actif

## Installation et exécution locale

1.  **Cloner le dépôt**
    ```bash
    git clone <url-du-depot>
    cd recommandation-contenu
    ```

2.  **Créer un environnement virtuel et installer les dépendances**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Linux/macOS
    # .\.venv\Scripts\activate  # Sur Windows
    pip install -r requirements.txt
    ```

3.  **Configurer les paramètres locaux**
    - Créez un fichier `local.settings.json` à la racine du projet.
    - Copiez le contenu ci-dessous et remplacez la valeur de `AZURE_CONNECTION_STRING` par votre chaîne de connexion au compte de stockage Azure.

    ```json
    {
      "IsEncrypted": false,
      "Values": {
        "AzureWebJobsStorage": "",
        "FUNCTIONS_WORKER_RUNTIME": "python",
        "AZURE_CONNECTION_STRING": "Collez-votre-chaîne-de-connexion-ici"
      }
    }
    ```

4.  **Lancer l'application localement**
    ```bash
    func start
    ```
    L'API sera accessible à l'adresse `http://localhost:7071/api/recommend`.

## Endpoint de l'API

L'API expose un unique endpoint GET pour obtenir des recommandations.

- **URL** : `/api/recommend`
- **Méthode** : `GET`
- **Paramètre de requête** :
  - `user_id` (obligatoire) : L'identifiant de l'utilisateur pour lequel générer les recommandations.
- **Exemple d'appel (une fois déployé)** :
  `https://<nom-de-votre-app>.azurewebsites.net/api/recommend?user_id=123`

## Déploiement

Le déploiement est automatisé via le workflow GitHub Actions défini dans `.github/workflows/ci.yml`. Un simple `push` sur la branche `main` déclenchera la construction et le déploiement de l'application sur Azure Functions.