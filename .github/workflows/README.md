# 📚 Système de Recommandation de Contenu

Ce projet est une application web complète qui fournit des recommandations de contenu personnalisées aux utilisateurs. Il est construit avec une architecture moderne et découplée, entièrement hébergée sur Microsoft Azure.

## 🏛️ Architecture

L'application est composée des services Azure suivants :

-   **Frontend (Interface Utilisateur)** : Une application [Streamlit](https://streamlit.io/) hébergée sur **Azure App Service**. Elle permet aux utilisateurs de se connecter, d'obtenir des recommandations, de noter des articles et de consulter leur historique.
-   **Backend (API de Recommandation)** : Une **Azure Function** qui expose une API REST. Elle reçoit un ID utilisateur et retourne une liste de recommandations personnalisées.
-   **Stockage de Données** : Un **Azure Blob Storage** qui stocke toutes les données brutes sous forme de fichiers CSV (utilisateurs, articles, interactions, logs d'entraînement).
-   **Gestion des Secrets** : Un **Azure Key Vault** qui stocke de manière sécurisée les informations sensibles comme la chaîne de connexion au stockage et l'URL de l'API.
-   **Identité et Authentification** : Les **Identités Managées** d'Azure sont utilisées pour permettre à l'App Service et à l'Azure Function de s'authentifier de manière sécurisée auprès du Key Vault sans stocker de secrets dans le code.
-   **Déploiement Continu (CI/CD)** : **GitHub Actions** est utilisé pour automatiser le déploiement du frontend sur l'App Service à chaque modification du code sur la branche `main`.

## 📁 Structure du Projet

```
recommandation-contenu/
├── .github/workflows/
│   └── deploy-frontend.yml   # Workflow de déploiement du frontend
├── .streamlit/
│   └── secrets.toml          # Fichier de secrets pour le développement local
├── frontend/
│   └── interface.py          # Code de l'application Streamlit
├── backend/
│   └── ...                   # (Emplacement pour le code de l'Azure Function)
├── .gitignore
├── README.md
└── requirements.txt          # Dépendances Python du projet
```

## 🚀 Démarrage Rapide (Développement Local)

### Prérequis

-   Python 3.11 ou supérieur
-   Un compte Azure
-   Azure CLI

### Étapes d'installation

1.  **Cloner le dépôt**
    ```bash
    git clone <URL_DU_DEPOT>
    cd recommandation-contenu
    ```

2.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

3.  **S'authentifier sur Azure**
    Pour que `DefaultAzureCredential` fonctionne localement, connectez-vous via l'Azure CLI.
    ```bash
    az login
    ```

4.  **Configurer les secrets locaux**
    Créez un fichier `.streamlit/secrets.toml` et ajoutez-y l'URL de votre Key Vault. Votre compte utilisateur doit avoir les permissions "get" et "list" sur les secrets du Key Vault.
    ```toml
    # .streamlit/secrets.toml
    KEY_VAULT_URL = "https://<NOM_DE_VOTRE_KEY_VAULT>.vault.azure.net/"
    ```

5.  **Lancer l'application**
    ```bash
    streamlit run frontend/interface.py
    ```

## ☁️ Déploiement sur Azure

Le déploiement du frontend est entièrement automatisé via GitHub Actions.

### 1. Préparation de l'infrastructure Azure

Assurez-vous que les ressources suivantes sont créées sur Azure :

-   Un groupe de ressources (ex: `rg-recommandation-contenu`).
-   Un compte de stockage avec un conteneur (ex: `reco-data`).
-   Un Key Vault avec les secrets `STORAGE-CONNECTION-STRING` et `API-URL`.
-   Une Azure Function pour le backend.
-   Un **App Service** nommé `reco-contenu-interface` pour le frontend.

### 2. Configuration de l'App Service

L'App Service doit être configuré pour fonctionner correctement :

-   **Identité Managée** : Activez l'identité managée affectée par le système.
-   **Permissions Key Vault** : Donnez à cette identité le rôle `Utilisateur des secrets Key Vault` sur votre Key Vault.
-   **Commande de démarrage** : Dans la configuration de l'App Service, définissez la commande de démarrage :
    ```
    streamlit run frontend/interface.py --server.port 8000 --server.address 0.0.0.0
    ```
-   **Variable d'environnement** : Ajoutez une variable d'environnement `KEY_VAULT_URL` avec l'URL de votre Key Vault.

### 3. Configuration de GitHub Actions

1.  **Créer un Principal de Service** : Suivez la documentation Azure pour créer un principal de service ayant le rôle `Contributeur` sur votre groupe de ressources.

2.  **Ajouter le secret à GitHub** :
    -   Allez dans `Settings` > `Secrets and variables` > `Actions` sur votre dépôt GitHub.
    -   Créez un nouveau secret nommé `AZURE_CREDENTIALS`.
    -   Collez le JSON de sortie de la commande de création du principal de service.

### 4. Déployer

Poussez simplement vos modifications sur la branche `main`. GitHub Actions se chargera de construire et de déployer automatiquement votre application sur l'App Service.

```bash
git push origin main
```

---