# 📚 Système de Recommandation de Contenu

Ce projet est une application web complète qui fournit des recommandations de contenu personnalisées. Il est construit avec une architecture moderne et découplée, entièrement hébergée sur Microsoft Azure et déployée via des pipelines CI/CD avec GitHub Actions.

## 🏛️ Architecture

L'application est conçue autour des services Azure et d'une automatisation via GitHub Actions.

*   **Frontend (Interface Utilisateur)** : Une application Streamlit hébergée sur **Azure App Service**. Elle permet aux utilisateurs de se connecter, d'obtenir des recommandations, de noter des articles et de consulter leur historique.
*   **Backend (API de Recommandation)** : Une **Azure Function** qui expose une API REST. Elle sert les recommandations générées par le modèle.
*   **Stockage de Données et Modèles** : Un **Azure Blob Storage** qui centralise les données brutes (CSV) et les modèles de machine learning entraînés.
*   **Gestion des Secrets** : Les secrets (`API_URL`, `STORAGE_CONNECTION_STRING`) sont stockés de manière sécurisée dans les **GitHub Secrets**. Le pipeline CI/CD les injecte en tant que variables d'environnement dans l'App Service lors du déploiement.
*   **Authentification** : Le pipeline CI/CD s'authentifie à Azure via un **Principal de Service (Service Principal)** pour configurer les ressources Azure.
*   **CI/CD** : Les workflows **GitHub Actions** automatisent le déploiement du frontend, du backend, et l'entraînement des modèles.

## ✨ Fonctionnalités

- **Connexion Utilisateur** : Système simple de connexion basé sur un `user_id`.
- **Recommandations Personnalisées** : Appel à une API backend pour récupérer et afficher une liste d'articles recommandés pour l'utilisateur connecté.
- **Notation d'Articles** : Possibilité pour l'utilisateur de noter les articles sur une échelle de 1 à 5.
- **Historique des Interactions** : Page dédiée où l'utilisateur peut consulter et modifier les notes qu'il a précédemment attribuées.
- **Création de Compte et d'Article** : Interfaces pour ajouter de nouveaux utilisateurs et de nouveaux articles à la base de données.
- **Performance du Modèle** : Visualisation de l'historique des entraînements du modèle de recommandation.
- **Haute Disponibilité** : L'infrastructure Azure est configurée pour la mise à l'échelle automatique (autoscaling) en fonction de la charge CPU.
- **Bilan de Santé (Health Check)** : Un point de terminaison `/health` permet à Azure de surveiller la disponibilité de l'application.

## 🚀 Technologies Utilisées

*   **Langage** : Python 3.11
*   **Framework Frontend** : Streamlit
*   **Librairies Principales** : Pandas, Requests, Flask
*   **Plateforme Cloud** : Microsoft Azure
    *   App Service
    *   Blob Storage
    *   Key Vault
    *   Monitor (pour l'autoscaling)
*   **CI/CD** : GitHub Actions

## 🚀 Démarrage Rapide (Développement Local)

### Prérequis

*   Python 3.11 ou supérieur
*   Un compte Azure
*   Azure CLI

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

3.  **Configurer les secrets locaux**
    Créez un fichier `.streamlit/secrets.toml` à la racine du projet. Ce fichier contiendra les informations de connexion nécessaires pour faire tourner l'application sur votre machine.
    ```toml
    # .streamlit/secrets.toml
    STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=..."
    API_URL = "http://localhost:7071/api/recommend"
    ```

4.  **Lancer l'application**
    ```bash
    streamlit run frontend/Accueil.py
    ```

## ☁️ Déploiement sur Azure

Le déploiement est entièrement automatisé par les workflows GitHub Actions.
 
### Prérequis

1.  Un compte Azure avec les permissions nécessaires pour créer et gérer des ressources.
2.  Un dépôt GitHub.

### Secrets Requis

Pour que les workflows fonctionnent, les secrets suivants doivent être configurés dans les **Paramètres du dépôt GitHub** (`Settings > Secrets and variables > Actions`):

1.  `AZURE_CREDENTIALS` : Les informations d'identification d'un Principal de Service (Service Principal) Azure, au format JSON, ayant les permissions de contribuer sur le groupe de ressources.

2.  `STORAGE_CONNECTION_STRING` : La chaîne de connexion complète pour le compte de stockage Azure.

3.  `API_URL` : L'URL de l'API de recommandation (Azure Function).

### Déclenchement du Workflow

Les workflows se déclenchent automatiquement à chaque `push` sur la branche `main`.

Le workflow effectue les actions suivantes :
1.  Se connecte à Azure.
2.  Configure l'infrastructure (si `setup-infra.yml` est lancé) :
    *   Met à jour le plan App Service vers le SKU `S1`.
    *   Active le Health Check et configure les variables d'environnement (`API_URL`, etc.) en utilisant les secrets GitHub.
    *   Configure les règles de mise à l'échelle automatique.
3.  Installe les dépendances Python.
4.  Empaquette et déploie l'application sur Azure App Service.

## 📖 Comment Utiliser l'Application

1.  Accédez à l'URL de l'application déployée (`https://reco-contenu-interface.azurewebsites.net`).
2.  Utilisez la barre latérale pour vous connecter avec un `user_id` existant (une liste est affichée pour faciliter les tests) ou créez un nouveau compte via le menu "Créer un compte".
3.  Une fois connecté, la page "Recommandations" affichera des articles personnalisés.
4.  Vous pouvez noter chaque article. Vos notes apparaîtront dans la page "Mon Historique".