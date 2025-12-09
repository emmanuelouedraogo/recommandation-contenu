# 📚 Système de Recommandation de Contenu

Ce projet est une application web complète pour un système de recommandation de contenu. L'interface, développée avec Streamlit, permet aux utilisateurs de recevoir des recommandations personnalisées, de noter des articles et de consulter l'historique de leurs interactions. L'ensemble est déployé sur Microsoft Azure et utilise une architecture cloud moderne et sécurisée.

## 🏛️ Architecture

L'application est conçue autour des services Azure et d'une automatisation via GitHub Actions.

*   **Frontend** : Une application **Streamlit** interactive déployée sur **Azure App Service**. Elle constitue l'interface utilisateur principale.
*   **Backend API** : Un service d'API (non inclus dans ce dépôt) qui calcule et fournit les recommandations en temps réel.
*   **Stockage de Données** : **Azure Blob Storage** est utilisé pour stocker les fichiers CSV contenant les données des utilisateurs, des articles, et des interactions (clics).
*   **Gestion des Secrets** : **Azure Key Vault** stocke de manière centralisée et sécurisée les secrets de l'application, comme l'URL du compte de stockage et l'URL de l'API.
*   **Authentification Inter-Services** : Les **Identités Managées Azure** sont utilisées pour permettre à l'App Service de s'authentifier de manière sécurisée auprès du Key Vault et du Blob Storage, sans avoir besoin de stocker de mots de passe ou de clés dans le code.
*   **CI/CD** : Un workflow **GitHub Actions** (`.github/workflows/deploy-frontend.yml`) automatise entièrement le processus de déploiement.

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

## ⚙️ Configuration et Déploiement

Le déploiement est entièrement automatisé par le workflow GitHub Actions défini dans `.github/workflows/deploy-frontend.yml`.

### Prérequis

1.  Un compte Azure avec les permissions nécessaires pour créer et gérer des ressources.
2.  Un dépôt GitHub.

### Secrets Requis

Pour que le déploiement fonctionne, les secrets suivants doivent être configurés dans les **Paramètres du dépôt GitHub** (`Settings > Secrets and variables > Actions`):

1.  `AZURE_CREDENTIALS` : Les informations d'identification d'un Principal de Service (Service Principal) Azure, au format JSON, ayant les permissions de contribuer sur le groupe de ressources.

    ```json
    {
      "clientId": "...",
      "clientSecret": "...",
      "subscriptionId": "...",
      "tenantId": "..."
    }
    ```

2.  `KEY_VAULT_URL` : L'URL du coffre de secrets Azure (Key Vault) où sont stockés les secrets de l'application.
    *   Exemple : `https://mon-coffre-secret.vault.azure.net/`

### Secrets dans Azure Key Vault

Le Key Vault doit contenir les secrets suivants, auxquels l'Identité Managée de l'App Service doit avoir accès (rôle `Key Vault Secrets User`) :

*   `STORAGE-ACCOUNT-URL` : L'URL du service Blob du compte de stockage Azure.
*   `API-URL` : L'URL de base de l'API de recommandation.

### Déclenchement du Workflow

Le workflow se déclenche automatiquement à chaque `push` sur la branche `main` si des fichiers dans le dossier `frontend/` ou le workflow lui-même ont été modifiés.

Le workflow effectue les actions suivantes :
1.  Se connecte à Azure.
2.  Configure l'infrastructure :
    *   Met à jour le plan App Service vers le SKU `S1`.
    *   Active le Health Check.
    *   Définit les variables d'environnement (`KEY_VAULT_URL`, etc.).
    *   Configure les règles de mise à l'échelle automatique.
3.  Attend 45 secondes pour la stabilisation des services Azure.
4.  Installe les dépendances Python, empaquette l'application et la déploie sur Azure App Service.

## 📖 Comment Utiliser l'Application

1.  Accédez à l'URL de l'application déployée (`https://reco-contenu-interface.azurewebsites.net`).
2.  Utilisez la barre latérale pour vous connecter avec un `user_id` existant (une liste est affichée pour faciliter les tests) ou créez un nouveau compte via le menu "Créer un compte".
3.  Une fois connecté, la page "Recommandations" affichera des articles personnalisés.
4.  Vous pouvez noter chaque article. Vos notes apparaîtront dans la page "Mon Historique".