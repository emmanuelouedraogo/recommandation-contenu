# Système de Recommandation de Contenu

Ce projet met en œuvre un système de recommandation d'articles pour les utilisateurs, exposé via une API RESTful construite avec FastAPI. Le modèle de recommandation est un système hybride qui combine des approches basées sur le contenu (Content-Based) et le filtrage collaboratif (Collaborative Filtering) pour fournir des suggestions personnalisées.

## ✨ Fonctionnalités

- **Modèle Hybride** : Combine la puissance du filtrage collaboratif (SVD++) et des méthodes basées sur le contenu avec une décroissance temporelle pour des recommandations pertinentes.
- **Gestion du Démarrage à Froid (Cold Start)** : Pour les nouveaux utilisateurs, le système recommande les articles les plus populaires.
- **API RESTful** : Une interface simple et robuste basée sur FastAPI pour obtenir des recommandations.
- **Conteneurisation** : L'application est entièrement conteneurisée avec Docker pour un déploiement facile et reproductible.
- **Déploiement Continu (CI/CD)** : Un pipeline GitHub Actions est configuré pour automatiser la construction de l'image Docker et le déploiement sur Azure App Service.

## 🛠️ Technologies Utilisées

- **Backend** : Python, FastAPI
- **Data Science** : Pandas, NumPy, Scikit-learn, Surprise
- **Conteneurisation** : Docker
- **Cloud & DevOps** : Azure App Service, Azure Container Registry, GitHub Actions

## 📂 Structure du Projet

```
recommandation-contenu/
├── .github/
│   └── workflows/
│       └── ci.yml        # Pipeline de CI/CD pour le déploiement sur Azure
├── api/
│   ├── api.py            # Logique de l'API FastAPI
│   └── Dockerfile        # Fichier pour construire l'image Docker de l'API
├── data/                 # Données brutes et traitées (non versionné)
├── notebooks/            # Notebooks Jupyter pour l'exploration et l'entraînement du modèle
├── save/
│   └── hybrid_recommender_pipeline.pkl # Modèle de recommandation entraîné
├── models.py             # Implémentation des différents modèles de recommandation
├── requirements.txt      # Dépendances Python du projet
└── README.md             # Ce fichier
```

## 🚀 Démarrage Rapide

### Prérequis

- Docker
- Un compte Azure avec les permissions pour créer un groupe de ressources, un Container Registry et un App Service.
- Git

### 1. Configuration Locale (via Docker)

1.  **Cloner le dépôt :**
    ```bash
    git clone <URL_DU_DEPOT>
    cd recommandation-contenu
    ```

2.  **Construire l'image Docker :**
    Assurez-vous que votre modèle entraîné `hybrid_recommender_pipeline.pkl` est présent dans le dossier `save/`.
    ```bash
    docker build -t recommandation-api -f api/Dockerfile .
    ```

3.  **Lancer le conteneur :**
    ```bash
    docker run -d -p 8000:8000 --name api-reco recommandation-api
    ```

4.  L'API est maintenant accessible à l'adresse `http://localhost:8000`.

### 2. Déploiement sur Azure

Le déploiement est automatisé via le fichier `.github/workflows/ci.yml`.

1.  **Configurer les secrets GitHub :**
    Dans les paramètres de votre dépôt GitHub (`Settings > Secrets and variables > Actions`), ajoutez les secrets suivants :
    - `AZURE_CREDENTIALS` : Les identifiants de votre principal de service Azure au format JSON.
    - `AZURE_CONNECTION_STRING` : La chaîne de connexion à votre compte de stockage Azure (si nécessaire pour le modèle).

2.  **Mettre à jour les variables d'environnement :**
    Modifiez les variables dans le fichier `.github/workflows/ci.yml` pour correspondre à votre configuration Azure :
    - `AZURE_RESOURCE_GROUP`
    - `AZURE_ACR_NAME`
    - `AZURE_APP_SERVICE_NAME`

3.  **Pousser sur la branche `main` :**
    Chaque `push` sur la branche `main` déclenchera le workflow qui construira et déploiera automatiquement l'application sur Azure App Service.

## 📖 Utilisation de l'API

L'API expose la documentation Swagger/OpenAPI à l'endpoint `/docs`.

### Obtenir des recommandations

- **Endpoint** : `POST /recommendations/`
- **Description** : Retourne une liste de 5 articles recommandés pour un utilisateur donné.
- **Corps de la requête** (`JSON`) :
  ```json
  {
    "user_id": 123
  }
  ```
- **Réponse** (`JSON`) :
  ```json
  [
    {
      "article_id": 456,
      "final_score": 0.89
    },
    {
      "article_id": 789,
      "final_score": 0.75
    }
  ]
  ```

### Vérifier l'état de santé de l'API

- **Endpoint** : `GET /health/`
- **Description** : Vérifie si l'API est en cours d'exécution et si le modèle est chargé.
- **Réponse** (`JSON`) :
  ```json
  {
    "status": "ok"
  }
  ```