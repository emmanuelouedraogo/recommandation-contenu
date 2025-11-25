# recommandation-contenu 

Ce projet implémente un système de recommandation de contenu (articles, livres) de bout en bout, incluant l'entraînement de modèles, une interface utilisateur interactive et un pipeline de ré-entraînement automatisé.

## 🚀 Fonctionnalités

* **Modèle Hybride Sophistiqué** : Combine le **filtrage collaboratif** (basé sur les interactions des utilisateurs) et le **filtrage basé sur le contenu** (basé sur la similarité sémantique des articles) pour des recommandations pertinentes.
* **Gestion du "Cold Start"** : Fournit des recommandations basées sur la popularité pour les nouveaux utilisateurs.
* **Interface Utilisateur Interactive** : Une application **Streamlit** permet aux utilisateurs de créer un compte et d'obtenir des recommandations personnalisées.
* **API Dédiée** : Les prédictions du modèle sont servies via une API **FastAPI**, découplant le front-end du back-end de machine learning.
* **Automatisation MLOps** : Un workflow **GitHub Actions** ré-entraîne automatiquement le modèle chaque semaine et le déploie sur Azure Blob Storage, garantissant que le système reste à jour.
* **Stockage Cloud** : Les données, les modèles et les informations utilisateurs sont stockés sur **Azure Blob Storage**.

## 🏛️ Architecture

Le projet est structuré en trois composants principaux :

1. **Pipeline d'Entraînement (`reco_model_script.py`)**

   * Charge les données brutes (clics, métadonnées, embeddings) depuis Azure.
   * Entraîne un modèle hybride qui pondère les scores d'un modèle SVDpp (collaboratif) et d'un modèle Content-Based (avec décroissance temporelle).
   * Sauvegarde le pipeline de modèle finalisé dans un fichier `hybrid_recommender_pipeline.pkl`.
2. **API de Recommandation (basée sur FastAPI - non incluse dans ce dépôt)**

   * Charge le modèle `.pkl` sauvegardé.
   * Expose un endpoint `/recommendations/` qui accepte un `user_id` et retourne une liste d'articles recommandés.
3. **Application Web (`app.py`)**

   * Fournit une interface utilisateur construite avec Streamlit.
   * Communique avec l'API FastAPI pour récupérer et afficher les recommandations.
   * Permet la création de nouveaux utilisateurs, en sauvegardant les informations sur Azure.

Le workflow **GitHub Actions (`retrain_model.yml`)** orchestre ce processus en exécutant périodiquement le script d'entraînement et en téléversant le nouveau modèle sur Azure, où l'API peut le charger.

---

## 🛠️ Installation et Lancement

### Prérequis

* Python 3.9 ou supérieur
* Un compte Azure avec un conteneur de stockage Blob.

### 1. Cloner le Dépôt

```bash
git clone https://github.com/emmanuelouedraogo/recommandation-contenu.git
cd recommandation-contenu
```

### 2. Installer les Dépendances

Il est recommandé d'utiliser un environnement virtuel.

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurer les Secrets

Pour que l'application fonctionne, vous devez configurer les secrets nécessaires.

#### Pour l'Application Streamlit

Créez un fichier `.streamlit/secrets.toml` avec le contenu suivant :

```toml
# .streamlit/secrets.toml

AZURE_CONNECTION_STRING = "votre_chaine_de_connexion_azure"
API_URL = "http://adresse_de_votre_api_fastapi:8000"
```

#### Pour le Workflow GitHub Actions

Dans votre dépôt GitHub, allez dans `Settings` > `Secrets and variables` > `Actions` et ajoutez les secrets suivants :

* `AZURE_CONNECTION_STRING` : Votre chaîne de connexion Azure.

### 4. Lancer l'Application Streamlit

Assurez-vous que votre API FastAPI est en cours d'exécution, puis lancez l'application Streamlit :

```bash
streamlit run app.py
```

Ouvrez votre navigateur à l'adresse indiquée (généralement `http://localhost:8501`).

---

## 🔄 Pipeline de Ré-entraînement

Le fichier `.github/workflows/retrain_model.yml` définit le pipeline CI/CD.

* **Déclenchement** :
  * **Manuel** : Peut être lancé à tout moment depuis l'onglet "Actions" de GitHub.
  * **Automatique** : S'exécute tous les dimanches à 2h00 du matin (UTC).
* **Processus** :
  1. Récupère le code source.
  2. Installe les dépendances Python.
  3. Exécute le script `reco_model_script.py` pour générer un nouveau fichier `save/hybrid_recommender_pipeline.pkl`.
  4. Téléverse ce fichier `.pkl` comme artefact de l'action pour l'archivage.
  5. Déploie le nouveau modèle sur Azure Blob Storage, écrasant la version précédente.

Ce mécanisme garantit que l'API de recommandation utilise toujours la version la plus récente du modèle sans aucune interruption de service.
