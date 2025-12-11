# Système de Recommandation de Contenu

[![Statut du Workflow CI/CD](https://github.com/VOTRE_NOM/VOTRE_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/VOTRE_NOM/VOTRE_REPO/actions/workflows/ci.yml)

Ce projet est une application web complète développée avec Flask qui fournit des recommandations de contenu personnalisées aux utilisateurs. Elle intègre une logique métier pour interagir avec des données stockées sur Azure Blob Storage, une API RESTful pour la communication frontend-backend, et un pipeline de déploiement continu (CI/CD) avec GitHub Actions vers Azure App Service.

## 🚀 Fonctionnalités

- **Recommandations Personnalisées** : Fournit des recommandations d'articles basées sur l'ID de l'utilisateur.
- **Historique Utilisateur** : Affiche les articles précédemment consultés par un utilisateur.
- **Gestion des Interactions** : Enregistre les nouvelles interactions (notations) des utilisateurs de manière performante via un système de logs.
- **Panneau d'Administration** : Une interface sécurisée pour visualiser, supprimer (soft delete) et réactiver des utilisateurs.
- **API RESTful** : Expose des endpoints clairs pour toutes les fonctionnalités.
- **Déploiement Automatisé** : Intégration continue et déploiement continu sur Azure App Service à chaque push sur la branche `main`.
- **Tests Unitaires** : Validation de la logique métier grâce à des tests unitaires avec `pytest`.

## 🏛️ Architecture

- **Backend** : **Flask**, une micro-framework Python, servant à la fois l'API et l'interface utilisateur.
- **Logique Métier** : **Pandas** pour la manipulation des données en mémoire.
- **Stockage de Données** : **Azure Blob Storage** pour la persistance des données (articles, clics, utilisateurs) au format Parquet, plus performant.
- **Authentification** :
  - **Identité Managée (Managed Identity)** pour une connexion sécurisée entre l'App Service et Azure Storage.
  - **Basic Auth** pour la protection du panneau d'administration.
- **CI/CD** : **GitHub Actions** pour automatiser les tests et le déploiement.
- **Hébergement** : **Azure App Service** pour l'exécution de l'application web en production.

## 📂 Structure du Projet

```
recommandation-contenu/
├── .github/
│   └── workflows/
│       └── ci.yml           # Pipeline de CI/CD pour les tests et le déploiement
├── templates/
│   ├── admin.html           # Page d'administration
│   └── index.html           # Page d'accueil
├── tests/
│   └── test_logic.py        # Tests unitaires pour la logique métier
├── app.py                   # Fichier principal de l'application Flask (routes API)
├── logic.py                 # Logique métier (interaction avec Azure, manipulation de données)
├── requirements.txt         # Dépendances de production
├── requirements-dev.txt     # Dépendances de développement (ex: pytest)
└── README.md                # Ce fichier
```

## ⚙️ Installation et Lancement Local

Suivez ces étapes pour exécuter le projet sur votre machine locale.

### 1. Prérequis

- Python 3.11 ou supérieur
- Un compte Azure avec les permissions pour créer un compte de stockage.

### 2. Cloner le Dépôt

```bash
git clone https://github.com/VOTRE_NOM/VOTRE_REPO.git
cd recommandation-contenu
```

### 3. Créer un Environnement Virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 4. Installer les Dépendances

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 5. Configurer les Variables d'Environnement

Créez un fichier `.env` à la racine du projet et ajoutez les variables suivantes. Pour le développement local, vous pouvez vous authentifier à Azure via l'Azure CLI (`az login`).

```
AZURE_STORAGE_ACCOUNT_NAME="nomdevotrestorage"
API_URL="http://127.0.0.1:8080" # URL de l'application locale
```

### 6. Lancer l'Application

```bash
python app.py
```

L'application sera accessible à l'adresse `http://127.0.0.1:8080`.

## 🧪 Tests

Pour exécuter la suite de tests unitaires, utilisez `pytest`. Les tests simulent les interactions avec Azure Blob Storage pour valider la logique de manière isolée.

```bash
python -m pytest
```

## 🚀 Déploiement (CI/CD)

Le déploiement est entièrement automatisé via le workflow GitHub Actions défini dans `.github/workflows/ci.yml`.

1.  **Déclencheur** : Un `push` sur la branche `main`.
2.  **Job `test`** : Les tests unitaires sont exécutés. Si un test échoue, le pipeline s'arrête.
3.  **Job `build-and-deploy`** : Si les tests réussissent, l'application est empaquetée et déployée sur l'Azure App Service configuré.

### Configuration Requise

- **Secret GitHub** : Vous devez configurer un secret nommé `AZURE_CREDENTIALS` dans les paramètres de votre dépôt GitHub. Ce secret contient les informations d'identification d'un principal de service Azure autorisé à déployer sur votre groupe de ressources.
- **Identité Managée** : L'Azure App Service doit avoir son identité managée activée et posséder le rôle **"Contributeur aux données Blob du stockage"** sur le compte de stockage pour pouvoir lire et écrire les données.

## 🔑 Page d'Administration

- **URL** : `/admin`
- **Identifiants par défaut** :
  - **Utilisateur** : `admin`
  - **Mot de passe** : `password`

> **⚠️ Avertissement de Sécurité** : Ces identifiants sont codés en dur. Pour un environnement de production sécurisé, il est impératif de les gérer via des variables d'environnement sur Azure App Service.

## 📖 Documentation de l'API

| Méthode | Endpoint                               | Protection | Description                                                              |
|---------|----------------------------------------|------------|--------------------------------------------------------------------------|
| `GET`   | `/api/users`                           | Aucune     | Récupère la liste des ID utilisateurs actifs.                            |
| `POST`  | `/api/users`                           | Aucune     | Crée un nouvel utilisateur.                                              |
| `GET`   | `/api/admin/users`                     | Admin      | Récupère tous les utilisateurs avec leur statut (actif/supprimé).        |
| `DELETE`| `/api/users/<int:user_id>`             | Admin      | Désactive un utilisateur (soft delete).                                  |
| `POST`  | `/api/users/<int:user_id>/reactivate`  | Admin      | Réactive un utilisateur désactivé.                                       |
| `GET`   | `/api/recommendations/<int:user_id>`   | Aucune     | Obtient les recommandations pour un utilisateur.                         |
| `GET`   | `/api/history/<int:user_id>`           | Aucune     | Obtient l'historique des interactions d'un utilisateur.                  |
| `POST`  | `/api/interactions`                    | Aucune     | Enregistre une nouvelle interaction (ex: notation d'un article).         |
| `GET`   | `/api/global_trends`                   | Aucune     | Récupère les tendances globales de clics.                                |
| `GET`   | `/api/performance`                     | Aucune     | Récupère les métriques de performance du modèle.                         |

---

*Ce README a été généré pour fournir une vue d'ensemble claire et fonctionnelle du projet.*