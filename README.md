# 📚 Système de Recommandation de Contenu

Ce projet est un système de recommandation de contenu complet, composé d'une **interface web front-end** et d'une **API back-end** pour la prédiction. Il offre une interface utilisateur riche pour visualiser des recommandations, interagir avec des articles, et consulter des statistiques.

L'architecture est découplée : l'interface web (servie par Flask) communique avec une API de machine learning (déployée sur Azure Functions) qui exécute les modèles de recommandation.

## Table des Matières
1.  Vue d'ensemble
2.  Stack Technique
3.  Structure des Fichiers
4.  Fonctionnalités de l'Interface
5.  Guide de l'API
6.  Installation et Déploiement
7.  Améliorations et Prochaines Étapes

---

## 🎯 Vue d'ensemble

Ce projet fournit une interface web complète pour interagir avec un système de recommandation. Il ne s'agit pas seulement d'un outil de visualisation, mais aussi d'une plateforme d'administration et de monitoring. Les utilisateurs peuvent obtenir des recommandations personnalisées, tandis que les administrateurs peuvent gérer le contenu et surveiller la santé et les performances du système.

L'architecture est composée de deux services principaux :
1.  **Interface Web (Flask)** : Une application web qui sert l'interface utilisateur (HTML/CSS/JS) et gère l'authentification des administrateurs.
2.  **API de Recommandation (Azure Function)** : Un service serverless qui charge un modèle de machine learning et expose un endpoint pour générer des recommandations.

## 💻 Stack Technique

- **Interface Web & Serveur** :
  - **Python / Flask** : Pour servir l'application web et gérer les templates.
  - **Flask-WTF** : Pour la protection contre les attaques CSRF.
  - **HTML5, CSS3, JavaScript (ES6+)** : Pour la structure, le style et la logique de l'interface.
  - **Chart.js** : Pour la visualisation des données.

- **API & Machine Learning** :
  - **Azure Functions** : Pour héberger l'API de recommandation de manière serverless.
  - **Azure Blob Storage** : Pour stocker les modèles de ML (`.pkl`) et les datasets.
  - **Pandas, Scikit-learn, Joblib** : Pour la manipulation des données et l'utilisation du modèle.

- **CI/CD & Déploiement** :
  - **GitHub Actions** : Pour l'intégration continue, les tests (linting) et le déploiement automatisé sur Azure.

## 📂 Structure des Fichiers

```
recommandation-contenu/
├── static/
│   ├── js/
│   │   └── main.js       # Fichier principal contenant toute la logique JavaScript
│   └── style.css         # Feuille de style principale
├── templates/
│   └── index.html        # Fichier HTML unique servant de template de base
├── app.py                # (Hypothétique) Serveur Flask pour l'API et le service des templates
└── README.md             # Ce fichier
```

## ✨ Fonctionnalités Détaillées

#### Gestion des Utilisateurs
- **Connexion par ID** : L'utilisateur entre son ID. Un mécanisme de *debounce* (400ms) évite les appels API excessifs pendant la saisie.
- **Contexte Dynamique** : Une fois l'ID saisi, le contexte de l'utilisateur (pays, appareil) est récupéré et affiché.
- **Création/Suppression** : Des boutons permettent de créer un nouvel utilisateur (le nouvel ID est automatiquement inséré dans le champ de saisie) ou de supprimer l'utilisateur courant.

#### Navigation par Onglets
L'interface principale est organisée en trois onglets :
1.  **Recommandations** : Affiche les articles recommandés. Le contenu est automatiquement mis à jour lors d'un changement d'utilisateur ou de filtre.
2.  **Historique** : Affiche la liste des articles déjà notés par l'utilisateur, avec la note et la date.
3.  **Tendances Globales** : Affiche des graphiques sur la répartition des clics par pays et par appareil. Le contenu de cet onglet est statique et ne dépend pas de l'utilisateur connecté.

#### Interactions et Recommandations
- **Notation d'articles** : Chaque carte de recommandation contient un menu déroulant pour noter l'article de 1 à 5. La soumission est gérée via la délégation d'événements pour optimiser les performances.
- **Filtrage** : Les recommandations peuvent être filtrées par pays et par appareil. La sélection d'un filtre déclenche automatiquement un nouvel appel API si l'onglet "Recommandations" est actif.

#### Panneau d'Administration
La barre latérale regroupe des outils d'administration :
- **Ajout d'Articles** : Un formulaire simple pour insérer de nouveaux articles dans le système.
- **Performances du Modèle** : Un bouton pour afficher un graphique linéaire montrant l'évolution des métriques de validation (`recall@10`, `precision@10`) par époque d'entraînement.
- **Statut du Réentraînement** : Un indicateur visuel dans l'en-tête, mis à jour toutes les 30 secondes, informe sur l'état du modèle (`Actif`, `Réentraînement en cours`, `Échec`).

## 🔌 Guide de l'API Back-end

Le front-end s'attend à ce que le back-end expose les endpoints suivants.

---

#### `POST /api/users`
- **Action** : Crée un nouvel utilisateur.
- **Réponse Succès (200)** : `{ "user_id": 123 }`
- **Réponse Erreur (500)** : `{ "error": "Impossible de créer l'utilisateur" }`

---

#### `DELETE /api/users/{userId}`
- **Action** : Désactive un utilisateur.
- **Réponse Succès (200)** : `{ "message": "Utilisateur 123 désactivé" }`
- **Réponse Erreur (404)** : `{ "error": "Utilisateur non trouvé" }`

---

#### `GET /api/user_context/{userId}`
- **Action** : Récupère le contexte d'un utilisateur.
- **Réponse Succès (200)** : `{ "country": "France", "deviceGroup": "Desktop" }`
- **Réponse Erreur (404)** : `{ "error": "Contexte non trouvé pour l'utilisateur" }`

---

#### `GET /api/recommendations` (Note: a été changé, n'utilise plus de paramètre dans l'URL)
- **Action** : Récupère les recommandations.
- **Paramètres Query** : `user_id` (obligatoire), `country` (optionnel), `device` (optionnel).
- **Réponse Succès (200)** : `[ { "article_id": 1, "title": "...", "content": "..." }, ... ]` ou `[]` si aucune recommandation.

---

#### `POST /api/interactions`
- **Action** : Enregistre une interaction (notation).
- **Corps de la requête** : `{ "user_id": 123, "article_id": 456, "rating": 5 }`
- **Réponse Succès (200)** : `{ "message": "Interaction enregistrée" }`
- **Réponse Erreur (400)** : `{ "error": "Données invalides" }`

---

#### `GET /api/history/{userId}`
- **Action** : Récupère l'historique de notation.
- **Réponse Succès (200)** : `[ { "title": "...", "nb": 5, "click_timestamp": 1672531200 }, ... ]`

---

#### `POST /api/articles`
- **Action** : Ajoute un nouvel article.
- **Corps de la requête** : `{ "title": "...", "content": "...", "category_id": 2 }`
- **Réponse Succès (201)** : `{ "article_id": 789 }`

---

#### `GET /api/global_trends`
- **Action** : Récupère les agrégats pour les graphiques.
- **Réponse Succès (200)** : `{ "clicks_by_country": [...], "clicks_by_device": [...] }`

---

#### `GET /api/performance`
- **Action** : Récupère les métriques de performance du modèle.
- **Réponse Succès (200)** : `[ { "epoch": 1, "val_recall_at_10": 0.15, "val_precision_at_10": 0.08 }, ... ]`

---

#### `GET /api/retraining_status`
- **Action** : Vérifie le statut du réentraînement.
- **Réponse Succès (200)** : `{ "status": "idle" | "in_progress" | "failed" }`

---

## 🚀 Installation et Lancement

1.  **Clonez le dépôt** sur votre machine locale.
2.  **Créez un serveur Flask minimal** : Créez un fichier `app.py` à la racine du projet avec le contenu suivant pour servir l'application (ceci est un exemple de base sans la logique API) :
    ```python
    from flask import Flask, render_template

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    # ... Ajoutez ici les routes de l'API (ex: @app.route('/api/users', methods=['POST']))

    if __name__ == '__main__':
        app.run(debug=True)
    ```
3.  **Installez Flask** :
    ```bash
    pip install Flask
    ```
4.  **Lancez le serveur** :
    ```bash
    python app.py
    ```
5.  **Accédez à l'application** : Ouvrez votre navigateur et allez à l'adresse `http://127.0.0.1:5000`.

## 🛠️ Améliorations Possibles

- **Gestion d'Erreurs** : Remplacer les `alert()` et `console.error()` par un système de notifications (modales ou "toasts") non bloquant pour une meilleure expérience utilisateur.
- **Authentification** : Mettre en place un vrai système de connexion pour sécuriser les actions d'administration.
- **Pagination** : Ajouter une pagination pour l'historique des utilisateurs et les listes d'articles si elles deviennent longues.
- **Tests** : Écrire des tests unitaires et d'intégration pour la logique JavaScript afin d'assurer la robustesse du code.
- **Composants Web** : Refactoriser les éléments répétitifs (comme les cartes d'articles) en composants Web pour une meilleure réutilisabilité.