# -*- coding: utf-8 -*-

"""
Description des Bases de Données du Système de Recommandation

Ce fichier a pour but de documenter la structure des données utilisées par l'application
de recommandation. Les données sont stockées dans des fichiers CSV et Pickle situés
dans des sous-dossiers spécifiques, reflétant une organisation modulaire.

Structure des dossiers :
-----------------------
data/
├── articles_metadata.csv
├── articles_embeddings.pickle
└── clicks/
    └── clicks/
        ├── clicks_hour_000.csv
        ├── clicks_hour_001.csv
        └── ... (jusqu'à 364)


Fichiers de données (Tables) :
------------------------------
1. data/articles_metadata.csv      -> Table `articles`
2. data/articles_embeddings.pickle -> Table `embeddings`
3. data/clicks/clicks/             -> Table `interactions` (ou `clicks`)

Note : Une table des utilisateurs (`users`) est gérée par l'application principale
(`app.py`) via le fichier `data/users.csv`.

"""

# ==============================================================================
# 1. Table `articles` (stockée dans data/articles_metadata.csv)
# ==============================================================================

"""
Description :
-------------
Ce fichier est le catalogue central de tous les contenus disponibles dans le système.
Chaque ligne représente un article unique avec ses métadonnées associées.

Chemin : `data/articles_metadata.csv`

Contenu / Colonnes possibles :
------------------------------
- article_id (Clé Primaire) :
  - Type : Entier (Integer)
  - Description : Identifiant numérique unique pour chaque article. C'est la clé
    qui sera utilisée pour joindre cette table avec les interactions.

- title :
  - Type : Chaîne de caractères (String)
  - Description : Le titre de l'article.

- content / text :
  - Type : Chaîne de caractères (String / Text)
  - Description : Le contenu textuel complet ou un résumé de l'article. Ce champ
    est crucial pour les modèles de recommandation basés sur le contenu (Content-Based)
    ou pour générer des aperçus dans l'interface.

- category_id :
  - Type : Entier (Integer)
  - Description : Identifiant de la catégorie de l'article (ex: 1 pour 'Sport',
    2 pour 'Technologie').

- created_at_ts :
  - Type : Timestamp (Integer)
  - Description : Timestamp Unix de la date de création de l'article, utile pour
    filtrer ou recommander des articles récents.

Informations utiles :
--------------------
- Ce fichier est la source de vérité pour tout ce qui concerne les articles.
- L'interface Streamlit permet d'ajouter de nouveaux articles dans ce fichier.

"""

# ==============================================================================
# 2. Table `embeddings` (stockée dans data/articles_embeddings.pickle)
# ==============================================================================

"""
Description :
-------------
Ce fichier contient les vecteurs de caractéristiques (embeddings) pré-calculés
pour chaque article. Ces embeddings sont des représentations numériques denses
des articles dans un espace vectoriel.

Chemin : `data/articles_embeddings.pickle`

Format :
--------
- Fichier binaire Python (pickle).
- Contient probablement un dictionnaire ou un DataFrame Pandas qui mappe un
  `article_id` à son vecteur d'embedding (par exemple, un tableau NumPy).

Informations utiles :
--------------------
- Ces embeddings sont essentiels pour les modèles de recommandation modernes
  (Collaborative Filtering, Content-Based, Hybrides).
- Ils permettent de calculer la similarité entre les articles, de trouver des
  articles similaires à ceux qu'un utilisateur a aimés, etc.
- Le modèle `items2vec` est un exemple d'algorithme qui peut générer de tels embeddings.

"""

# ==============================================================================
# 3. Table `interactions` (stockée dans data/clicks/clicks/)
# ==============================================================================

"""
Description :
-------------
Ce fichier journalise tous les événements d'interaction (clics, lectures, "j'aime")
entre les utilisateurs et les articles. C'est la source de données la plus critique
pour les modèles de filtrage collaboratif et pour le calcul de la popularité. Les
données sont réparties dans 365 fichiers, représentant chacun une heure d'interactions.

Chemin : `data/clicks/clicks/clicks_hour_XXX.csv`

Contenu / Colonnes possibles :
------------------------------
- user_id (Clé Étrangère) :
  - Type : Entier (Integer)
  - Référence : `users.user_id`
  - Description : L'identifiant de l'utilisateur qui a interagi.

- article_id (Clé Étrangère) :
  - Type : Entier (Integer)
  - Référence : `articles_metadata.article_id`
  - Description : L'identifiant de l'article avec lequel l'utilisateur a interagi.

- click_timestamp :
  - Type : Timestamp (Integer)
  - Description : Timestamp Unix du moment de l'interaction.

- interaction_score / event_type :
  - Type : Nombre (Float/Integer) ou Chaîne de caractères (String)
  - Description : Un score ou un type qui quantifie l'interaction.
    Exemples : 1.0 pour un simple clic, 5.0 pour un "j'aime", ou des chaînes
    comme 'click', 'like', 'read_long'.

Informations utiles :
--------------------
- Un utilisateur est considéré comme "nouveau" s'il n'a aucune entrée dans cette table.
- Les données de cette table sont utilisées pour :
  a) Entraîner les modèles de recommandation personnalisée (filtrage collaboratif).
  b) Calculer les articles les plus populaires pour les nouveaux utilisateurs.
- Pour être utilisées, ces données doivent être agrégées en un seul DataFrame.

"""

# ==============================================================================
# Relations entre les tables (Schéma Logique)
# ==============================================================================

"""
                                     +--------------------------+
                                (1)--|   articles_embeddings    |
                                     +--------------------------+
                                     | article_id          (PK) |
                                     | embedding_vector         |
                                     +--------------------------+
                                                 ^ (1)
                                                 |
  +-----------+      (1)----(N)      +--------------------+      (N)----(1)      +-----------------------+
  |   users   |                      |   clicks           |                      |   articles_metadata   |
  +-----------+                      +--------------------+                      +-----------------------+
  | user_id   | (PK) --<-- (FK) ---- | user_id            |                      | article_id            | (PK)
  +-----------+                      | article_id         | ---- (FK) --<--      | title                 |
                                     | click_timestamp    |                      | content               |
                                     | interaction_score  |                      | ...                   |
                                     +--------------------+                      +-----------------------+

1. Relation `users` <-> `interactions` (Un-à-Plusieurs) :
   - Un utilisateur peut avoir de zéro à plusieurs interactions (clics).
   - Chaque clic est associé à un et un seul utilisateur.
   - La clé `clicks.user_id` est une clé étrangère qui pointe vers `users.user_id`.

2. Relation `articles_metadata` <-> `clicks` (Un-à-Plusieurs) :
   - Un article peut être l'objet de zéro à plusieurs interactions.
   - Chaque clic concerne un et un seul article.
   - La clé `clicks.article_id` est une clé étrangère qui pointe vers `articles_metadata.article_id`.

3. Relation `articles_metadata` <-> `articles_embeddings` (Un-à-Un) :
   - Chaque article a un et un seul vecteur d'embedding.
   - La clé `articles_embeddings.article_id` est une clé étrangère qui pointe vers `articles_metadata.article_id`.

Ce modèle de données est classique pour un système de recommandation et permet de capturer
efficacement les préférences des utilisateurs et les caractéristiques des articles.

"""
