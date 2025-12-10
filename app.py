# app.py
import os
from flask import Flask, render_template, jsonify, request
import logging

# Importer les fonctions logiques depuis le nouveau fichier
import logic as logic

# Crée une instance de l'application Flask.
# En spécifiant template_folder et static_folder, on indique à Flask
# où trouver les fichiers frontend, même si app.py est à la racine.
app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

# --- Configuration ---
# Assurez-vous que ces variables d'environnement sont définies avant de lancer l'application
app.config["API_URL"] = os.environ.get("API_URL") # type: ignore
app.config["AZURE_STORAGE_ACCOUNT_NAME"] = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME") # type: ignore

logging.basicConfig(level=logging.INFO)

# --- Routes pour servir les pages HTML ---


@app.route("/")
def index():
    """Sert la page d'accueil principale."""
    return render_template("index.html")  # Cherchera dans 'frontend/templates/index.html'


@app.route("/health")
def health_check():
    """Point de terminaison pour le bilan de santé (Health Check)."""
    return "OK", 200


# --- Routes API pour le Frontend ---
@app.route("/api/users", methods=["GET", "POST"])
def handle_users():
    """Gère la récupération et la création d'utilisateurs."""
    if request.method == "POST":
        # Créer un nouvel utilisateur
        try:
            new_user_id = logic.creer_nouvel_utilisateur()
            return jsonify({"message": "Nouvel utilisateur créé avec succès", "user_id": new_user_id}), 201
        except Exception as e:
            app.logger.error(f"Erreur API POST /api/users: {e}")
            return jsonify({"error": "Impossible de créer un nouvel utilisateur"}), 500
    else:  # GET
        # Obtenir la liste des utilisateurs
        try:
            users = logic.obtenir_utilisateurs()
            return jsonify(users)
        except Exception as e:
            app.logger.error(f"Erreur API GET /api/users: {e}")
            return jsonify({"error": "Impossible de charger les utilisateurs"}), 500


@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id: int):
    """API pour obtenir les recommandations pour un utilisateur."""
    try:
        country_filter = request.args.get("country")
        device_filter = request.args.get("deviceGroup")

        recos = logic.obtenir_recommandations_pour_utilisateur(
            app.config["API_URL"],
            user_id,
            # app.config["STORAGE_CONNECTION_STRING"], # Removed, using Managed Identity
            country_filter=country_filter,
            device_filter=device_filter,
        )
        if "error" in recos:
            return jsonify(recos), 404
        return jsonify(recos)
    except Exception as e:
        app.logger.error(f"Erreur API /api/recommendations/{user_id}: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/api/history/<int:user_id>", methods=["GET"])
def get_user_history(user_id):
    """API pour obtenir l'historique de notation d'un utilisateur."""
    try:
        history = logic.obtenir_historique_utilisateur(user_id)
        return jsonify(history)
    except Exception as e:
        app.logger.error(f"Erreur API /api/history/{user_id}: {e}")
        return jsonify({"error": "Impossible de charger l'historique"}), 500


@app.route("/api/interactions", methods=["POST"])
def add_interaction():
    """API pour ajouter ou mettre à jour une notation d'article."""
    data = request.get_json()
    if not data or "user_id" not in data or "article_id" not in data or "rating" not in data:
        return jsonify({"error": "Données manquantes (user_id, article_id, rating)"}), 400

    try:
        user_id = int(data["user_id"])
        article_id = int(data["article_id"])
        rating = int(data["rating"])
        logic.ajouter_ou_mettre_a_jour_interaction(user_id=user_id, article_id=article_id, rating=rating)
        return jsonify({"message": "Interaction enregistrée avec succès"}), 200
    except ValueError:
        return jsonify({"error": "Les IDs et la note doivent être des entiers"}), 400
    except Exception as e:
        app.logger.error(f"Erreur API POST /api/interactions: {e}")
        return jsonify({"error": "Impossible d'enregistrer l'interaction"}), 500


@app.route("/api/articles", methods=["POST"])
def add_article():
    """API pour ajouter un nouvel article."""
    data = request.get_json()
    if not data or not data.get("title") or not data.get("content") or "category_id" not in data:
        return jsonify({"error": "Données manquantes (title, content, category_id)"}), 400

    try:
        title = data["title"]
        content = data["content"]
        category_id = int(data["category_id"])
        new_article_id = logic.creer_nouvel_article(title=title, content=content, category_id=category_id)
        return jsonify({"message": "Article ajouté avec succès", "article_id": new_article_id}), 201
    except ValueError:
        return jsonify({"error": "L'ID de catégorie doit être un nombre"}), 400
    except Exception as e:
        app.logger.error(f"Erreur API POST /api/articles: {e}")
        return jsonify({"error": "Impossible d'ajouter l'article"}), 500


@app.route("/api/performance", methods=["GET"])
def get_model_performance():
    """API pour obtenir les données de performance du modèle."""
    try:
        performance_data = logic.obtenir_performance_modele()
        return jsonify(performance_data)
    except Exception as e:
        app.logger.error(f"Erreur API GET /api/performance: {e}")
        return jsonify({"error": "Impossible de charger les données de performance"}), 500


@app.route("/api/user_context/<int:user_id>", methods=["GET"])
def get_user_context(user_id):
    """API pour obtenir le contexte (pays, appareil) du dernier clic d'un utilisateur."""
    try:
        user_context = logic.obtenir_contexte_utilisateur(user_id)
        if user_context:
            return jsonify(user_context)
        return jsonify({"message": "Contexte utilisateur non trouvé"}), 404
    except Exception as e:
        app.logger.error(f"Erreur API GET /api/user_context/{user_id}: {e}")
        return jsonify({"error": "Impossible de charger le contexte utilisateur"}), 500


@app.route("/api/retraining_status", methods=["GET"])
def get_retraining_status():
    """
    API pour obtenir le statut actuel du processus de réentraînement.
    Lit le fichier de statut depuis Azure Blob Storage.
    """
    try:
        # Cette fonction devrait lire le blob 'status/retraining_status.json'
        # Pour simplifier, nous supposons qu'une fonction dans 'logic' le fait. # type: ignore
        # status_data = logic.obtenir_statut_retraining(app.config["STORAGE_CONNECTION_STRING"])
        # En attendant, voici une réponse simulée :
        status_data = {"status": "idle", "last_update": "2025-12-10T10:00:00Z"}
        return jsonify(status_data)
    except Exception as e:
        app.logger.error(f"Erreur API GET /api/retraining_status: {e}")
        return jsonify({"status": "unknown", "error": "Impossible de récupérer le statut"}), 503


# Permet de lancer l'application en mode débogage
if __name__ == "__main__":
    # En production, utilisez un serveur WSGI comme Gunicorn
    app.run(debug=True)
