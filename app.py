import os
from flask import Flask, render_template, jsonify, request
import logic

# Initialisation de l'application Flask
app = Flask(__name__)

# --- Configuration ---
# L'URL de l'API de recommandation (le modèle déployé comme une fonction Azure)
API_RECO_URL = os.getenv("API_RECO_URL", "http://localhost:7071")


# --- Routes pour servir l'interface utilisateur (Frontend) ---


@app.route("/")
def index():
    """Sert la page d'accueil principale."""
    return render_template("index.html")


# --- Routes de l'API pour connecter le Frontend au Backend ---


@app.route("/api/users", methods=["GET"])
def get_users():
    """Retourne la liste de tous les utilisateurs."""
    users = logic.obtenir_utilisateurs()
    return jsonify(users)


@app.route("/api/users/new", methods=["POST"])
def create_user():
    """Crée un nouvel utilisateur."""
    new_user_id = logic.creer_nouvel_utilisateur()
    return jsonify({"user_id": new_user_id, "message": f"Utilisateur {new_user_id} créé."})


@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    """Obtient les recommandations pour un utilisateur."""
    country = request.args.get("country")
    device = request.args.get("device")
    recos = logic.obtenir_recommandations_pour_utilisateur(API_RECO_URL, user_id, country, device)
    return jsonify(recos)


@app.route("/api/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    """Obtient l'historique de consultation d'un utilisateur."""
    history = logic.obtenir_historique_utilisateur(user_id)
    return jsonify(history)


@app.route("/api/interaction", methods=["POST"])
def post_interaction():
    """Enregistre une nouvelle interaction (notation)."""
    data = request.json
    user_id = data.get("user_id")
    article_id = data.get("article_id")
    rating = data.get("rating")

    if not all([user_id, article_id, rating]):
        return jsonify({"error": "Données manquantes : user_id, article_id et rating sont requis."}), 400

    logic.ajouter_ou_mettre_a_jour_interaction(int(user_id), int(article_id), int(rating))
    return jsonify({"message": "Interaction enregistrée avec succès."}), 201


@app.route("/api/trends", methods=["GET"])
def get_trends():
    """Obtient les tendances globales."""
    trends = logic.obtenir_tendances_globales_clics()
    return jsonify(trends)


@app.route("/api/model/performance", methods=["GET"])
def get_model_performance():
    """Obtient les métriques de performance du modèle."""
    performance = logic.obtenir_performance_modele()
    return jsonify(performance)


if __name__ == "__main__":
    # Permet de lancer l'application en local pour le développement
    app.run(debug=True, port=5000)
