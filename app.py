import os
from flask import Flask, render_template, jsonify, request, Response
from functools import wraps

# --- Configuration and Initialization ---
# We wrap the import in a try-except block to provide a user-friendly error
# if the essential configuration is missing.
try:
    import logic
except ValueError as e:
    # This will stop the app from starting and print a clear error message.
    raise SystemExit(f"FATAL: Configuration error - {e}") from e

# Determine the absolute path for the project directory.
project_dir = os.path.dirname(os.path.abspath(__file__))

# Initialisation de l'application Flask
app = Flask(
    __name__,
    template_folder=os.path.join(project_dir, "templates"),
    static_folder=os.path.join(project_dir, "static"),
)

# --- Authentification pour la page Admin ---
# Dans une application réelle, chargez-les depuis des variables d'environnement sécurisées.
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"


def check_auth(username, password):
    """Vérifie si un couple nom d'utilisateur/mot de passe est valide."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD


def authenticate():
    """Envoie une réponse 401 pour demander une authentification."""
    return Response(
        "Accès non autorisé. Vous devez vous connecter avec des identifiants valides.",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def register_routes(app):
    """Registers all the API and frontend routes for the Flask app."""

    # --- Routes pour servir l'interface utilisateur (Frontend) ---

    @app.route("/")
    def index():
        """Sert la page d'accueil principale."""
        return render_template("index.html")

    @app.route("/admin")
    @requires_auth
    def admin_page():
        """Sert la page d'administration des utilisateurs."""
        return render_template("admin.html")

    # --- Routes de l'API pour connecter le Frontend au Backend ---

    @app.route("/api/users", methods=["GET"])
    def get_users():
        """Retourne la liste de tous les utilisateurs."""
        users = logic.obtenir_utilisateurs()
        return jsonify(users)

    @app.route("/api/admin/users", methods=["GET"])
    @requires_auth
    def get_all_users_with_status():
        """Retourne la liste de tous les utilisateurs avec leur statut pour l'admin."""
        all_users = logic.obtenir_tous_les_utilisateurs_avec_statut()
        return jsonify(all_users)

    @app.route("/api/users", methods=["POST"])  # Corrigé pour correspondre au JS
    def create_user():
        """Crée un nouvel utilisateur."""
        new_user_id = logic.creer_nouvel_utilisateur()
        return jsonify({"user_id": new_user_id, "message": f"Utilisateur {new_user_id} créé."})

    @app.route("/api/users/<int:user_id>", methods=["DELETE"])
    @requires_auth
    def delete_user(user_id):
        """Supprime un utilisateur et ses données."""
        success = logic.supprimer_utilisateur(user_id)
        if success:
            return jsonify({"message": f"Utilisateur {user_id} et ses données ont été supprimés."}), 200
        else:
            return jsonify({"error": f"Utilisateur {user_id} non trouvé."}), 404

    @app.route("/api/users/<int:user_id>/reactivate", methods=["POST"])
    @requires_auth
    def reactivate_user(user_id):
        """Réactive un utilisateur qui a été désactivé (soft-deleted)."""
        success = logic.reactiver_utilisateur(user_id)
        if success:
            return jsonify({"message": f"L'utilisateur {user_id} a été réactivé."}), 200
        else:
            error_msg = f"Impossible de réactiver l'utilisateur {user_id}. Il n'a pas été trouvé ou était déjà actif."
            return jsonify({"error": error_msg}), 404

    @app.route("/api/recommendations/<int:user_id>", methods=["GET"])
    def get_recommendations(user_id):
        """Obtient les recommandations pour un utilisateur."""
        country = request.args.get("country")
        device = request.args.get("device")
        try:
            recos = logic.obtenir_recommandations_pour_utilisateur(user_id, country, device)
            return jsonify(recos)
        except Exception as e:
            # Log the underlying error for debugging
            app.logger.error(f"Failed to get recommendations for user {user_id}: {e}")
            return jsonify({"error": "Le service de recommandation est actuellement indisponible."}), 503

    @app.route("/api/history/<int:user_id>", methods=["GET"])
    def get_history(user_id):
        """Obtient l'historique de consultation d'un utilisateur."""
        history = logic.obtenir_historique_utilisateur(user_id)
        return jsonify(history)

    @app.route("/api/interactions", methods=["POST"])  # Corrigé pour correspondre au JS
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

    @app.route("/api/global_trends", methods=["GET"])  # Corrigé pour correspondre au JS
    def get_trends():
        """Obtient les tendances globales."""
        trends = logic.obtenir_tendances_globales_clics()
        return jsonify(trends)

    @app.route("/api/performance", methods=["GET"])  # Corrigé pour correspondre au JS
    def get_model_performance():
        """Obtient les métriques de performance du modèle."""
        performance = logic.obtenir_performance_modele()
        return jsonify(performance)

    @app.route("/api/user_context/<int:user_id>", methods=["GET"])  # Ajouté
    def get_user_context(user_id):
        """Obtient le contexte (pays, appareil) d'un utilisateur."""
        context = logic.obtenir_contexte_utilisateur(user_id)
        if context:
            return jsonify(context)
        return jsonify({"error": "Contexte non trouvé pour cet utilisateur."}), 404

    @app.route("/api/articles", methods=["POST"])  # Ajouté
    def add_article():
        """Crée un nouvel article."""
        data = request.json
        title = data.get("title")
        content = data.get("content")
        category_id = data.get("category_id")

        if not all([title, content, category_id is not None]):
            return jsonify({"error": "Données manquantes : title, content et category_id sont requis."}), 400

        new_article_id = logic.creer_nouvel_article(title, content, int(category_id))
        return jsonify({"article_id": new_article_id, "message": f"Article {new_article_id} créé."}), 201

    @app.route("/api/retraining_status", methods=["GET"])  # Ajouté
    def get_retraining_status():
        """Obtient le statut du processus de ré-entraînement."""
        status = logic.obtenir_statut_reentrainement()
        return jsonify(status)


# Register all routes with the app instance
register_routes(app)
if __name__ == "__main__":
    # Permet de lancer l'application en local pour le développement
    # Le port est configurable via la variable d'environnement PORT, avec 5000 par défaut.
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
