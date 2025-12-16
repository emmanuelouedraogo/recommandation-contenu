import os
from secrets import compare_digest
import logging
from flask import Flask, Blueprint, render_template, jsonify, request, Response
from functools import wraps

# --- Configuration and Initialization ---
project_dir = os.path.dirname(os.path.abspath(__file__))  # NOSONAR

# Initialisation de l'application Flask
app = Flask(
    __name__,
    template_folder=os.path.join(project_dir, "templates"),
    static_folder=os.path.join(project_dir, "static"),
)

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    import logic
except ValueError as e:
    # Log the fatal error before exiting. This makes startup issues much easier to debug.
    app.logger.critical(f"CRITICAL STARTUP FAILURE: {e}")
    raise SystemExit(f"FATAL: Configuration error - {e}") from e

# --- Authentification pour la page Admin ---
# Charger les identifiants depuis les variables d'environnement pour la sécurité.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")


def check_auth(username, password):
    """Vérifie si un couple nom d'utilisateur/mot de passe est valide."""
    # Utilise compare_digest pour une comparaison sécurisée et résistante aux attaques temporelles.
    is_user_valid = compare_digest(username, ADMIN_USERNAME)
    is_pass_valid = compare_digest(password, ADMIN_PASSWORD)
    return is_user_valid and is_pass_valid


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
            # Journaliser la tentative de connexion échouée pour la surveillance de la sécurité.
            # On inclut l'adresse IP de la source pour le contexte.
            ip_address = request.remote_addr
            username_attempt = auth.username if auth else "None"
            app.logger.warning(f"Failed admin login attempt from IP: {ip_address} with username: '{username_attempt}'")
            return authenticate()
        return f(*args, **kwargs)

    return decorated


# --- Gestionnaire d'erreurs global ---
@app.errorhandler(Exception)
def handle_generic_error(e):
    """
    Capture toutes les exceptions non gérées et retourne une réponse JSON standard.
    Cela évite de répéter les blocs try/except dans chaque route.
    """
    # Journaliser l'erreur complète pour le débogage
    app.logger.error(f"Unhandled exception occurred: {e}", exc_info=True)
    # Retourner une réponse générique à l'utilisateur
    return jsonify({"error": "Une erreur interne inattendue est survenue sur le serveur."}), 500


# Création d'un Blueprint pour les routes principales de l'API et de l'interface.
main_bp = Blueprint("main", __name__)


# --- Routes pour servir l'interface utilisateur (Frontend) ---
@main_bp.route("/health")
def health_check():
    """Endpoint de vérification de l'état pour les sondes de santé."""
    return "OK", 200


@main_bp.route("/")
def index():
    """Sert la page d'accueil principale."""
    return render_template("index.html")


@main_bp.route("/admin")
@requires_auth
def admin_page():
    """Sert la page d'administration des utilisateurs."""
    return render_template("admin.html")


# --- Routes de l'API pour connecter le Frontend au Backend ---
@main_bp.route("/api/users", methods=["GET"])
def get_users():
    """Retourne la liste de tous les utilisateurs."""
    users = logic.obtenir_utilisateurs()
    return jsonify(users)


@main_bp.route("/api/admin/users", methods=["GET"])
@requires_auth
def get_all_users_with_status():
    """Retourne la liste de tous les utilisateurs avec leur statut pour l'admin."""
    all_users = logic.obtenir_tous_les_utilisateurs_avec_statut()
    app.logger.info(f"API /api/admin/users returning {len(all_users)} users.")
    return jsonify(all_users)


@main_bp.route("/api/users", methods=["POST"])
def create_user():
    """Crée un nouvel utilisateur."""
    new_user_id = logic.creer_nouvel_utilisateur()
    return jsonify({"user_id": new_user_id, "message": f"Utilisateur {new_user_id} créé."})


@main_bp.route("/api/users/<int:user_id>", methods=["DELETE"])
@requires_auth
def delete_user(user_id):
    """Supprime un utilisateur (soft delete)."""
    success = logic.supprimer_utilisateur(user_id)
    if success:
        return jsonify({"message": f"Utilisateur {user_id} marqué comme supprimé."}), 200
    else:
        return jsonify({"error": f"Utilisateur {user_id} non trouvé."}), 404


@main_bp.route("/api/users/<int:user_id>/reactivate", methods=["POST"])
@requires_auth
def reactivate_user(user_id):
    """Réactive un utilisateur."""
    success = logic.reactiver_utilisateur(user_id)
    if success:
        return jsonify({"message": f"L'utilisateur {user_id} a été réactivé."}), 200
    else:
        error_msg = f"Impossible de réactiver l'utilisateur {user_id}. Il n'a pas été trouvé ou était déjà actif."
        return jsonify({"error": error_msg}), 404


@main_bp.route("/api/recommendations", methods=["GET"])
def get_recommendations():
    """Obtient les recommandations pour un utilisateur."""
    user_id = request.args.get("user_id", type=int)
    country = request.args.get("country")
    device = request.args.get("device")
    recos = logic.obtenir_recommandations_pour_utilisateur(user_id, country, device)
    return jsonify(recos)


@main_bp.route("/api/history/<int:user_id>", methods=["GET"])
def get_history(user_id):
    """Obtient l'historique de consultation d'un utilisateur."""
    history = logic.obtenir_historique_utilisateur(user_id)
    return jsonify(history)


@main_bp.route("/api/interactions", methods=["POST"])
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


@main_bp.route("/api/global_trends", methods=["GET"])
def get_trends():
    """Obtient les tendances globales."""
    trends = logic.obtenir_tendances_globales_clics()
    return jsonify(trends)


@main_bp.route("/api/performance", methods=["GET"])
def get_model_performance():
    """Obtient les métriques de performance du modèle."""
    performance = logic.obtenir_performance_modele()
    return jsonify(performance)


@main_bp.route("/api/user_context/<int:user_id>", methods=["GET"])
def get_user_context(user_id):
    """Obtient le contexte (pays, appareil) d'un utilisateur."""
    context = logic.obtenir_contexte_utilisateur(user_id)
    if context:
        return jsonify(context)
    return jsonify({"error": "Contexte non trouvé pour cet utilisateur."}), 404


@main_bp.route("/api/articles", methods=["POST"])
@requires_auth
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


@main_bp.route("/api/retraining_status", methods=["GET"])
def get_retraining_status():
    """Obtient le statut du processus de ré-entraînement."""
    status = logic.obtenir_statut_reentrainement()
    return jsonify(status)


# Enregistrer le Blueprint auprès de l'application Flask.
app.register_blueprint(main_bp)

if __name__ == "__main__":
    # Permet de lancer l'application en local pour le développement
    # Le port est configurable via la variable d'environnement PORT, avec 5000 par défaut.
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, port=port)
