import os
from secrets import compare_digest
import logging
from flask import Flask, render_template, jsonify, request, Response
from functools import wraps
from flask_wtf.csrf import CSRFProtect

# --- Configuration and Initialization ---
import config  # Importer le nouveau fichier de configuration

project_dir = os.path.dirname(os.path.abspath(__file__))  # NOSONAR

# Initialisation de l'application Flask
app = Flask(
    __name__,
    template_folder=os.path.join(project_dir, "templates"),
    static_folder=os.path.join(project_dir, "static"),
)
# Charger la configuration depuis l'objet Config
app.config.from_object(config.Config)

# Initialiser la protection CSRF
csrf = CSRFProtect(app)

# Configure logging
# Configuration de la journalisation plus robuste
if not app.debug:  # Appliquer une configuration plus avancée en production
    # Créer un gestionnaire de fichiers pour les logs
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.WARNING)  # Log les avertissements et erreurs dans un fichier
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"))
    app.logger.addHandler(file_handler)

logging.basicConfig(level=logging.INFO)  # Conserver pour la console en développement

try:
    import logic  # noqa: F401
except ValueError as e:
    # Log the fatal error before exiting. This makes startup issues much easier to debug.
    app.logger.critical(f"CRITICAL STARTUP FAILURE: {e}")
    raise SystemExit(f"FATAL: Configuration error - {e}") from e

# --- Logique d'authentification (pourrait être dans son propre module auth.py) ---


def check_auth(username, password):
    """Vérifie si un couple nom d'utilisateur/mot de passe est valide."""
    # Utilise compare_digest pour une comparaison sécurisée et résistante aux attaques temporelles.
    is_user_valid = compare_digest(username, app.config["ADMIN_USERNAME"])
    is_pass_valid = compare_digest(password, app.config["ADMIN_PASSWORD"])
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


# --- Gestionnaires d'erreurs pour les pages HTML ---
@app.errorhandler(404)
def not_found_error(error):
    """Affiche une page 404 personnalisée."""
    return render_template("error.html", error_code=404, error_message="Page non trouvée!"), 404


@app.errorhandler(500)
def internal_error(error):
    """Affiche une page 500 personnalisée."""
    app.logger.error(f"Server Error: {error}", exc_info=True)
    return render_template("error.html", error_code=500, error_message="Erreur interne du serveur"), 500


# --- Route pour le bilan de santé (Health Check) ---
@app.route("/health")
def health_check():
    """Point de terminaison simple pour le bilan de santé d'Azure App Service."""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":

    @app.errorhandler(Exception)
    def handle_api_error(e):
        """
        Capture toutes les exceptions non gérées pour les routes de l'API
        et retourne une réponse JSON standard.
        """
        # Journaliser l'erreur complète pour le débogage
        app.logger.error(f"Unhandled exception occurred: {e}", exc_info=True)
        # Retourner une réponse générique à l'utilisateur
        return jsonify({"error": "Une erreur interne inattendue est survenue sur le serveur."}), 500

    # Enregistrer le Blueprint auprès de l'application Flask.
    # app.register_blueprint(views_bp)
    # app.register_blueprint(api_bp, url_prefix="/api")  # Toutes les routes de l'API auront le préfixe /api
    # Permet de lancer l'application en local pour le développement
    # Le port est configurable via la variable d'environnement PORT, avec 5000 par défaut.
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, port=port)
