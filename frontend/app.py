# app.py
import os
from flask import Flask, render_template, jsonify, request
import logging

# Importer les fonctions logiques depuis le nouveau fichier
import logic as logic

# Crée une instance de l'application Flask
app = Flask(__name__)

# --- Configuration ---
app.config['API_URL'] = os.environ.get('API_URL')
app.config['STORAGE_CONNECTION_STRING'] = os.environ.get('STORAGE_CONNECTION_STRING')

logging.basicConfig(level=logging.INFO)

# --- Routes pour servir les pages HTML ---

@app.route('/')
def index():
    """Sert la page d'accueil principale."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Point de terminaison pour le bilan de santé (Health Check)."""
    return "OK", 200

# --- Routes API pour le Frontend ---

@app.route('/api/users', methods=['GET'])
def get_users():
    """API pour obtenir la liste des utilisateurs."""
    try:
        users = logic.obtenir_utilisateurs(app.config['STORAGE_CONNECTION_STRING'])
        return jsonify(users)
    except Exception as e:
        app.logger.error(f"Erreur API /api/users: {e}")
        return jsonify({"error": "Impossible de charger les utilisateurs"}), 500

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """API pour obtenir les recommandations pour un utilisateur."""
    try:
        recos = logic.obtenir_recommandations_pour_utilisateur(
            app.config['API_URL'],
            user_id,
            app.config['STORAGE_CONNECTION_STRING']
        )
        if "error" in recos:
            return jsonify(recos), 404
        return jsonify(recos)
    except Exception as e:
        app.logger.error(f"Erreur API /api/recommendations/{user_id}: {e}")
        return jsonify({"error": "Erreur interne du serveur"}), 500

# Ajoutez ici d'autres routes API si nécessaire (ex: pour l'historique, la notation, etc.)

# Permet de lancer l'application en mode débogage
if __name__ == '__main__':
    # En production, utilisez un serveur WSGI comme Gunicorn
    app.run(debug=True)
