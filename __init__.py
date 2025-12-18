from flask import Blueprint

# Crée une instance de Blueprint pour les vues web
views_bp = Blueprint("views", __name__, template_folder="templates")

from . import routes  # Importe les routes
