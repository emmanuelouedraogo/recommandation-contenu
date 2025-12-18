from flask import Blueprint

# Crée une instance de Blueprint pour l'API
api_bp = Blueprint("api", __name__)

from . import routes  # noqa: F401, E402
