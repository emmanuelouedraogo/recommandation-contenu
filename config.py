import os

try:
    from dotenv import load_dotenv

    # Charger les variables d'environnement depuis un fichier .env, s'il existe.
    # C'est utile pour le développement local.
    load_dotenv()
except ImportError:
    # Le module dotenv n'est pas nécessaire en production.
    pass


class Config:
    """
    Configuration de base pour l'application Flask.
    Les configurations spécifiques à un environnement peuvent hériter de cette classe.
    """

    # Clé secrète pour la protection CSRF et la gestion des sessions.
    # DOIT être définie dans les variables d'environnement en production.
    SECRET_KEY = os.getenv("SECRET_KEY", "une-cle-secrete-par-defaut-pour-le-dev")

    # Identifiants pour la page d'administration
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
