import os
from app import create_app

# Crée une instance de l'application en utilisant la factory
app = create_app()

if __name__ == "__main__":
    # Le port est configurable via la variable d'environnement PORT
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, port=port)
