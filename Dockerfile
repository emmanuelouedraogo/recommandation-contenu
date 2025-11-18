# Étape 1: Utiliser une image Python officielle et légère comme image de base
FROM python:3.10-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
# Toutes les commandes suivantes s'exécuteront à partir de ce dossier
WORKDIR /app

# Étape 3: Copier le fichier des dépendances
COPY requirements.txt .

# Étape 4: Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home appuser
USER appuser

# Étape 5: Copier les fichiers et dossiers de l'application dans le conteneur
# On copie le code de l'API, les classes de modèles, et les données/modèles nécessaires
COPY ./api ./api
COPY ./models.py .

# Étape 6: Installer les dépendances
# L'option --no-cache-dir réduit la taille de l'image finale
# Installer les dépendances après avoir copié le code tire parti du cache Docker
RUN pip install --no-cache-dir --user -r requirements.txt

# Étape 7: Exposer le port sur lequel l'API s'exécutera (standard pour FastAPI)
EXPOSE 8000

# Variable d'environnement pour la connexion à Azure (sera passée au `docker run`)
ENV AZURE_CONNECTION_STRING=""

# Étape 8: Définir la commande pour lancer l'application lorsque le conteneur démarre
CMD ["/home/appuser/.local/bin/uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]