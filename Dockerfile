# Étape 1: Utiliser une image Python officielle et légère comme image de base
FROM python:3.10-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
# Toutes les commandes suivantes s'exécuteront à partir de ce dossier
WORKDIR /app

# Étape 3: Copier le fichier des dépendances
COPY requirements.txt .

# Étape 6: Installer les dépendances
# L'option --no-cache-dir réduit la taille de l'image finale
# Installer les dépendances après avoir copié le code tire parti du cache Docker
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape pour télécharger le modèle depuis Azure Blob Storage ---
# On installe la CLI Azure, on l'utilise pour télécharger le modèle, puis on la désinstalle.
# Cela permet de garder l'image finale légère.
# Le --account-name et --container-name doivent correspondre à votre configuration.
RUN apt-get update && apt-get install -y curl && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    az storage blob download \
      --account-name recoappstorage123 \
      --container-name reco-data \
      --name models/hybrid_recommender_pipeline.pkl \
      --file save/hybrid_recommender_pipeline.pkl \
      --auth-mode login --no-progress && \
    apt-get purge -y curl && apt-get autoremove -y && apt-get clean

# Étape 4: Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home appuser

# Étape 5: Copier les fichiers et dossiers de l'application dans le conteneur
# On copie le code de l'API, les classes de modèles, et les données/modèles nécessaires
COPY --chown=appuser:appuser ./api ./api
COPY --chown=appuser:appuser ./models.py .
COPY --chown=appuser:appuser ./save ./save

# Basculer vers l'utilisateur non-root
USER appuser

# Étape 7: Exposer le port sur lequel l'API s'exécutera (standard pour FastAPI)
EXPOSE 8000

# Variable d'environnement pour la connexion à Azure (sera passée au `docker run`)
ENV AZURE_CONNECTION_STRING=""

# Étape 8: Définir la commande pour lancer l'application lorsque le conteneur démarre
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]