# Étape 1: Image de base
# Utilise une image Python légère.
FROM python:3.10-slim-bullseye

# Étape 2: Installation des dépendances système
# Installe 'build-essential' qui contient le compilateur 'gcc', nécessaire pour des paquets comme scikit-surprise.
# --no-install-recommends réduit la taille de l'image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Étape 3: Définir le répertoire de travail
WORKDIR /app

# Étape 4: Installation des dépendances Python
# Copier uniquement requirements.txt et installer les paquets permet de tirer parti du cache Docker.
# Cette étape ne sera ré-exécutée que si le fichier requirements.txt change.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Étape 5: Copier le code de l'application
# Copie le reste du code de l'application dans le conteneur.
COPY . .

# Étape 6: Exposer le port
# Doit correspondre au port utilisé par uvicorn et configuré dans App Service (WEBSITES_PORT).
EXPOSE 8000

# Étape 7: Commande de démarrage
# Lance l'application FastAPI avec Uvicorn.
# --host 0.0.0.0 est nécessaire pour que l'application soit accessible de l'extérieur du conteneur.
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]