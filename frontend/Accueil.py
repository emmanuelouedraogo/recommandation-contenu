import streamlit as st
import logging
from flask import Flask
from threading import Thread

# Import des services partagés
from services import ARTICLES_BLOB_NAME, obtenir_recommandations, USERS_BLOB_NAME, ARTICLES_BLOB_NAME

# --- Configuration de la Page ---
st.set_page_config(page_title="Recommandation de Contenu", page_icon="📚", layout="wide")


def afficher_page_recommandations(api_url: str):
    """Affiche la page des recommandations."""
    st.header("Obtenez vos recommandations")

    # Affiche la liste des utilisateurs pour faciliter le test
    users_df_display = charger_df_depuis_blob(USERS_BLOB_NAME)
    if users_df_display is not None and not users_df_display.empty:
        with st.expander("Voir les utilisateurs existants (pour les tests)"):
            st.dataframe(users_df_display, use_container_width=True)

    if st.session_state.user_id is None:
        st.info("Veuillez vous connecter via la barre latérale pour obtenir vos recommandations.")
    else:
        user_id = st.session_state.user_id
        recommendations = get_recommendations(api_url, user_id)

        if recommendations is not None and not recommendations.empty:
            articles_df = charger_df_depuis_blob(ARTICLES_BLOB_NAME)
            reco_details = recommendations.merge(articles_df, on="article_id", how="left")

            st.success(f"Bienvenue, Utilisateur {user_id} ! Voici vos recommandations personnalisées :")

            for _, row in reco_details.iterrows():
                with st.container(border=True):
                    st.subheader(f"{row.get('title', 'Titre inconnu')}")
                    st.caption(
                        f"Score de recommandation : {row.get('final_score', 0):.2f} | ID Article : {row['article_id']}"
                    )
                    st.write(str(row.get("content", "Contenu non disponible."))[:250] + "...")

                    rating = st.slider("Notez cet article :", 1, 5, 3, key=f"rating_{row['article_id']}")
                    if st.button("Envoyer ma note", key=f"btn_{row['article_id']}", use_container_width="always"):
                        ajouter_interaction(user_id, row["article_id"], rating)
                    st.divider()
        elif recommendations is not None:
            st.warning("Il n'y a pas assez d'articles à recommander pour le moment.")


def setup_health_check():
    """Configure et démarre le serveur Flask pour le bilan de santé dans un thread."""
    health_app = Flask(__name__)

    @health_app.route("/health")
    def health_check():
        return "OK", 200

    def run_server():
        health_app.run(host="0.0.0.0", port=8080)

    health_thread = Thread(target=run_server, daemon=True)
    health_thread.start()


def setup_login():
    """Gère la logique de connexion dans la barre latérale."""
    st.sidebar.divider()
    if st.session_state.get("user_id") is None:
        st.sidebar.header("Connexion")
        login_user_id = st.sidebar.text_input("Entrez votre identifiant utilisateur", key="login_input")
        if st.sidebar.button("Se connecter"):
            if login_user_id:
                try:
                    user_id_to_check = int(login_user_id)
                    users_df = charger_df_depuis_blob(USERS_BLOB_NAME)
                    if user_id_to_check in users_df["user_id"].unique():
                        st.session_state.user_id = user_id_to_check
                        st.rerun()
                    else:
                        st.sidebar.error("Cet utilisateur n'existe pas.")
                except ValueError:
                    st.sidebar.error("L'ID doit être un nombre.")
    else:
        st.sidebar.success(f"Connecté en tant que : **{st.session_state.user_id}**")
        if st.sidebar.button("Se déconnecter"):
            st.session_state.user_id = None
            st.rerun()


def main_app():
    """Point d'entrée principal de l'application Streamlit."""
    st.title("📚 Système de Recommandation de Contenu")

    # --- Initialisation Globale ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    setup_health_check()

    try:
        api_url = st.secrets["API_URL"]
    except KeyError as e:
        st.error(f"Erreur de configuration : Le secret '{e.args[0]}' est manquant.")
        st.info(
            "Veuillez vérifier vos variables d'environnement sur Azure App Service ou votre fichier .streamlit/secrets.toml en local."
        )
        st.stop()

    setup_login()

    # --- Affichage de la page d'accueil (Recommandations) ---
    display_recommendation_page(api_url)


if __name__ == "__main__":
    main_app()
