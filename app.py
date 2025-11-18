# app.py

import streamlit as st
import pandas as pd
import os
import requests

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Recommandation de Contenu",
    page_icon="📚",
    layout="wide"
)

# --- Constantes ---
USERS_FILE = 'data/users.csv'
ARTICLES_FILE = 'data/articles_metadata.csv'
# Remplacez par l'URL de votre Azure Function
AZURE_FUNCTION_URL = st.secrets.get("AZURE_FUNCTION_URL", "http://localhost:7071/api/myfunction")

# --- Fonctions de Chargement des Données ---

@st.cache_resource
def load_csv_data(file_path, columns):
    """Charge un fichier CSV ou en crée un nouveau. Mise en cache."""
    # S'assure que le dossier parent existe
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        return df
    return pd.read_csv(file_path)

def save_csv_data(df, file_path):
    """Sauvegarde un DataFrame dans un fichier CSV."""
    df.to_csv(file_path, index=False)

# --- Initialisation et Chargement ---
# Chargement des données
users_df = load_csv_data(USERS_FILE, columns=['user_id'])
articles_df = load_csv_data(ARTICLES_FILE, columns=['article_id', 'title', 'content', 'category_id', 'created_at_ts'])

# --- Fonctions du Système de Recommandation ---

def get_recommendations(user_id):
    """
    Appelle l'Azure Function pour obtenir les recommandations.
    """
    if user_id not in users_df['user_id'].unique():
        st.error("Cet identifiant utilisateur n'existe pas. Veuillez créer un compte.")
        return None
    
    try:
        params = {'user_id': user_id}
        response = requests.get(AZURE_FUNCTION_URL, params=params, timeout=20)
        response.raise_for_status() # Lève une exception si la requête échoue
        
        data = response.json()
        
        if data.get("is_cold_start"):
            st.info("Nouvel utilisateur ! Voici les 5 articles les plus populaires :")
        else:
            st.success(f"Bienvenue, Utilisateur {user_id} ! Voici vos recommandations personnalisées :")
            
        return pd.DataFrame(data.get("recommendations", []))
        
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion au service de recommandation : {e}")
        return None

# --- Interface Streamlit ---

st.title("📚 Système de Recommandation de Contenu")

# Menu dans la barre latérale
menu = ["Recommandations", "Créer un compte", "Ajouter un article"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Recommandations":
    st.header("Obtenez vos recommandations")
    
    # Affiche la liste des utilisateurs pour faciliter le test
    if not users_df.empty:
        st.info("Utilisateurs existants (pour tester) :")
        st.dataframe(users_df, use_container_width=True)

    user_id_input = st.text_input("Entrez votre identifiant utilisateur :")

    if st.button("Obtenir mes recommandations"):
        if user_id_input:
            try:
                # Tente de convertir en int si vos IDs sont numériques
                user_id = int(user_id_input)
                recommendations = get_recommendations(user_id)
                if recommendations is not None and not recommendations.empty:
                    for _, row in recommendations.iterrows():
                        with st.container():
                            st.subheader(f"{row['title']} (Article ID: {row['article_id']})")
                            st.write(row['content'][:200] + "...") # Affiche un aperçu
                            st.markdown("---")
                elif recommendations is not None:
                     st.warning("Il n'y a pas assez d'articles à recommander pour le moment.")

            except ValueError:
                st.error("L'identifiant utilisateur doit être un nombre.")
        else:
            st.warning("Veuillez entrer un identifiant utilisateur.")

elif choice == "Créer un compte":
    st.header("Créez votre compte")
    
    if st.button("Créer un nouvel identifiant"):
        # Génère un nouvel ID unique (plus robuste qu'un simple incrément)
        new_user_id = int(users_df['user_id'].max() + 1) if not users_df.empty else 1
        
        # Ajoute au DataFrame et sauvegarde
        new_user_df = pd.DataFrame([{'user_id': new_user_id}])
        users_df = pd.concat([users_df, new_user_df], ignore_index=True)
        save_csv_data(users_df, USERS_FILE)
        
        st.success(f"Votre nouveau compte a été créé avec succès ! Votre identifiant est :")
        st.code(new_user_id, language='text')
        st.info("Vous pouvez maintenant utiliser cet identifiant dans la section 'Recommandations'.")

elif choice == "Ajouter un article":
    st.header("Ajouter un nouvel article ou livre")

    with st.form(key="article_form", clear_on_submit=True):
        article_title = st.text_input("Titre de l'article/livre")
        article_category = st.number_input("ID de la catégorie", min_value=0, step=1)
        article_content = st.text_area("Contenu")
        submit_button = st.form_submit_button(label="Ajouter à la base de données")

        if submit_button and article_title and article_content:
            if article_title and article_content:
                # Génère un ID unique pour l'article
                new_article_id = int(articles_df['article_id'].max() + 1) if not articles_df.empty else 1
                
                new_article = pd.DataFrame([{
                    'article_id': new_article_id,
                    'title': article_title,
                    'content': article_content,
                    'category_id': article_category,
                    'created_at_ts': int(pd.Timestamp.now().timestamp())
                }])
                
                articles_df = pd.concat([articles_df, new_article], ignore_index=True)
                save_csv_data(articles_df, ARTICLES_FILE)
                
                st.success(f"L'article '{article_title}' a été ajouté avec succès !")
            else:
                st.warning("Veuillez remplir le titre et le contenu.")
    
    st.divider()
    st.subheader("Articles actuels dans la base de données")
    st.dataframe(articles_df, use_container_width=True)
