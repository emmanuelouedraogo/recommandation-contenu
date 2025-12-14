import streamlit as st
import pandas as pd
from services import charger_df_depuis_blob, sauvegarder_df_vers_blob, ARTICLES_BLOB_NAME


def afficher_page_ajout_article():
    """Affiche la page d'ajout d'article."""
    st.header("Ajouter un nouvel article ou livre")

    with st.form(key="article_form", clear_on_submit=True):
        article_title = st.text_input("Titre de l'article/livre")
        article_category = st.number_input("ID de la catégorie", min_value=0, step=1)
        article_content = st.text_area("Contenu")
        submit_button = st.form_submit_button(label="Ajouter à la base de données")

        if submit_button and article_title and article_content:
            current_articles_df = charger_df_depuis_blob(ARTICLES_BLOB_NAME)

            # Génère un ID unique pour l'article
            if current_articles_df.empty:
                new_article_id = 1
            else:
                # Assure que 'article_id' est numérique pour max()
                current_articles_df["article_id"] = pd.to_numeric(current_articles_df["article_id"], errors="coerce")
                new_article_id = (
                    int(current_articles_df["article_id"].max() if not current_articles_df.empty else 0) + 1
                )

            new_article = pd.DataFrame(
                [
                    {
                        "article_id": new_article_id,
                        "title": article_title,
                        "content": article_content,
                        "category_id": article_category,
                        "created_at_ts": int(pd.Timestamp.now().timestamp()),
                    }
                ]
            )

            updated_articles_df = pd.concat([current_articles_df, new_article], ignore_index=True)
            if sauvegarder_df_vers_blob(updated_articles_df, ARTICLES_BLOB_NAME):
                st.cache_data.clear()  # Invalide le cache pour que la liste des articles soit mise à jour
                st.success(f"L'article '{article_title}' a été ajouté avec succès !")

    st.divider()
    st.subheader("Articles actuels dans la base de données")
    st.dataframe(charger_df_depuis_blob(ARTICLES_BLOB_NAME), use_container_width=True)


afficher_page_ajout_article()
