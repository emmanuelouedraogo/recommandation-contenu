import streamlit as st
import pandas as pd
from services import charger_df_depuis_blob, sauvegarder_df_vers_blob, USERS_BLOB_NAME


def afficher_page_creation_compte():
    """Affiche la page de création de compte."""
    st.header("Créez votre compte")

    if st.button("Créer un nouvel identifiant"):
        current_users_df = charger_df_depuis_blob(USERS_BLOB_NAME)

        # Génère un nouvel ID unique (plus robuste qu'un simple incrément)
        if current_users_df.empty:
            new_user_id = 1
        else:
            # Assure que 'user_id' est numérique pour max()
            current_users_df["user_id"] = pd.to_numeric(current_users_df["user_id"], errors="coerce")
            new_user_id = int(current_users_df["user_id"].max() if not current_users_df.empty else 0) + 1
            # Boucle pour assurer l'unicité si des IDs ont été supprimés ou sont non numériques
            while new_user_id in current_users_df["user_id"].values:
                new_user_id += 1

        # Ajoute au DataFrame et sauvegarde
        new_user_df = pd.DataFrame([{"user_id": new_user_id}])
        updated_users_df = pd.concat([current_users_df, new_user_df], ignore_index=True)

        if sauvegarder_df_vers_blob(updated_users_df, USERS_BLOB_NAME):
            st.cache_data.clear()  # Invalide le cache pour que la liste des utilisateurs soit mise à jour
            st.success(f"Votre nouveau compte a été créé avec succès ! Votre identifiant est :")
            st.code(new_user_id, language="text")
            st.info("Vous pouvez maintenant utiliser cet identifiant dans la section 'Recommandations'.")


afficher_page_creation_compte()
