import streamlit as st
import pandas as pd
from services import charger_df_depuis_blob, mettre_a_jour_interaction, CLICKS_BLOB_NAME, ARTICLES_BLOB_NAME

def afficher_page_historique():
    st.header("Historique de vos notations")
    
    if st.session_state.get('user_id') is None:
        st.info("Veuillez vous connecter via la barre latérale pour voir votre historique.")
        return

    user_id = st.session_state.user_id
    clicks_df = charger_df_depuis_blob(CLICKS_BLOB_NAME)
    
    if clicks_df.empty:
        st.warning("Aucune notation n'a encore été enregistrée dans le système.")
        return

    user_history_df = clicks_df[clicks_df['user_id'] == user_id]
    
    if user_history_df.empty:
        st.info("Vous n'avez encore noté aucun article.")
        return

    # --- Validation des données ---
    required_cols = ['user_id', 'article_id', 'click_timestamp', 'nb']
    if not all(col in user_history_df.columns for col in required_cols):
        st.error("Le fichier d'historique est mal formaté. Des colonnes sont manquantes.")
        return

    user_history_df = user_history_df.sort_values('click_timestamp').drop_duplicates(subset=['user_id', 'article_id'], keep='last')
    articles_df_history = charger_df_depuis_blob(ARTICLES_BLOB_NAME)
    history_details = user_history_df.merge(articles_df_history, on='article_id', how='left').fillna({'title': 'Titre inconnu'})
    history_details = history_details.sort_values(by='click_timestamp', ascending=False)
    
    st.subheader(f"Articles que vous avez notés, Utilisateur {user_id} :")
    for _, row in history_details.iterrows():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**{row.get('title', 'Titre inconnu')}**")
            st.caption(f"Dernière modification : {pd.to_datetime(row['click_timestamp'], unit='s').strftime('%Y-%m-%d %H:%M')}")
        with col2:
            new_rating = st.number_input("Votre note", min_value=1, max_value=5, value=int(row.get('nb', 0)), key=f"update_rating_{row['article_id']}")
            if st.button("Modifier la note", key=f"update_btn_{row['article_id']}", use_container_width=True):
                mettre_a_jour_interaction(user_id, row['article_id'], new_rating)
        st.divider()

afficher_page_historique()