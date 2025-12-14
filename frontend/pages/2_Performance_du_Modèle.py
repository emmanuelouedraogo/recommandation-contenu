import streamlit as st
import pandas as pd
from services import charger_df_depuis_blob, TRAINING_LOG_BLOB_NAME


def afficher_page_performance():
    """Affiche la page de performance du modèle."""
    st.header("Historique et Performance des Entraînements")

    log_df = charger_df_depuis_blob(TRAINING_LOG_BLOB_NAME)

    if log_df.empty:
        st.info("Aucun historique d'entraînement n'a encore été enregistré.")
    else:
        # Assurez-vous que la colonne 'timestamp' existe et est au bon format
        if "timestamp" in log_df.columns:
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df = log_df.sort_values("timestamp").reset_index(drop=True)

            st.subheader("Évolution de la Précision@10")
            st.line_chart(log_df, x="timestamp", y="precision_at_10")

            st.subheader("Précision@10 en fonction du nombre d'interactions")
            st.line_chart(log_df, x="click_count", y="precision_at_10")
        else:
            st.warning(
                "La colonne 'timestamp' est manquante dans le log d'entraînement. Impossible d'afficher les graphiques."
            )

        st.subheader("Détail des entraînements")
        st.dataframe(log_df, use_container_width=True)


afficher_page_performance()
