import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from azure.core.exceptions import ResourceNotFoundError
from io import BytesIO

# Assurez-vous que le module 'logic' est importable.
# Vous pourriez avoir besoin de configurer votre PYTHONPATH ou d'utiliser un conftest.py
import logic

# --- Préparation des données de test ---

# Données pour simuler un fichier Parquet
sample_parquet_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
parquet_bytes = BytesIO()
sample_parquet_df.to_parquet(parquet_bytes, index=False)
parquet_bytes.seek(0)

# Données pour simuler un fichier CSV
sample_csv_df = pd.DataFrame({"article_id": [101, 102], "category_id": [1, 2]})
csv_string = sample_csv_df.to_csv(index=False)
csv_bytes = csv_string.encode("utf-8")

# Données pour tester le renommage de colonne dans le CSV
sample_csv_rename_df = pd.DataFrame({"click_article_id": [201, 202], "category_id": [3, 4]})
csv_rename_string = sample_csv_rename_df.to_csv(index=False)
csv_rename_bytes = csv_rename_string.encode("utf-8")


@patch("logic.recuperer_client_blob_service")
def test_charger_df_depuis_blob_succes_parquet(mock_get_client):
    """
    Vérifie que la fonction charge correctement un fichier Parquet quand il existe.
    """
    # Arrange : Configuration du mock
    mock_blob_service_client = MagicMock()
    mock_get_client.return_value = mock_blob_service_client

    # Simuler le client pour le fichier Parquet
    mock_parquet_client = MagicMock()
    mock_parquet_downloader = MagicMock()
    parquet_bytes.seek(0)  # <-- FIX: Reset stream before use
    mock_parquet_downloader.readall.return_value = parquet_bytes.read()
    mock_parquet_client.download_blob.return_value = mock_parquet_downloader

    # La fonction `get_blob_client` sera appelée deux fois (une pour parquet, une pour csv)
    # Nous configurons le mock pour qu'il retourne le bon client simulé.
    mock_blob_service_client.get_blob_client.side_effect = [
        mock_parquet_client,  # Premier appel (pour .parquet)
        MagicMock(),  # Deuxième appel (pour .csv), ne sera pas utilisé
    ]

    # Act : Appel de la fonction à tester
    # On s'assure de vider le cache avant le test pour éviter les interférences
    logic.charger_df_depuis_blob.cache_clear()
    result_df = logic.charger_df_depuis_blob("dummy_blob.csv")

    # Assert : Vérification des résultats
    assert not result_df.empty
    pd.testing.assert_frame_equal(result_df, sample_parquet_df)
    mock_parquet_client.download_blob.assert_called_once()


@patch("logic.recuperer_client_blob_service")
def test_charger_df_depuis_blob_fallback_csv(mock_get_client):
    """
    Vérifie que la fonction se rabat sur le CSV si le Parquet n'est pas trouvé.
    """
    # Arrange
    mock_blob_service_client = MagicMock()
    mock_get_client.return_value = mock_blob_service_client

    # Simuler le client Parquet qui lève une erreur "non trouvé"
    mock_parquet_client = MagicMock()
    mock_parquet_client.download_blob.side_effect = ResourceNotFoundError("Parquet blob not found")

    # Simuler le client CSV qui retourne des données
    mock_csv_client = MagicMock()
    mock_csv_downloader = MagicMock()
    csv_bytes.seek(0)  # <-- FIX: Reset stream before use
    mock_csv_downloader.readall.return_value = csv_bytes
    mock_csv_client.download_blob.return_value = mock_csv_downloader

    mock_blob_service_client.get_blob_client.side_effect = [mock_parquet_client, mock_csv_client]

    # Act
    logic.charger_df_depuis_blob.cache_clear()
    result_df = logic.charger_df_depuis_blob("dummy_blob.csv")

    # Assert
    # Le DataFrame résultant doit correspondre au CSV, avec la colonne 'article_id' en int
    expected_df = sample_csv_df.copy()
    expected_df["article_id"] = expected_df["article_id"].astype(int)

    pd.testing.assert_frame_equal(result_df, expected_df)
    mock_parquet_client.download_blob.assert_called_once()
    mock_csv_client.download_blob.assert_called_once()


@patch("logic.recuperer_client_blob_service")
def test_charger_df_depuis_blob_renommage_colonne_csv(mock_get_client):
    """
    Vérifie que la colonne 'click_article_id' est bien renommée en 'article_id'.
    """
    # Arrange
    mock_blob_service_client = MagicMock()
    mock_get_client.return_value = mock_blob_service_client
    mock_parquet_client = MagicMock()
    mock_parquet_client.download_blob.side_effect = ResourceNotFoundError("Not found")
    mock_csv_client = MagicMock()
    mock_csv_downloader = MagicMock()
    csv_rename_bytes.seek(0)  # <-- FIX: Reset stream before use
    mock_csv_downloader.readall.return_value = csv_rename_bytes
    mock_csv_client.download_blob.return_value = mock_csv_downloader
    mock_blob_service_client.get_blob_client.side_effect = [mock_parquet_client, mock_csv_client]

    # Act
    logic.charger_df_depuis_blob.cache_clear()
    result_df = logic.charger_df_depuis_blob("dummy_clicks.csv")

    # Assert
    assert "article_id" in result_df.columns
    assert "click_article_id" not in result_df.columns
    assert result_df["article_id"].iloc[0] == 201


@patch("logic.recuperer_client_blob_service")
def test_charger_df_depuis_blob_aucun_fichier_trouve(mock_get_client):
    """
    Vérifie que la fonction retourne un DataFrame vide si ni Parquet ni CSV ne sont trouvés.
    """
    # Arrange
    mock_blob_service_client = MagicMock()
    mock_get_client.return_value = mock_blob_service_client
    mock_blob_service_client.get_blob_client.return_value.download_blob.side_effect = ResourceNotFoundError("Not found")

    # Act
    logic.charger_df_depuis_blob.cache_clear()
    result_df = logic.charger_df_depuis_blob("non_existent_blob.csv")

    # Assert
    assert result_df.empty
