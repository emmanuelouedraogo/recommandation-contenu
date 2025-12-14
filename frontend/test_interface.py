import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError

# Importe les fonctions à tester depuis votre script d'interface
from frontend.interface import charger_df_depuis_blob, sauvegarder_df_vers_blob


@pytest.fixture
def sample_df():
    """Fixture pour créer un DataFrame d'exemple pour les tests."""
    return pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})


# =========================================
# == Tests pour charger_df_depuis_blob ==
# =========================================


@patch("frontend.interface.blob_service_client")
def test_charger_df_depuis_blob_succes(mock_blob_service_client, sample_df):
    """
    Teste le cas où le chargement du DataFrame depuis le blob réussit.
    """
    # --- Préparation (Arrange) ---
    # Simule la lecture d'un fichier CSV
    csv_output = sample_df.to_csv(index=False)
    mock_downloader = MagicMock()
    mock_downloader.readall.return_value = csv_output.encode("utf-8")

    # Simule le client blob et sa méthode download_blob
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.return_value = mock_downloader
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client

    # --- Action (Act) ---
    result_df = charger_df_depuis_blob("dummy_blob.csv")

    # --- Vérification (Assert) ---
    # Vérifie que le client a été appelé correctement
    mock_blob_service_client.get_blob_client.assert_called_once_with(container="reco-data", blob="dummy_blob.csv")
    # Vérifie que le DataFrame retourné est identique à celui d'origine
    assert_frame_equal(result_df, sample_df)


@patch("frontend.interface.blob_service_client")
def test_charger_df_depuis_blob_non_trouve(mock_blob_service_client):
    """
    Teste le cas où le blob n'est pas trouvé (ResourceNotFoundError).
    La fonction doit retourner un DataFrame vide.
    """
    # --- Préparation (Arrange) ---
    # Simule le client qui lève une exception quand on essaie de télécharger
    mock_blob_client = MagicMock()
    mock_blob_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client

    # --- Action (Act) ---
    result_df = charger_df_depuis_blob("non_existent_blob.csv")

    # --- Vérification (Assert) ---
    assert result_df.empty


# ===========================================
# == Tests pour sauvegarder_df_vers_blob ==
# ===========================================


@patch("frontend.interface.blob_service_client")
def test_sauvegarder_df_vers_blob_succes(mock_blob_service_client, sample_df):
    """
    Teste le cas où la sauvegarde du DataFrame réussit.
    """
    # --- Préparation (Arrange) ---
    mock_blob_client = MagicMock()
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client

    # --- Action (Act) ---
    success = sauvegarder_df_vers_blob(sample_df, "test_save.csv")

    # --- Vérification (Assert) ---
    # Vérifie que la méthode d'upload a été appelée une fois
    mock_blob_client.upload_blob.assert_called_once()
    # Vérifie que la fonction retourne True en cas de succès
    assert success is True

    # Vérification avancée : s'assurer que le contenu envoyé est correct
    sent_data = mock_blob_client.upload_blob.call_args[0][0]
    expected_data = sample_df.to_csv(index=False)
    assert sent_data == expected_data
