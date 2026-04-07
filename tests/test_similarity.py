"""Unit tests for scoring_bank.models.similarity."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scoring_bank.models.similarity import find_similar_clients, load_nn_model, load_scaler


def _make_pkl(tmp_path: Path, obj, name: str) -> Path:
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


@pytest.fixture()
def tiny_nn(tmp_path: Path) -> Path:
    X = np.random.default_rng(0).random((10, 3))
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    return _make_pkl(tmp_path, nn, "nn.pkl")


@pytest.fixture()
def tiny_scaler(tmp_path: Path) -> Path:
    X = np.random.default_rng(0).random((10, 3))
    scaler = StandardScaler().fit(X)
    return _make_pkl(tmp_path, scaler, "scaler.pkl")


class TestLoadNnModel:
    def test_loads_valid_file(self, tiny_nn: Path) -> None:
        model = load_nn_model(tiny_nn)
        assert isinstance(model, NearestNeighbors)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_nn_model(tmp_path / "missing.pkl")


class TestLoadScaler:
    def test_loads_valid_file(self, tiny_scaler: Path) -> None:
        scaler = load_scaler(tiny_scaler)
        assert isinstance(scaler, StandardScaler)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_scaler(tmp_path / "missing.pkl")


class TestFindSimilarClients:
    def test_returns_dataframe(self, mock_nn: MagicMock, mock_scaler: MagicMock) -> None:
        df_train = pd.DataFrame({"a": range(10), "b": range(10)})
        df_client = pd.DataFrame({"a": [5], "b": [5]})
        result = find_similar_clients(mock_nn, mock_scaler, df_client, df_train, ["a", "b"])
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number_of_rows(
        self, mock_nn: MagicMock, mock_scaler: MagicMock
    ) -> None:
        df_train = pd.DataFrame({"a": range(10), "b": range(10)})
        df_client = pd.DataFrame({"a": [5], "b": [5]})
        result = find_similar_clients(mock_nn, mock_scaler, df_client, df_train, ["a", "b"])
        assert len(result) == 5  # mock_nn returns 5 neighbours

    def test_adds_distance_column(self, mock_nn: MagicMock, mock_scaler: MagicMock) -> None:
        df_train = pd.DataFrame({"a": range(10), "b": range(10)})
        df_client = pd.DataFrame({"a": [5], "b": [5]})
        result = find_similar_clients(mock_nn, mock_scaler, df_client, df_train, ["a", "b"])
        assert "_distance" in result.columns

    def test_raises_for_multiple_rows(self, mock_nn: MagicMock, mock_scaler: MagicMock) -> None:
        df_train = pd.DataFrame({"a": range(10)})
        df_client = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Expected exactly one client row"):
            find_similar_clients(mock_nn, mock_scaler, df_client, df_train, ["a"])
