"""Unit tests for scoring_bank.models.scorer."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scoring_bank.models.scorer import (
    load_model,
    predict_default_proba,
    predict_default_proba_batch,
)


class TestLoadModel:
    def test_loads_from_valid_path(self, temp_model_pkl: Path) -> None:
        model = load_model(temp_model_pkl)
        assert model is not None

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")


class TestPredictDefaultProba:
    def test_returns_float(self, mock_lgbm: MagicMock, client_df: pd.DataFrame) -> None:
        feature_cols = ["feature_a", "feature_b"]
        client = client_df.copy()
        client["feature_a"] = 0.5
        client["feature_b"] = 1.2
        result = predict_default_proba(mock_lgbm, client, feature_cols)
        assert isinstance(result, float)

    def test_result_in_unit_interval(self, mock_lgbm: MagicMock, client_df: pd.DataFrame) -> None:
        feature_cols = ["feature_a", "feature_b"]
        result = predict_default_proba(mock_lgbm, client_df, feature_cols)
        assert 0.0 <= result <= 1.0

    def test_raises_for_multiple_rows(self, mock_lgbm: MagicMock, client_df: pd.DataFrame) -> None:
        two_rows = pd.concat([client_df, client_df], ignore_index=True)
        with pytest.raises(ValueError, match="Expected exactly one client row"):
            predict_default_proba(mock_lgbm, two_rows, ["feature_a"])


class TestPredictDefaultProbaBatch:
    def test_returns_series(self, client_df: pd.DataFrame) -> None:
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.6, 0.4]])
        two_rows = pd.concat([client_df, client_df], ignore_index=True)
        result = predict_default_proba_batch(model, two_rows, ["feature_a", "feature_b"])
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_values_in_unit_interval(self, client_df: pd.DataFrame) -> None:
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        result = predict_default_proba_batch(model, client_df, ["feature_a"])
        assert all(0.0 <= v <= 1.0 for v in result)
