"""Shared pytest fixtures for the scoring_bank test suite."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small DataFrame with mixed column types for feature engineering tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "int_col": rng.integers(0, 100, size=20).astype(np.int64),
            "float_col": rng.uniform(0.0, 1.0, size=20).astype(np.float64),
            "cat_col": ["A", "B", "C", "A", "B"] * 4,
            "rare_cat": ["common"] * 18 + ["rare_x", "rare_y"],
            "all_missing": [np.nan] * 20,
        }
    )


@pytest.fixture()
def client_df() -> pd.DataFrame:
    """Single-row DataFrame representing one client."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100002],
            "PAYMENT_RATE": [0.05],
            "AMT_ANNUITY": [24000.0],
            "DAYS_BIRTH": [-14000],
            "DAYS_EMPLOYED": [-2000],
            "ANNUITY_INCOME_PERC": [0.2],
            "feature_a": [0.5],
            "feature_b": [1.2],
            "TARGET": [0],
        }
    )


@pytest.fixture()
def temp_csv(tmp_path: Path) -> Path:
    """Write a small CSV file and return its path."""
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100002, 100003],
            "PAYMENT_RATE": [0.05, 0.08],
            "AMT_ANNUITY": [24000.0, 18000.0],
            "DAYS_BIRTH": [-14000, -12000],
            "DAYS_EMPLOYED": [-2000, -3000],
            "ANNUITY_INCOME_PERC": [0.2, 0.15],
            "TARGET": [0, 1],
        }
    )
    csv_path = tmp_path / "data_api.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Mock ML models
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lgbm() -> MagicMock:
    """Mock LGBMClassifier that always returns a fixed probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    model.predict.return_value = np.array([0])
    return model


@pytest.fixture()
def mock_scaler() -> MagicMock:
    """Mock StandardScaler that returns its input unchanged."""
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: np.array(x)
    return scaler


@pytest.fixture()
def mock_nn() -> MagicMock:
    """Mock NearestNeighbors returning indices [0, 1, 2, 3, 4]."""
    nn = MagicMock()
    nn.kneighbors.return_value = (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        np.array([[0, 1, 2, 3, 4]]),
    )
    return nn


@pytest.fixture()
def temp_model_pkl(tmp_path: Path) -> Path:
    """Serialise a real (tiny) sklearn model to a temp pickle and return the path."""
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="constant", constant=0)
    model.fit([[0, 0], [1, 1]], [0, 1])  # minimal fit so predict_proba works
    path = tmp_path / "LightGBMModel.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
