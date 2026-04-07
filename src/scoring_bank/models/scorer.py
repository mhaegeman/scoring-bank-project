"""LightGBM credit scoring helpers: model loading and probability prediction."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from scoring_bank import config

logger = logging.getLogger(__name__)


def load_model(path: Path = config.LGBM_MODEL_PATH) -> LGBMClassifier:
    """Load a serialised LightGBM classifier from disk.

    Args:
        path: Path to the pickle file. Defaults to config.LGBM_MODEL_PATH.

    Returns:
        Deserialised LGBMClassifier instance.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded LightGBM model from %s", path)
    return model


def predict_default_proba(
    model: LGBMClassifier,
    df_client: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """Return the probability of payment default for a single client row.

    Args:
        model: Trained LGBMClassifier.
        df_client: Single-row DataFrame containing the client's features.
        feature_cols: Ordered list of feature column names expected by the model.

    Returns:
        Default probability as a float in [0, 1].

    Raises:
        ValueError: If df_client does not contain exactly one row.
    """
    if len(df_client) != 1:
        raise ValueError(f"Expected exactly one client row, got {len(df_client)}.")

    proba = model.predict_proba(df_client[feature_cols])[0, 1]
    logger.debug("Default probability: %.4f", proba)
    return float(proba)


def predict_default_proba_batch(
    model: LGBMClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Return default probabilities for all rows in a DataFrame.

    Args:
        model: Trained LGBMClassifier.
        df: DataFrame containing the features.
        feature_cols: Ordered list of feature column names expected by the model.

    Returns:
        Series of default probabilities indexed the same as df.
    """
    probas = model.predict_proba(df[feature_cols])[:, 1]
    return pd.Series(probas, index=df.index, name="default_proba")
