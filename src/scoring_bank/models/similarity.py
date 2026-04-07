"""Nearest-neighbour client similarity: model loading and lookup."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scoring_bank import config

logger = logging.getLogger(__name__)


def load_nn_model(path: Path = config.NN_MODEL_PATH) -> NearestNeighbors:
    """Load a serialised NearestNeighbors model from disk.

    Args:
        path: Path to the pickle file. Defaults to config.NN_MODEL_PATH.

    Returns:
        Deserialised NearestNeighbors instance.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"NearestNeighbors model not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded NearestNeighbors model from %s", path)
    return model


def load_scaler(path: Path = config.SCALER_PATH) -> StandardScaler:
    """Load a serialised StandardScaler from disk.

    Args:
        path: Path to the pickle file. Defaults to config.SCALER_PATH.

    Returns:
        Deserialised StandardScaler instance.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded StandardScaler from %s", path)
    return scaler


def find_similar_clients(
    nn: NearestNeighbors,
    scaler: StandardScaler,
    df_client: pd.DataFrame,
    df_train: pd.DataFrame,
    feature_cols: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Find the n most similar clients in the training set for a given client.

    The client features are scaled with the pre-fitted scaler before querying
    the nearest-neighbour model.

    Args:
        nn: Fitted NearestNeighbors model.
        scaler: Fitted StandardScaler (trained on the same feature set).
        df_client: Single-row DataFrame for the query client.
        df_train: Full training DataFrame to retrieve neighbour rows from.
        feature_cols: Feature columns to use for the distance computation.
        n_neighbors: Number of similar clients to return.

    Returns:
        DataFrame with n_neighbors rows, each corresponding to a similar client
        from df_train (in order of ascending distance).

    Raises:
        ValueError: If df_client does not contain exactly one row.
    """
    if len(df_client) != 1:
        raise ValueError(f"Expected exactly one client row, got {len(df_client)}.")

    client_scaled = scaler.transform(df_client[feature_cols])
    distances, indices = nn.kneighbors(client_scaled, n_neighbors=n_neighbors)
    neighbour_indices = indices[0]

    similar = df_train.iloc[neighbour_indices].copy()
    similar["_distance"] = distances[0]
    logger.debug("Found %d similar clients (distances: %s)", n_neighbors, distances[0])
    return similar
