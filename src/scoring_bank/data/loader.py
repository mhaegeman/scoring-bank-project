"""Data loading utilities for the bank scoring dashboard."""

import logging
from pathlib import Path

import pandas as pd

from scoring_bank import config

logger = logging.getLogger(__name__)


def load_api_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the main API dataset (data_api.csv).

    Drops rows with missing values in the core interpretable features used
    by the scoring model and nearest-neighbour lookup.

    Args:
        path: Path to the CSV file. Defaults to config.DATA_PATH.

    Returns:
        Cleaned DataFrame with all client records.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"API data file not found: {path}")

    df = pd.read_csv(path)
    required_cols = config.INTERPRETABLE_FEATURES + ["TARGET"]
    df.dropna(subset=required_cols, inplace=True)
    logger.info("Loaded API data: %d rows, %d columns from %s", *df.shape, path)
    return df


def load_interpretable_data(path: Path = config.INTERPRETABLE_CSV) -> pd.DataFrame:
    """Load the human-readable interpretable feature table.

    Args:
        path: Path to the CSV file. Defaults to config.INTERPRETABLE_CSV.

    Returns:
        DataFrame with client-friendly feature names and values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Interpretable data file not found: {path}")

    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    logger.info("Loaded interpretable data: %d rows from %s", len(df), path)
    return df


def load_group_data(path: Path = config.GROUP_CSV) -> pd.DataFrame:
    """Load the demographic group classification data.

    Args:
        path: Path to the CSV file. Defaults to config.GROUP_CSV.

    Returns:
        DataFrame with client demographic group assignments.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Group data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded group data: %d rows from %s", len(df), path)
    return df


def load_segment_data(segment: str) -> pd.DataFrame:
    """Load a pre-aggregated demographic segment comparison table.

    Valid segments: 'genre', 'income', 'education_type', 'organization_type', 'family'.

    Args:
        segment: Name of the demographic segment.

    Returns:
        Aggregated DataFrame for the requested segment.

    Raises:
        ValueError: If segment name is not recognised.
        FileNotFoundError: If the CSV file does not exist.
    """
    segment_paths: dict[str, Path] = {
        "genre": config.GENRE_CSV,
        "income": config.INCOME_CSV,
        "education_type": config.EDUCATION_CSV,
        "organization_type": config.ORGANIZATION_CSV,
        "family": config.FAMILY_CSV,
    }

    if segment not in segment_paths:
        raise ValueError(
            f"Unknown segment '{segment}'. Valid options: {list(segment_paths.keys())}"
        )

    path = segment_paths[segment]
    if not path.exists():
        raise FileNotFoundError(f"Segment data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded '%s' segment data: %d rows from %s", segment, len(df), path)
    return df


def load_nn_data(path: Path = config.NN_CSV) -> pd.DataFrame:
    """Load the nearest-neighbour reference dataset.

    Args:
        path: Path to the CSV file. Defaults to config.NN_CSV.

    Returns:
        DataFrame used as the reference pool for nearest-neighbour queries.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nearest-neighbour data file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded NN reference data: %d rows from %s", len(df), path)
    return df
