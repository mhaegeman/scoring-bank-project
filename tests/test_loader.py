"""Unit tests for scoring_bank.data.loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoring_bank.data.loader import (
    load_api_data,
    load_group_data,
    load_interpretable_data,
    load_nn_data,
    load_segment_data,
)


def _write_csv(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


class TestLoadApiData:
    def test_returns_dataframe(self, temp_csv: Path) -> None:
        df = load_api_data(temp_csv)
        assert isinstance(df, pd.DataFrame)

    def test_drops_rows_with_missing_key_features(self, tmp_path: Path) -> None:
        """Rows missing any INTERPRETABLE_FEATURES+TARGET should be dropped."""
        from scoring_bank import config

        cols = config.INTERPRETABLE_FEATURES + ["TARGET"]
        data = {c: [1.0, 2.0] for c in cols}
        data["PAYMENT_RATE"] = [np.nan, 0.05]  # first row will be dropped
        path = _write_csv(tmp_path, "test.csv", data)

        result = load_api_data(path)
        assert len(result) == 1

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_api_data(tmp_path / "not_here.csv")


class TestLoadInterpretableData:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "interp.csv", {"Identifiant": [1, 2], "val": [0.1, 0.2]})
        df = load_interpretable_data(path)
        assert isinstance(df, pd.DataFrame)

    def test_drops_unnamed_index_column(self, tmp_path: Path) -> None:
        df_raw = pd.DataFrame({"Identifiant": [1], "val": [0.1]})
        path = tmp_path / "interp.csv"
        df_raw.to_csv(path, index=True)  # writes "Unnamed: 0" index column
        df = load_interpretable_data(path)
        assert "Unnamed: 0" not in df.columns

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_interpretable_data(tmp_path / "missing.csv")


class TestLoadGroupData:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "group.csv", {"SK_ID_CURR": [1, 2], "CODE_GENDER": ["M", "F"]})
        df = load_group_data(path)
        assert isinstance(df, pd.DataFrame)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_group_data(tmp_path / "missing.csv")


class TestLoadNnData:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, "nn.csv", {"SK_ID_CURR": [1, 2], "val": [0.1, 0.2]})
        df = load_nn_data(path)
        assert isinstance(df, pd.DataFrame)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_nn_data(tmp_path / "missing.csv")


class TestLoadSegmentData:
    def test_raises_for_unknown_segment(self) -> None:
        with pytest.raises(ValueError, match="Unknown segment"):
            load_segment_data("invalid_segment")

    def test_raises_for_missing_file(self, monkeypatch, tmp_path: Path) -> None:
        from scoring_bank import config

        monkeypatch.setattr(config, "GENRE_CSV", tmp_path / "genre_missing.csv")
        with pytest.raises(FileNotFoundError):
            load_segment_data("genre")

    def test_loads_valid_segment(self, tmp_path: Path, monkeypatch) -> None:
        from scoring_bank import config

        path = _write_csv(tmp_path, "genre.csv", {"CODE_GENDER": ["M", "F"], "Count": [10, 20]})
        monkeypatch.setattr(config, "GENRE_CSV", path)
        df = load_segment_data("genre")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
