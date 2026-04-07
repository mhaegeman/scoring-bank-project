"""Unit tests for scoring_bank.data.loader."""

from pathlib import Path

import pandas as pd
import pytest

from scoring_bank.data.loader import load_api_data, load_segment_data


class TestLoadApiData:
    def test_returns_dataframe(self, temp_csv: Path) -> None:
        df = load_api_data(temp_csv)
        assert isinstance(df, pd.DataFrame)

    def test_drops_rows_with_missing_key_features(self, tmp_path: Path) -> None:
        """Rows missing any INTERPRETABLE_FEATURES+TARGET should be dropped."""
        import numpy as np

        from scoring_bank import config

        cols = config.INTERPRETABLE_FEATURES + ["TARGET"]
        data = {c: [1.0, 2.0] for c in cols}
        data["PAYMENT_RATE"] = [np.nan, 0.05]  # first row will be dropped
        df_raw = pd.DataFrame(data)
        path = tmp_path / "test.csv"
        df_raw.to_csv(path, index=False)

        result = load_api_data(path)
        assert len(result) == 1

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_api_data(tmp_path / "not_here.csv")


class TestLoadSegmentData:
    def test_raises_for_unknown_segment(self) -> None:
        with pytest.raises(ValueError, match="Unknown segment"):
            load_segment_data("invalid_segment")

    def test_raises_for_missing_file(self, monkeypatch, tmp_path: Path) -> None:
        from scoring_bank import config

        monkeypatch.setattr(config, "GENRE_CSV", tmp_path / "genre_missing.csv")
        with pytest.raises(FileNotFoundError):
            load_segment_data("genre")
