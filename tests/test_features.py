"""Unit tests for scoring_bank.features.engineering."""

import numpy as np
import pandas as pd

from scoring_bank.features.engineering import (
    grab_col_names,
    missing_values,
    one_hot_encoder,
    rare_encoder,
    reduce_mem_usage,
)


class TestReduceMemUsage:
    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        result = reduce_mem_usage(sample_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.drop("all_missing", axis=1).copy()
        result = reduce_mem_usage(df)
        assert result.shape == df.shape

    def test_reduces_memory(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.drop("all_missing", axis=1).copy()
        before = df.memory_usage(deep=True).sum()
        result = reduce_mem_usage(df)
        after = result.memory_usage(deep=True).sum()
        # At minimum memory should not increase significantly
        assert after <= before * 1.1  # allow 10% slack for overhead


class TestOneHotEncoder:
    def test_adds_columns(self, sample_df: pd.DataFrame) -> None:
        df = sample_df[["int_col", "cat_col"]].copy()
        encoded, new_cols = one_hot_encoder(df)
        assert len(new_cols) > 0

    def test_new_cols_are_boolean_or_numeric(self, sample_df: pd.DataFrame) -> None:
        df = sample_df[["int_col", "cat_col"]].copy()
        encoded, new_cols = one_hot_encoder(df)
        for col in new_cols:
            assert encoded[col].dtype in (np.dtype("bool"), np.dtype("uint8"), np.dtype("float64"))

    def test_original_cat_col_removed(self, sample_df: pd.DataFrame) -> None:
        df = sample_df[["int_col", "cat_col"]].copy()
        encoded, _ = one_hot_encoder(df)
        assert "cat_col" not in encoded.columns


class TestGrabColNames:
    def test_returns_four_tuples(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.drop("all_missing", axis=1).copy()
        result = grab_col_names(df)
        assert len(result) == 4

    def test_numeric_cols_not_in_cat_cols(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.drop("all_missing", axis=1).copy()
        cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
        for col in num_cols:
            assert col not in cat_cols

    def test_show_date_returns_five(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = grab_col_names(df, show_date=True)
        assert len(result) == 5


class TestMissingValues:
    def test_drops_fully_missing_column(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        assert "all_missing" in df.columns
        missing_values(df)
        assert "all_missing" not in df.columns

    def test_returns_summary_dataframe(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        result = missing_values(df)
        assert isinstance(result, pd.DataFrame)
        assert "Num_Missing" in result.columns

    def test_no_missing_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = missing_values(df)
        assert len(result) == 0


class TestRareEncoder:
    def test_rare_categories_replaced(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        result = rare_encoder(df, "rare_cat", rare_perc=0.1)
        assert "Rare" in result["rare_cat"].values

    def test_common_category_preserved(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        result = rare_encoder(df, "rare_cat", rare_perc=0.1)
        assert "common" in result["rare_cat"].values

    def test_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        result = rare_encoder(df, "rare_cat", rare_perc=0.1)
        assert isinstance(result, pd.DataFrame)
