"""Feature engineering and EDA utility functions for the bank scoring pipeline."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

__all__ = [
    "reduce_mem_usage",
    "one_hot_encoder",
    "grab_col_names",
    "cat_analyzer",
    "corr_plot",
    "num_plot",
    "high_correlation",
    "missing_values",
    "rare_encoder",
    "plt_confusion_matrix",
    "display_importances",
    "display_roc_curve",
    "display_precision_recall",
]


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric column dtypes.

    Iterates over all columns and casts integers/floats to the smallest
    sub-type that can hold the observed min/max values without overflow.

    Args:
        df: Input DataFrame to optimise.

    Returns:
        The same DataFrame with reduced-precision numeric columns.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info("Memory usage before optimisation: %.2f MB", start_mem)

    for col in df.columns:
        col_type = df[col].dtype

        # Skip non-numeric columns (object, string, category, etc.)
        if not pd.api.types.is_numeric_dtype(col_type):
            df[col] = df[col].astype("category")
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # Skip columns where all values are NaN (no valid range to downcast)
        if pd.isna(c_min) or pd.isna(c_max):
            continue

        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(
        "Memory usage after optimisation: %.2f MB (reduced by %.1f%%)",
        end_mem,
        100 * (start_mem - end_mem) / start_mem,
    )
    return df


def one_hot_encoder(
    df: pd.DataFrame, nan_as_category: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode all categorical / object columns.

    Args:
        df: Input DataFrame.
        nan_as_category: If True, add a separate dummy column for NaN values.

    Returns:
        Tuple of (encoded DataFrame, list of newly added column names).
    """
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def grab_col_names(
    dataframe: pd.DataFrame,
    cat_th: int = 10,
    car_th: int = 20,
    show_date: bool = False,
) -> tuple:
    """Categorise DataFrame columns into semantic groups.

    Columns are split into:
    - date columns (datetime64)
    - categorical columns (object/category, or numeric with low cardinality)
    - high-cardinality categorical columns (cat_but_car)
    - numerical columns
    - numeric-but-categorical columns (numeric with < cat_th unique values)

    Args:
        dataframe: Input DataFrame.
        cat_th: Unique-value threshold below which a numeric column is treated as categorical.
        car_th: Unique-value threshold above which a categorical column is treated as high-cardinality.
        show_date: If True, include date columns in the return tuple.

    Returns:
        (cat_cols, cat_but_car, num_cols, num_but_cat) or
        (date_cols, cat_cols, cat_but_car, num_cols, num_but_cat) when show_date=True.
    """
    date_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "datetime64[ns]"]

    cat_cols = dataframe.select_dtypes(["object", "category"]).columns.tolist()

    num_but_cat = [
        col
        for col in dataframe.select_dtypes(["float", "integer"]).columns
        if dataframe[col].nunique() < cat_th
    ]
    cat_but_car = [
        col
        for col in dataframe.select_dtypes(["object", "category"]).columns
        if dataframe[col].nunique() > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [
        col
        for col in dataframe.select_dtypes(["float", "integer"]).columns
        if col not in num_but_cat
    ]

    logger.debug(
        "Columns — observations: %d, variables: %d, date: %d, cat: %d, num: %d, "
        "cat_but_car: %d, num_but_cat: %d",
        dataframe.shape[0],
        dataframe.shape[1],
        len(date_cols),
        len(cat_cols),
        len(num_cols),
        len(cat_but_car),
        len(num_but_cat),
    )

    if show_date:
        return date_cols, cat_cols, cat_but_car, num_cols, num_but_cat
    return cat_cols, cat_but_car, num_cols, num_but_cat


def cat_analyzer(
    dataframe: pd.DataFrame,
    variable: str,
    target: Optional[str] = None,
) -> pd.DataFrame:
    """Analyse a categorical variable, optionally against a binary target.

    Args:
        dataframe: Input DataFrame.
        variable: Name of the categorical column to analyse.
        target: Optional name of the binary target column. When provided,
                target statistics (count, mean, median, std) are included.

    Returns:
        Summary DataFrame printed to stdout.
    """
    if target is None:
        summary = pd.DataFrame(
            {
                "COUNT": dataframe[variable].value_counts(),
                "RATIO": dataframe[variable].value_counts() / len(dataframe),
            }
        )
    else:
        temp = dataframe[dataframe[target].notnull()]
        summary = pd.DataFrame(
            {
                "COUNT": dataframe[variable].value_counts(),
                "RATIO": dataframe[variable].value_counts() / len(dataframe),
                "TARGET_COUNT": dataframe.groupby(variable)[target].count(),
                "TARGET_MEAN": temp.groupby(variable)[target].mean(),
                "TARGET_MEDIAN": temp.groupby(variable)[target].median(),
                "TARGET_STD": temp.groupby(variable)[target].std(),
            }
        )
    print(variable)
    print(summary, end="\n\n\n")
    return summary


def corr_plot(
    data: pd.DataFrame,
    remove: list[str] = None,
    corr_coef: str = "pearson",
    figsize: tuple[int, int] = (20, 20),
) -> None:
    """Plot a lower-triangle correlation heatmap.

    Args:
        data: DataFrame containing numeric columns.
        remove: Column names to exclude from the plot.
        corr_coef: Correlation method passed to DataFrame.corr().
        figsize: Matplotlib figure size (width, height).
    """
    if remove is None:
        remove = ["Id"]
    cols = [x for x in data.columns if x not in remove]

    sns.set(font_scale=1.1)
    c = data[cols].corr(method=corr_coef)
    mask = np.triu(c.corr(method=corr_coef))
    plt.figure(figsize=figsize)
    sns.heatmap(
        c,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        square=True,
        mask=mask,
        linewidths=1,
        cbar=False,
    )
    plt.show()


def num_plot(
    data: pd.DataFrame,
    num_cols: list[str],
    remove: list[str] = None,
    hist_bins: int = 10,
    figsize: tuple[int, int] = (20, 4),
) -> None:
    """Plot histogram, boxplot, and KDE for each numeric column.

    Args:
        data: DataFrame containing the columns.
        num_cols: List of numeric column names to plot.
        remove: Column names to skip.
        hist_bins: Number of bins for the histogram.
        figsize: Matplotlib figure size per row.
    """
    if remove is None:
        remove = ["Id"]
    cols = [x for x in num_cols if x not in remove]

    for col in cols:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        data.hist(col, bins=hist_bins, ax=axes[0])
        data.boxplot(col, ax=axes[1], vert=False)
        try:
            sns.kdeplot(np.array(data[col]), ax=axes[2])
        except ValueError:
            pass

        axes[1].set_yticklabels([])
        axes[1].set_yticks([])
        axes[0].set_title(col + " | Histogram")
        axes[1].set_title(col + " | Boxplot")
        axes[2].set_title(col + " | Density")
        plt.show()


def high_correlation(
    data: pd.DataFrame,
    remove: list[str] = None,
    corr_coef: str = "pearson",
    corr_value: float = 0.7,
) -> None:
    """Print all variable pairs whose absolute correlation exceeds corr_value.

    Args:
        data: DataFrame containing numeric columns.
        remove: Column names to exclude (e.g. ID columns).
        corr_coef: Correlation method passed to DataFrame.corr().
        corr_value: Minimum absolute correlation to report.
    """
    if remove is None:
        remove = ["SK_ID_CURR", "SK_ID_BUREAU"]
    cols = [x for x in data.columns if x not in remove]
    c = data[cols].corr(method=corr_coef)

    for i in c.columns:
        cr = c.loc[i].loc[(c.loc[i] >= corr_value) | (c.loc[i] <= -corr_value)].drop(i)
        if len(cr) > 0:
            print(i)
            print("-------------------------------")
            print(cr.sort_values(ascending=False))
            print("\n")


def missing_values(data: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
    """Report and remove columns that are entirely missing.

    Columns with 100% missing values are dropped in-place.

    Args:
        data: DataFrame to analyse (modified in-place for fully-missing columns).
        plot: If True, display a bar chart of missing ratios.

    Returns:
        Summary DataFrame with columns Feature, Num_Missing, Missing_Ratio, DataTypes.
    """
    mst = pd.DataFrame(
        {
            "Num_Missing": data.isnull().sum(),
            "Missing_Ratio": data.isnull().sum() / data.shape[0],
        }
    ).sort_values("Num_Missing", ascending=False)
    mst["DataTypes"] = data[mst.index].dtypes.values
    mst = mst[mst.Num_Missing > 0].reset_index().rename({"index": "Feature"}, axis=1)

    logger.info("Variables with missing values: %d", mst.shape[0])

    fully_missing = mst[mst.Missing_Ratio >= 1.0].Feature.tolist()
    if fully_missing:
        logger.warning("Dropping fully-missing columns: %s", fully_missing)
        data.drop(fully_missing, axis=1, inplace=True)

    if plot:
        plt.figure(figsize=(25, 8))
        p = sns.barplot(x=mst.Feature, y=mst.Missing_Ratio)
        for label in p.get_xticklabels():
            label.set_rotation(90)
        plt.show()

    return mst


def rare_encoder(data: pd.DataFrame, col: str, rare_perc: float) -> pd.DataFrame:
    """Replace infrequent categories in a column with the label 'Rare'.

    Args:
        data: DataFrame containing the column.
        col: Name of the categorical column to encode.
        rare_perc: Frequency threshold below which a category is considered rare.

    Returns:
        DataFrame with rare categories replaced (modified in-place and returned).
    """
    freq = data[col].value_counts() / len(data)
    rare_categories = freq[freq < rare_perc].index
    data[col] = np.where(data[col].isin(rare_categories), "Rare", data[col])
    return data


def plt_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Plot a confusion matrix heatmap and print a classification report.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="YlGnBu")
    plt.ylabel("True classes", fontsize=14)
    plt.xlabel("Predicted classes", fontsize=14)
    plt.title("Confusion Matrix", fontsize=20)
    print(classification_report(y_true, y_pred))


def display_importances(feature_importance_df_: pd.DataFrame) -> None:
    """Plot the top-50 LightGBM feature importances averaged over cross-validation folds.

    Args:
        feature_importance_df_: DataFrame with columns 'feature' and 'importance',
                                 one row per (feature, fold) pair.
    """
    cols = (
        feature_importance_df_[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50]
        .index
    )
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


def display_roc_curve(
    y_: pd.Series,
    oof_preds_: np.ndarray,
    folds_idx_: list,
) -> None:
    """Plot per-fold and average ROC curves from cross-validation predictions.

    Args:
        y_: Full ground-truth label series.
        oof_preds_: Out-of-fold probability predictions (same length as y_).
        folds_idx_: List of (train_idx, val_idx) tuples from the CV splitter.
    """
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        fpr, tpr, _ = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"ROC fold {n_fold + 1} (AUC = {score:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random", alpha=0.8)
    fpr, tpr, _ = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color="b",
        label=f"Avg ROC (AUC = {score:.4f} ± {np.std(scores):.4f})",
        lw=2,
        alpha=0.8,
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LightGBM ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()


def display_precision_recall(
    y_: pd.Series,
    oof_preds_: np.ndarray,
    folds_idx_: list,
) -> None:
    """Plot per-fold and average Precision-Recall curves from cross-validation.

    Args:
        y_: Full ground-truth label series.
        oof_preds_: Out-of-fold probability predictions (same length as y_).
        folds_idx_: List of (train_idx, val_idx) tuples from the CV splitter.
    """
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        fpr, tpr, _ = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"AP fold {n_fold + 1} (AP = {score:.4f})")

    precision, recall, _ = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color="b",
        label=f"Avg PR (AP = {score:.4f} ± {np.std(scores):.4f})",
        lw=2,
        alpha=0.8,
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("LightGBM Precision-Recall Curve")
    plt.legend(loc="best")
    plt.tight_layout()
