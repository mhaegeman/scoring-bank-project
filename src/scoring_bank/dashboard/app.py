"""Bank credit scoring Streamlit dashboard.

Run with:
    streamlit run src/scoring_bank/dashboard/app.py
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from scoring_bank import config
from scoring_bank.dashboard.visualizations import bar_plot, radar_chart
from scoring_bank.data.loader import (
    load_api_data,
    load_group_data,
    load_interpretable_data,
    load_nn_data,
    load_segment_data,
)
from scoring_bank.models.scorer import load_model, predict_default_proba
from scoring_bank.models.similarity import find_similar_clients, load_nn_model, load_scaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Credit Acceptance Application")
st.write("# Credit Acceptance Application")

# ---------------------------------------------------------------------------
# Sidebar / input controls
# ---------------------------------------------------------------------------

col1, _, col3 = st.columns([5, 1, 10])

with col1:
    st.write("### Acceptance threshold")
    threshold = st.slider(
        "Maximum accepted default probability:",
        min_value=0.00,
        max_value=1.00,
        value=config.DEFAULT_THRESHOLD,
        step=0.01,
    )
    st.write("### Enter client ID:")
    client_id = st.number_input(
        " ",
        min_value=config.CLIENT_ID_MIN,
        max_value=config.CLIENT_ID_MAX,
    )

# ---------------------------------------------------------------------------
# Cached data & model loading
# ---------------------------------------------------------------------------


@st.cache_data
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_api_data()
    df_int = load_interpretable_data()
    df_int_no_unit = load_interpretable_data(config.INTERPRETABLE_NO_UNIT_CSV)
    df_group = load_group_data()
    df_nn = load_nn_data()
    return df, df_int, df_int_no_unit, df_group, df_nn


@st.cache_resource
def _load_models():
    lgbm = load_model()
    nn = load_nn_model()
    scaler = load_scaler()
    return lgbm, nn, scaler


@st.cache_data
def _load_segments() -> dict[str, pd.DataFrame]:
    return {
        "genre": load_segment_data("genre"),
        "income": load_segment_data("income"),
        "education_type": load_segment_data("education_type"),
        "organization_type": load_segment_data("organization_type"),
        "family": load_segment_data("family"),
    }


df, df_int, df_int_no_unit, df_group, df_nn = _load_data()
lgbm, nn, scaler = _load_models()
segments = _load_segments()

# Feature columns (everything except ID and target)
feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
x_test = df[feature_cols]
y_test = df["TARGET"]

# ---------------------------------------------------------------------------
# Main dashboard logic
# ---------------------------------------------------------------------------

if (df["SK_ID_CURR"] == client_id).sum() == 0:
    st.warning("Unknown client ID.")
else:
    df_client = df[df["SK_ID_CURR"] == client_id]
    df_client_int = df_int[df_int["Identifiant"] == client_id].copy()
    df_client_int_no_unit = df_int_no_unit[df_int_no_unit["Identifiant"] == client_id].copy()
    df_client_int.set_index("Identifiant", inplace=True)

    # --- Prediction ---
    with col1:
        proba = predict_default_proba(lgbm, df_client, feature_cols)
        st.write("## Prediction")
        st.metric("Default probability", f"{proba:.2%}")

        default_history = df_client_int["Défaut paiement"].iloc[0]
        note = f"Client has previously defaulted: {default_history}"

        if proba < threshold:
            st.success("Result: Low default risk")
            st.write(f"> {note}")
        else:
            st.warning("Result: High default risk — monitor carefully")
            st.write(f"> {note}")

    # --- Client info ---
    with col3:
        st.write("## Client information")
        st.dataframe(df_client_int.drop("Défaut paiement", axis=1, errors="ignore"))

    # --- Similar clients ---
    # The pre-trained StandardScaler and NearestNeighbors were fitted with all 6
    # INTERPRETABLE_FEATURES including SK_ID_CURR, so we must pass all 6 here.
    similar = find_similar_clients(
        nn,
        scaler,
        df_client,
        df_nn,
        config.INTERPRETABLE_FEATURES,
    )
    with col3:
        st.write("## Similar client profiles in database")
        similar_ids = similar["SK_ID_CURR"].tolist() if "SK_ID_CURR" in similar.columns else []
        if similar_ids:
            similar_int = df_int[df_int["Identifiant"].isin(similar_ids)].set_index("Identifiant")
            st.dataframe(similar_int)

    # --- Model metrics ---
    st.write("## Model training metrics")
    with st.expander("Show metrics"):
        col_m1, _, col_m3 = st.columns([3, 1, 5])
        with col_m1:
            metric_choice = st.selectbox(
                "Choose a metric:",
                options=("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            )

        with col_m3:
            y_pred = lgbm.predict(x_test)
            y_proba = lgbm.predict_proba(x_test)[:, 1]

            if metric_choice == "Confusion Matrix":
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                import seaborn as sns

                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                ax.set_ylabel("True label")
                ax.set_xlabel("Predicted label")
                st.pyplot(fig)
                plt.close(fig)

            elif metric_choice == "ROC Curve":
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fpr, tpr, color="b", lw=2, label="ROC curve")
                ax.plot([0, 1], [0, 1], linestyle="--", color="r", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)

            elif metric_choice == "Precision-Recall Curve":
                st.subheader("Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(recall, precision, color="b", lw=2)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                st.pyplot(fig)
                plt.close(fig)

    # --- Comparison charts ---
    st.write("## Interactive comparison charts")
    with st.expander("Show charts"):
        col_c1, _, col_c3 = st.columns([10, 1, 10])

        _SEGMENT_OPTIONS = {
            "Gender": ("genre", "CODE_GENDER"),
            "Company Type": ("organization_type", "ORGANIZATION_TYPE"),
            "Education Level": ("education_type", "NAME_EDUCATION_TYPE"),
            "Income Level": ("income", "AMT_INCOME"),
            "Marital Status": ("family", "NAME_FAMILY_STATUS"),
        }

        with col_c1:
            param = st.selectbox(
                "Choose a comparison parameter:",
                options=list(_SEGMENT_OPTIONS.keys()),
            )
            seg_key, col_name = _SEGMENT_OPTIONS[param]
            seg_df = segments[seg_key]

            client_cat_row = df_group[df_group["SK_ID_CURR"] == client_id]
            if not client_cat_row.empty and col_name in client_cat_row.columns:
                cat = client_cat_row[col_name].iloc[0]
                st.write(f"Client's {param.lower()}: **{cat}**")

            bar_plot(seg_df, col_name)

        with col_c3:
            st.write(f"Radar chart: client vs {param.lower()} peers")

            # Build client row and group averages for radar chart
            radar_cols_fr = [
                "Durée emprunt",
                "Annuités",
                "Âge",
                "Début contrat travail",
                "Annuités/revenus",
            ]
            if all(c in df_client_int_no_unit.columns for c in radar_cols_fr):
                client_radar = df_client_int_no_unit.drop("Identifiant", axis=1, errors="ignore")

                if not client_cat_row.empty and col_name in client_cat_row.columns:
                    cat = client_cat_row[col_name].iloc[0]
                    seg_with_col = (
                        seg_df[seg_df[col_name] == cat]
                        if col_name in seg_df.columns
                        else pd.DataFrame()
                    )
                    ok_rows = (
                        seg_with_col[seg_with_col["Cible"] == 0]
                        if "Cible" in seg_with_col.columns
                        else pd.DataFrame()
                    )
                    bad_rows = (
                        seg_with_col[seg_with_col["Cible"] == 1]
                        if "Cible" in seg_with_col.columns
                        else pd.DataFrame()
                    )

                    radar_data_cols = [c for c in radar_cols_fr if c in seg_df.columns]
                    if not ok_rows.empty and not bad_rows.empty and radar_data_cols:
                        radar_chart(
                            client_radar[radar_data_cols],
                            ok_rows[radar_data_cols].head(1),
                            bad_rows[radar_data_cols].head(1),
                            param,
                        )
