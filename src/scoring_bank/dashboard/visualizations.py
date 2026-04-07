"""Dashboard visualisation helpers: radar charts and bar plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

_RADAR_VARIABLES = ("Loan Duration", "Annuity", "Age", "Employment Start", "Annuity/Income")

_COLUMN_MAP = {
    "Durée emprunt": "Loan Duration",
    "Annuités": "Annuity",
    "Âge": "Age",
    "Début contrat travail": "Employment Start",
    "Annuités/revenus": "Annuity/Income",
}


class _ComplexRadar:
    """Multi-axis radar chart helper."""

    def __init__(
        self,
        fig: plt.Figure,
        variables: tuple[str, ...],
        ranges: list[tuple[float, float]],
        n_ordinate_levels: int = 6,
    ) -> None:
        angles = np.arange(0, 360, 360.0 / len(variables))
        axes = [
            fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True, label=f"axes{i}")
            for i in range(len(variables))
        ]
        axes[0].set_thetagrids(angles, labels=[])

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = [f"{round(x, 2)}" for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]
            gridlabel[0] = ""
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            ax.set_ylim(*ranges[i])

        ticks = angles
        axes[-1].set_xticks(np.deg2rad(ticks))
        axes[-1].set_xticklabels(variables, fontsize=10)

        angle_arr = np.linspace(0, 2 * np.pi, len(axes[-1].get_xticklabels()) + 1)
        angle_arr[np.cos(angle_arr) < 0] = angle_arr[np.cos(angle_arr) < 0] + np.pi
        angle_arr = np.rad2deg(angle_arr)

        for label, angle in zip(axes[-1].get_xticklabels(), angle_arr):
            x, y = label.get_position()
            lab = axes[-1].text(
                x,
                y - 0.5,
                label.get_text(),
                transform=label.get_transform(),
                ha=label.get_ha(),
                va=label.get_va(),
            )
            lab.set_rotation(angle)
            lab.set_fontsize(16)
            lab.set_fontweight("bold")
        axes[-1].set_xticklabels([])

        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def _scale_data(self, data: list[float]) -> list[float]:
        def _invert(x: float, limits: tuple[float, float]) -> float:
            return limits[1] - (x - limits[0])

        x1, x2 = self.ranges[0]
        d = data[0]
        if x1 > x2:
            d = _invert(d, (x1, x2))
            x1, x2 = x2, x1
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], self.ranges[1:]):
            if y1 > y2:
                d = _invert(d, (y1, y2))
                y1, y2 = y2, y1
            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
        return sdata

    def plot(self, data: list[float], *args, **kw) -> None:
        sdata = self._scale_data(data)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data: list[float], *args, **kw) -> None:
        sdata = self._scale_data(data)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


def radar_chart(
    client_row: pd.DataFrame,
    ok_group: pd.DataFrame,
    impayes_group: pd.DataFrame,
    param_label: str,
) -> None:
    """Render a radar chart comparing a client to good/bad-payer group averages.

    The chart is rendered directly into the active Streamlit context via st.pyplot.

    Args:
        client_row: Single-row DataFrame with the client's interpretable features
                    (columns use the English names from _COLUMN_MAP or the originals).
        ok_group: Single-row DataFrame with mean values for clients without default.
        impayes_group: Single-row DataFrame with mean values for clients with default.
        param_label: Human-readable label for the comparison parameter (used in legend).
    """
    # Rename French column names if present
    client_row = client_row.rename(columns=_COLUMN_MAP)
    ok_group = ok_group.rename(columns=_COLUMN_MAP)
    impayes_group = impayes_group.rename(columns=_COLUMN_MAP)

    variables = _RADAR_VARIABLES
    data_client = [client_row.iloc[0][v] for v in variables]

    ranges = [
        (
            min(client_row.iloc[0][v], ok_group.iloc[0][v], impayes_group.iloc[0][v]) - delta,
            max(client_row.iloc[0][v], ok_group.iloc[0][v], impayes_group.iloc[0][v]) + delta,
        )
        for v, delta in zip(variables, [5, 5000, 5, 1, 5])
    ]

    fig = plt.figure(figsize=(6, 6))
    radar = _ComplexRadar(fig, variables, ranges)
    radar.plot(data_client, label="Our client")
    radar.fill(data_client, alpha=0.2)
    radar.plot(
        [ok_group.iloc[0][v] for v in variables],
        label=f"Avg similar clients — no default ({param_label})",
        color="g",
    )
    radar.plot(
        [impayes_group.iloc[0][v] for v in variables],
        label=f"Avg similar clients — default ({param_label})",
        color="r",
    )
    fig.legend(bbox_to_anchor=(1.7, 1))
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------

_SEGMENT_LABELS: dict[str, str] = {
    "CODE_GENDER": "Gender",
    "ORGANIZATION_TYPE": "Company Type",
    "NAME_EDUCATION_TYPE": "Education Level",
    "AMT_INCOME": "Income Level",
    "NAME_FAMILY_STATUS": "Marital Status",
    "Count": "Count",
    "Cible": "Target",
    "Percentage": "Percentage",
}


def bar_plot(df: pd.DataFrame, col: str) -> None:
    """Render an interactive Plotly bar chart of default counts by demographic segment.

    Args:
        df: Aggregated segment DataFrame with columns [col, 'Count', 'Cible', 'Percentage'].
        col: Name of the grouping column (e.g. 'CODE_GENDER').
    """
    title = f"Distribution of defaults by {_SEGMENT_LABELS.get(col, col).lower()}"
    fig = px.bar(
        df,
        x=col,
        y="Count",
        color="Cible",
        text="Percentage",
        labels=_SEGMENT_LABELS,
        color_discrete_sequence=["#90ee90", "#ff4500"],
        title=title,
    )
    st.plotly_chart(fig)
