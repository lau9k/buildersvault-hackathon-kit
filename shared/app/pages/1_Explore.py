"""Explore page — per-table head, describe, null summary, and one EDA chart.

Reuses the track picker stored in ``st.session_state`` by the landing page.
Kept deliberately small and fast: no ML, no heavy joins. One tab per table.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

_KIT_ROOT = Path(__file__).resolve().parents[3]
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

from shared.src.loaders import load_track1, load_track2  # noqa: E402
from shared.src.validators import null_summary  # noqa: E402

TRACK1_LABEL = "Track 1 — Referral and Care Coordination"
TRACK2_LABEL = "Track 2 — Food Security Delivery"


@st.cache_data(show_spinner=False)
def _load(track: str, data_dir: str) -> Dict[str, pd.DataFrame]:
    """Cached loader that dispatches on the selected track."""
    if track == TRACK1_LABEL:
        return load_track1(data_dir)
    return load_track2(data_dir)


def _render_eda_chart(df: pd.DataFrame, table_name: str) -> None:
    """Render a single EDA chart that makes sense for this table.

    We pick a cheap, universally useful view:
      - A categorical column gets a value_counts bar chart.
      - Else a numeric column gets a histogram.
      - Else we show a compact dtype summary as a fallback.
    """
    if len(df) == 0:
        st.info("No rows to chart.")
        return

    # Prefer a known domain column per table when present; fall back to heuristics.
    preferred = {
        "orgs": "org_type",
        "clients": "housing_status",
        "referrals": "status",
        "encounters": "encounter_type",
        "consent": "status",
        "dsa": "type",
        "dup_flags": "review_status",
        "depots": "capacity_meals_per_day",
        "vehicles": "type",
        "drivers": "role_type",
        "requests": "status",
        "routes": "route_status",
        "stops": "status",
        "items": "category",
        "request_items": "quantity",
    }
    target = preferred.get(table_name)

    if target and target in df.columns:
        series = df[target].dropna()
        if series.dtype == object or str(series.dtype).startswith("category"):
            counts = series.astype(str).value_counts().head(15)
            st.bar_chart(counts)
            st.caption(f"Top values of `{target}` in `{table_name}`.")
            return
        if pd.api.types.is_numeric_dtype(series):
            st.bar_chart(series.value_counts().sort_index().head(25))
            st.caption(f"Distribution of `{target}` in `{table_name}`.")
            return

    # Generic fallback: first numeric column, else first categorical column.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    if numeric_cols:
        col = numeric_cols[0]
        st.bar_chart(df[col].dropna().value_counts().sort_index().head(25))
        st.caption(f"Distribution of `{col}` (fallback chart).")
    elif cat_cols:
        col = cat_cols[0]
        st.bar_chart(df[col].astype(str).value_counts().head(15))
        st.caption(f"Top values of `{col}` (fallback chart).")
    else:
        st.info("No chartable column detected.")


def main() -> None:
    """Entry point. Renders tabs and per-table views."""
    st.set_page_config(page_title="Explore — Starter Kit", layout="wide")
    st.title("Explore")

    # Pull selections from the landing page. Defaults keep the page usable
    # if a participant deep-links here without visiting the landing page.
    track = st.session_state.get("track", TRACK1_LABEL)
    track1_dir = st.session_state.get(
        "track1_dir",
        str(_KIT_ROOT / "tracks" / "referral-care-coordination" / "data"),
    )
    track2_dir = st.session_state.get(
        "track2_dir",
        str(_KIT_ROOT / "tracks" / "food-security-delivery" / "data"),
    )
    data_dir = track1_dir if track == TRACK1_LABEL else track2_dir

    st.caption(f"Track: **{track}** — Data dir: `{data_dir}`")

    try:
        tables = _load(track, data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    # One tab per table. Streamlit tab labels survive page reruns.
    table_names = list(tables.keys())
    tabs = st.tabs(table_names)
    for name, tab in zip(table_names, tabs):
        with tab:
            df = tables[name]
            st.markdown(f"### `{name}` — {len(df):,} rows x {df.shape[1]} cols")

            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Head (first 20 rows)**")
                st.dataframe(df.head(20), use_container_width=True)
            with right:
                st.markdown("**Describe (numeric)**")
                # `describe()` on mixed dtypes can be slow; limit to numeric to keep snappy.
                try:
                    st.dataframe(df.describe(include="number"), use_container_width=True)
                except ValueError:
                    st.info("No numeric columns to describe.")

            st.markdown("**Null summary (top 15 columns by null %)**")
            st.dataframe(null_summary(df, name).head(15), use_container_width=True)

            st.markdown("**EDA chart**")
            _render_eda_chart(df, name)


if __name__ == "__main__":
    main()
