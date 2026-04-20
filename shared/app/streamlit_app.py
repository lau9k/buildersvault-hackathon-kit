"""Streamlit landing page for the BuildersVault Social Services Hackathon starter kit.

Run with:
    streamlit run shared/app/streamlit_app.py

The sidebar lets the user switch between Track 1 (Referral and Care
Coordination) and Track 2 (Food Security Delivery). Selection is stored in
``st.session_state`` so the Explore, Baseline, and Map pages pick it up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

# Make the `shared.src` package importable regardless of where Streamlit is
# launched from. The app file lives at ``shared/app/streamlit_app.py`` so
# the starter-kit root is two parents up.
_KIT_ROOT = Path(__file__).resolve().parents[2]
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

from shared.src.loaders import load_track1, load_track2  # noqa: E402


TRACK1_LABEL = "Track 1 — Referral and Care Coordination"
TRACK2_LABEL = "Track 2 — Food Security Delivery"

# Default data directories. Participants can override via environment variables
# before launching Streamlit (useful when running outside the repo root).
DEFAULT_TRACK1_DIR = os.environ.get(
    "TRACK1_DATA_DIR",
    str(_KIT_ROOT / "tracks" / "referral-care-coordination" / "data"),
)
DEFAULT_TRACK2_DIR = os.environ.get(
    "TRACK2_DATA_DIR",
    str(_KIT_ROOT / "tracks" / "food-security-delivery" / "data"),
)


@st.cache_data(show_spinner=False)
def _cached_load_track1(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Cache wrapper so KPI queries don't re-read parquet on every rerun."""
    return load_track1(data_dir)


@st.cache_data(show_spinner=False)
def _cached_load_track2(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Cache wrapper so KPI queries don't re-read parquet on every rerun."""
    return load_track2(data_dir)


def _kpi(label: str, value: Any, help_text: str | None = None) -> None:
    """Render a single KPI metric in a compact card."""
    st.metric(label=label, value=value, help=help_text)


def _render_track1_kpis(data_dir: str) -> None:
    """Summary KPIs for Track 1. Five numbers that tell the shape of the dataset."""
    try:
        tables = _cached_load_track1(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    clients = tables["clients"]
    referrals = tables["referrals"]
    encounters = tables["encounters"]
    consent = tables["consent"]
    dup_flags = tables["dup_flags"]

    # Chronic homeless rate: share of clients with chronic_homeless_flag True.
    chronic_rate = (
        (clients["chronic_homeless_flag"] == True).mean() * 100  # noqa: E712
        if "chronic_homeless_flag" in clients.columns
        else 0.0
    )
    # Active consent share: consents whose status is 'active'.
    active_consent_pct = (
        (consent["status"] == "active").mean() * 100 if "status" in consent.columns else 0.0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _kpi("Clients", f"{len(clients):,}", help_text="Unique rows in clients")
    with col2:
        _kpi("Referrals", f"{len(referrals):,}", help_text="Total referral transactions")
    with col3:
        _kpi("Encounters", f"{len(encounters):,}", help_text="Service encounters recorded")
    with col4:
        _kpi("Chronic rate", f"{chronic_rate:.1f}%", help_text="Share of clients flagged chronic")
    with col5:
        _kpi(
            "Duplicate pairs",
            f"{len(dup_flags):,}",
            help_text=f"Active consents: {active_consent_pct:.1f}%",
        )


def _render_track2_kpis(data_dir: str) -> None:
    """Summary KPIs for Track 2. Five numbers that tell the shape of the dataset."""
    try:
        tables = _cached_load_track2(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    clients = tables["clients"]
    requests = tables["requests"]
    routes = tables["routes"]
    stops = tables["stops"]
    drivers = tables["drivers"]

    # On-time stop share: route_stops where actual_arrival <= planned_arrival + 10 min.
    on_time_pct = 0.0
    if {"planned_arrival", "actual_arrival", "status"}.issubset(stops.columns):
        delivered = stops[stops["status"] == "delivered"].copy()
        if len(delivered) > 0:
            # Coerce to datetime for arithmetic; any non-parsable cells fall out as NaT.
            planned = pd.to_datetime(delivered["planned_arrival"], errors="coerce")
            actual = pd.to_datetime(delivered["actual_arrival"], errors="coerce")
            valid = planned.notna() & actual.notna()
            if valid.any():
                delta_min = (actual[valid] - planned[valid]).dt.total_seconds() / 60.0
                on_time_pct = float((delta_min <= 10).mean() * 100)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _kpi("Clients", f"{len(clients):,}", help_text="Active + closed clients")
    with col2:
        _kpi("Requests", f"{len(requests):,}", help_text="Scheduled delivery requests")
    with col3:
        _kpi("Routes", f"{len(routes):,}", help_text="Routes across the dataset window")
    with col4:
        _kpi("Drivers", f"{len(drivers):,}", help_text="Staff + volunteers on file")
    with col5:
        _kpi("On-time rate", f"{on_time_pct:.1f}%", help_text="Delivered stops within 10-min window")


def main() -> None:
    """Entry point. Renders the landing page."""
    st.set_page_config(
        page_title="BuildersVault Social Services Hackathon — Starter Kit",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("BuildersVault Social Services Hackathon — Starter Kit Explorer")
    st.caption("Pick a track to view KPIs, explore tables, run the baseline, and open the map.")

    # -- Sidebar: track picker. Selection persists via session_state. --------
    with st.sidebar:
        st.header("Track")
        track = st.radio(
            "Which track?",
            options=[TRACK1_LABEL, TRACK2_LABEL],
            index=0 if st.session_state.get("track", TRACK1_LABEL) == TRACK1_LABEL else 1,
            key="track",
        )
        st.divider()
        st.header("Data directories")
        track1_dir = st.text_input("Track 1 data dir", value=DEFAULT_TRACK1_DIR, key="track1_dir")
        track2_dir = st.text_input("Track 2 data dir", value=DEFAULT_TRACK2_DIR, key="track2_dir")
        st.caption("Set these to the folder holding the .parquet files for each track.")

    # -- Main area: KPIs for the selected track. -----------------------------
    if track == TRACK1_LABEL:
        st.subheader(TRACK1_LABEL)
        st.write(
            "Synthetic HIFIS-inspired data covering organizations, clients, referrals, "
            "service encounters, consent, data sharing agreements, and duplicate flags."
        )
        _render_track1_kpis(track1_dir)
    else:
        st.subheader(TRACK2_LABEL)
        st.write(
            "Synthetic Meals-on-Wheels-style data covering depots, vehicles, drivers, "
            "clients, delivery requests, routes, stops, and inventory."
        )
        _render_track2_kpis(track2_dir)

    st.divider()

    # -- Links to the multi-page sub-apps. Streamlit auto-discovers files in
    # -- pages/ so this is purely a wayfinding block for participants.
    st.subheader("Next steps")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Explore** — tables, schemas, null summaries, one EDA chart per table.")
        st.page_link("pages/1_Explore.py", label="Open Explore", icon=":mag:")
    with col_b:
        st.markdown("**Baseline** — Track 1 duplicate detector or Track 2 route quality scorer.")
        st.page_link("pages/2_Baseline.py", label="Open Baseline", icon=":straight_ruler:")
    with col_c:
        st.markdown("**Map** — Track 2 client/depot map; Track 1 is a stub.")
        st.page_link("pages/3_Map.py", label="Open Map", icon=":world_map:")


if __name__ == "__main__":
    main()
