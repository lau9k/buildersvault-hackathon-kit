"""Map page — geospatial view of Track 2 clients and depots.

Track 1 has no lat/lng on organizations in the current schema, so this page
shows a stub pointing at how to extend the schema if a participant wants to
add geospatial views for Track 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

_KIT_ROOT = Path(__file__).resolve().parents[3]
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

from shared.src.loaders import load_track2  # noqa: E402

TRACK1_LABEL = "Track 1 — Referral and Care Coordination"
TRACK2_LABEL = "Track 2 — Food Security Delivery"


def _hash_color(value: str) -> list[int]:
    """Map an arbitrary string id to a stable pseudo-random RGB triple.

    Used to colour client points by home depot without pulling in a
    full palette library. Hash mod 256 gives enough contrast for <10 depots.
    """
    h = abs(hash(str(value)))
    return [h % 200 + 40, (h // 7) % 200 + 40, (h // 13) % 200 + 40]


def _render_track2(data_dir: str) -> None:
    """Render a pydeck map of clients coloured by home depot + depot pins."""
    try:
        tables = load_track2(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    clients = tables["clients"]
    depots = tables["depots"]

    # Guard: the map needs numeric lat/lng; drop rows that lack them.
    client_geo = clients.dropna(subset=["lat", "lng"]).copy()
    if client_geo.empty:
        st.warning("No clients have lat/lng populated.")
        return

    # Stable colour per home depot.
    client_geo["color"] = client_geo["home_depot_id"].fillna("NONE").map(_hash_color)
    client_geo["home_depot_label"] = client_geo["home_depot_id"].fillna("unassigned")

    depot_geo = depots.dropna(subset=["lat", "lng"]).copy()

    # Centre the view on the mean of client coordinates.
    view_state = pdk.ViewState(
        latitude=float(client_geo["lat"].mean()),
        longitude=float(client_geo["lng"].mean()),
        zoom=11,
        pitch=0,
    )

    client_layer = pdk.Layer(
        "ScatterplotLayer",
        data=client_geo,
        get_position="[lng, lat]",
        get_fill_color="color",
        get_radius=60,
        pickable=True,
        opacity=0.7,
    )
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=depot_geo,
        get_position="[lng, lat]",
        get_fill_color=[10, 10, 10],
        get_radius=250,
        pickable=True,
        opacity=0.9,
    )
    depot_label_layer = pdk.Layer(
        "TextLayer",
        data=depot_geo,
        get_position="[lng, lat]",
        get_text="name",
        get_size=16,
        get_color=[0, 0, 0],
        get_alignment_baseline="bottom",
    )

    deck = pdk.Deck(
        layers=[client_layer, depot_layer, depot_label_layer],
        initial_view_state=view_state,
        tooltip={"text": "{first_name} {last_name}\nDepot: {home_depot_label}"},
        map_style=None,
    )

    st.subheader("Clients coloured by home depot")
    st.caption(
        f"{len(client_geo):,} clients with coordinates, {len(depot_geo):,} depots. "
        "Hover a client point for details."
    )
    st.pydeck_chart(deck)

    st.markdown("**Depots**")
    st.dataframe(
        depot_geo[["depot_id", "name", "address", "capacity_meals_per_day"]],
        use_container_width=True,
    )


def _render_track1() -> None:
    """Stub view for Track 1. Orgs table has no lat/lng yet."""
    st.info(
        "Map coming in Track 1 — add lat/lng to organizations if needed.\n\n"
        "To enable: geocode the organizations.address_* columns and add `lat` and "
        "`lng` floats to the schema. This page can then mirror the Track 2 layout."
    )


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="Map — Starter Kit", layout="wide")
    st.title("Map")

    track = st.session_state.get("track", TRACK1_LABEL)
    track2_dir = st.session_state.get(
        "track2_dir",
        str(_KIT_ROOT / "tracks" / "food-security-delivery" / "data"),
    )

    st.caption(f"Track: **{track}**")

    if track == TRACK2_LABEL:
        _render_track2(track2_dir)
    else:
        _render_track1()


if __name__ == "__main__":
    main()
