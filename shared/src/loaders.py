"""Parquet loaders for Track 1 and Track 2 datasets.

Each loader returns a dictionary of pandas DataFrames keyed by a short,
human-friendly table name. The loaders fail loudly with a friendly message
when expected files are missing so that hackathon participants can tell
immediately whether they have synthesized the dataset yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# Mapping of short keys used throughout the kit to their parquet file names on disk.
# Keeping this here (rather than inlining file names at call sites) means notebooks,
# Streamlit pages, and tests all agree on the same schema surface.
TRACK1_FILES: Dict[str, str] = {
    "orgs": "organizations.parquet",
    "clients": "clients.parquet",
    "referrals": "referrals.parquet",
    "encounters": "service_encounters.parquet",
    "consent": "consent_records.parquet",
    "dsa": "data_sharing_agreements.parquet",
    "dup_flags": "duplicate_flags.parquet",
}

TRACK2_FILES: Dict[str, str] = {
    "depots": "depots.parquet",
    "vehicles": "vehicles.parquet",
    "drivers": "drivers.parquet",
    "clients": "clients.parquet",
    "requests": "delivery_requests.parquet",
    "routes": "routes.parquet",
    "stops": "route_stops.parquet",
    "items": "inventory_items.parquet",
    "request_items": "delivery_request_items.parquet",
}


def _load_bundle(data_dir: str | Path, files: Dict[str, str], track_label: str) -> Dict[str, pd.DataFrame]:
    """Load a bundle of parquet files into a dict of DataFrames.

    Args:
        data_dir: Directory containing the parquet files.
        files: Mapping from short key to parquet filename.
        track_label: Human-readable label used in error messages.

    Returns:
        Dict mapping short key to pandas DataFrame.

    Raises:
        FileNotFoundError: When ``data_dir`` does not exist or a required
            parquet file is missing. The message includes concrete next steps.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"[{track_label}] Data directory does not exist: {data_path}.\n"
            "Run the track's generator (make generate) before loading."
        )

    # Check for missing files up front so we can report all gaps in one message
    # rather than failing on the first one. This is gentler for participants.
    missing = [name for name in files.values() if not (data_path / name).exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"[{track_label}] Missing parquet files in {data_path}:\n  - {joined}\n"
            "Run the track's generator (make generate) to produce them."
        )

    # Lazy per-table read; parquet is fast but we still keep this explicit so
    # callers can see exactly which files back which keys.
    return {key: pd.read_parquet(data_path / filename) for key, filename in files.items()}


def load_track1(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load Track 1 (Referral and Care Coordination) parquet tables.

    Args:
        data_dir: Directory containing Track 1 parquet files.

    Returns:
        Dict with keys: orgs, clients, referrals, encounters, consent, dsa, dup_flags.
    """
    return _load_bundle(data_dir, TRACK1_FILES, track_label="Track 1")


def load_track2(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load Track 2 (Food Security Delivery) parquet tables.

    Args:
        data_dir: Directory containing Track 2 parquet files.

    Returns:
        Dict with keys: depots, vehicles, drivers, clients, requests, routes,
        stops, items, request_items.
    """
    return _load_bundle(data_dir, TRACK2_FILES, track_label="Track 2")
