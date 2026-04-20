"""Baseline page — rule-based references that participants must beat.

- Track 1: simple blocking + rule-based duplicate detector. Compare against
  the ``duplicate_flags`` table (treated as ground truth) and report
  precision, recall, and F1, plus a top-N leaderboard of matched pairs.
- Track 2: heuristic route quality scorer. Rank routes using on-time rate,
  no-answer count, and plan/actual deltas; show top 5 best and worst.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

_KIT_ROOT = Path(__file__).resolve().parents[3]
if str(_KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_KIT_ROOT))

from shared.src.loaders import load_track1, load_track2  # noqa: E402

TRACK1_LABEL = "Track 1 — Referral and Care Coordination"
TRACK2_LABEL = "Track 2 — Food Security Delivery"


# ---------------------------------------------------------------------------
# Track 1 — duplicate detector
# ---------------------------------------------------------------------------

def _normalise_name(value: object) -> str:
    """Lowercase and strip whitespace; return empty string for nulls.

    Used for blocking and string comparison so that case and padding noise
    injected by ``messiness.inject_name_case`` does not kill recall.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _generate_candidate_pairs(clients: pd.DataFrame) -> List[Tuple[str, str, float, str]]:
    """Yield candidate duplicate pairs using cheap blocking rules.

    Blocking keys (any match triggers comparison, to keep recall high):
      - (last_name_lower, dob)
      - (first_name_lower, last_name_lower)

    For each candidate pair we produce a score in [0, 1] combining:
      - exact last name match (weight 0.3)
      - exact first name match (weight 0.3)
      - exact dob match (weight 0.3)
      - any alias substring overlap (weight 0.1)

    Returns a list of (client_id_primary, client_id_secondary, score, reason).
    """
    # Pre-normalise the columns we compare on. Empty string for nulls so that
    # two "unknown" rows do not accidentally block together.
    df = clients[["client_id", "first_name", "last_name", "dob", "aliases"]].copy()
    df["fn"] = df["first_name"].map(_normalise_name)
    df["ln"] = df["last_name"].map(_normalise_name)
    df["dob_s"] = df["dob"].astype(str).fillna("")
    df["aliases_s"] = df["aliases"].map(_normalise_name)

    pairs: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}

    def _emit(a: pd.Series, b: pd.Series) -> None:
        # Keep pair IDs ordered so (a,b) and (b,a) collapse into one key.
        pk = (a["client_id"], b["client_id"]) if a["client_id"] < b["client_id"] else (b["client_id"], a["client_id"])
        score = 0.0
        reasons: List[str] = []
        if a["ln"] and a["ln"] == b["ln"]:
            score += 0.3
            reasons.append("ln")
        if a["fn"] and a["fn"] == b["fn"]:
            score += 0.3
            reasons.append("fn")
        if a["dob_s"] and a["dob_s"] == b["dob_s"]:
            score += 0.3
            reasons.append("dob")
        if a["aliases_s"] and b["aliases_s"]:
            # Alias overlap: cheap token intersection on semicolon-delimited lists.
            a_tokens = {t.strip() for t in a["aliases_s"].split(";") if t.strip()}
            b_tokens = {t.strip() for t in b["aliases_s"].split(";") if t.strip()}
            if a_tokens & b_tokens:
                score += 0.1
                reasons.append("alias")
        # Only keep if we have at least a last-name match, which is the
        # minimum credible signal across the injected messiness patterns.
        if score > 0 and "ln" in reasons:
            existing = pairs.get(pk)
            if existing is None or score > existing[0]:
                pairs[pk] = (score, reasons)

    # Block 1: last name + dob.
    for _, block in df.groupby(["ln", "dob_s"]):
        if len(block) < 2:
            continue
        rows = block.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                _emit(pd.Series(rows[i]), pd.Series(rows[j]))

    # Block 2: first name + last name.
    for _, block in df.groupby(["fn", "ln"]):
        if len(block) < 2:
            continue
        rows = block.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                _emit(pd.Series(rows[i]), pd.Series(rows[j]))

    return [(pk[0], pk[1], score, "+".join(reasons)) for pk, (score, reasons) in pairs.items()]


def _score_duplicates(
    clients: pd.DataFrame, dup_flags: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run the baseline detector and score it against ``dup_flags``.

    Args:
        clients: Clients table from Track 1.
        dup_flags: Ground-truth duplicate pairs (confirmed_duplicate rows).
        threshold: Score cutoff above which a pair is classified as duplicate.

    Returns:
        (leaderboard DataFrame, metrics dict with precision/recall/f1/counts).
    """
    candidates = _generate_candidate_pairs(clients)
    cand_df = pd.DataFrame(candidates, columns=["client_id_primary", "client_id_secondary", "match_score", "reason"])

    # Ground truth: confirmed_duplicate pairs only. Order-normalised.
    truth = dup_flags[dup_flags["review_status"] == "confirmed_duplicate"][
        ["client_id_primary", "client_id_secondary"]
    ].copy()
    truth["pair"] = truth.apply(
        lambda r: tuple(sorted((r["client_id_primary"], r["client_id_secondary"]))),
        axis=1,
    )
    truth_pairs = set(truth["pair"])

    predicted = cand_df[cand_df["match_score"] >= threshold].copy()
    predicted["pair"] = predicted.apply(
        lambda r: tuple(sorted((r["client_id_primary"], r["client_id_secondary"]))),
        axis=1,
    )
    predicted_pairs = set(predicted["pair"])

    tp = len(predicted_pairs & truth_pairs)
    fp = len(predicted_pairs - truth_pairs)
    fn = len(truth_pairs - predicted_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "candidates": len(cand_df),
        "predicted": len(predicted_pairs),
        "truth": len(truth_pairs),
    }

    # Leaderboard is sorted by score descending, showing top 50 pairs.
    leaderboard = cand_df.sort_values("match_score", ascending=False).head(50).reset_index(drop=True)
    leaderboard["is_true_match"] = leaderboard.apply(
        lambda r: tuple(sorted((r["client_id_primary"], r["client_id_secondary"]))) in truth_pairs,
        axis=1,
    )
    return leaderboard, metrics


def _render_track1(data_dir: str) -> None:
    """Render the Track 1 baseline view."""
    try:
        tables = load_track1(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("Baseline: rule-based duplicate detector")
    threshold = st.slider("Match-score threshold", 0.3, 1.0, 0.6, 0.05)

    with st.spinner("Blocking + scoring candidate pairs..."):
        leaderboard, metrics = _score_duplicates(tables["clients"], tables["dup_flags"], threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    c2.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    c3.metric("F1", f"{metrics['f1'] * 100:.1f}%")
    c4.metric("Candidate pairs", f"{metrics['candidates']:,}")

    st.caption(
        f"Predicted duplicates at threshold {threshold:.2f}: {metrics['predicted']:,} "
        f"(ground truth: {metrics['truth']:,}; TP {metrics['true_positive']}, "
        f"FP {metrics['false_positive']}, FN {metrics['false_negative']})."
    )

    st.markdown("**Top candidate pairs (by score)**")
    st.dataframe(leaderboard, use_container_width=True)


# ---------------------------------------------------------------------------
# Track 2 — route quality scorer
# ---------------------------------------------------------------------------

def _score_routes(routes: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite quality score per route.

    Components (each normalised to [0, 1]):
      - on_time_share: delivered stops arriving within 10 minutes of plan.
      - delivery_success: delivered stops over planned stops.
      - time_adherence: 1 - abs(actual_minutes - planned_minutes) / planned_minutes.

    Final score is a simple mean of the three components. Missing pieces
    are filled with 0 so that routes with no telemetry score low and
    surface as problems to investigate.
    """
    if len(routes) == 0:
        return pd.DataFrame()

    # Per-route on-time share from stops.
    stops_ok = stops.copy()
    stops_ok["planned_arrival"] = pd.to_datetime(stops_ok["planned_arrival"], errors="coerce")
    stops_ok["actual_arrival"] = pd.to_datetime(stops_ok["actual_arrival"], errors="coerce")
    delivered = stops_ok[stops_ok["status"] == "delivered"].copy()
    delivered["delta_min"] = (
        (delivered["actual_arrival"] - delivered["planned_arrival"]).dt.total_seconds() / 60.0
    )
    delivered["on_time"] = delivered["delta_min"] <= 10

    per_route_on_time = (
        delivered.groupby("route_id")["on_time"].mean().rename("on_time_share").to_frame()
    )

    out = routes.merge(per_route_on_time, left_on="route_id", right_index=True, how="left")
    # Fill missing on-time share with 0 — absence is evidence of trouble.
    out["on_time_share"] = out["on_time_share"].fillna(0.0)

    # Delivery success: meals_delivered / meals_planned, capped at 1.
    meals_planned = out["meals_planned"].replace(0, np.nan)
    out["delivery_success"] = (out["meals_delivered"] / meals_planned).clip(upper=1.0).fillna(0.0)

    # Time adherence: closer actual to planned is better.
    planned_min = out["planned_time_minutes"].replace(0, np.nan)
    deviation = (out["actual_time_minutes"] - out["planned_time_minutes"]).abs() / planned_min
    out["time_adherence"] = (1 - deviation).clip(lower=0.0, upper=1.0).fillna(0.0)

    out["quality_score"] = (
        out[["on_time_share", "delivery_success", "time_adherence"]].mean(axis=1).round(3)
    )
    return out[
        [
            "route_id",
            "service_date",
            "driver_id",
            "planned_stops",
            "actual_stops",
            "meals_planned",
            "meals_delivered",
            "no_answer_count",
            "on_time_share",
            "delivery_success",
            "time_adherence",
            "quality_score",
        ]
    ].sort_values("quality_score", ascending=False)


def _render_track2(data_dir: str) -> None:
    """Render the Track 2 baseline view."""
    try:
        tables = load_track2(data_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    st.subheader("Baseline: heuristic route quality scorer")

    with st.spinner("Scoring routes..."):
        scored = _score_routes(tables["routes"], tables["stops"])

    if scored.empty:
        st.info("No routes to score yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Routes scored", f"{len(scored):,}")
    c2.metric("Mean quality", f"{scored['quality_score'].mean():.3f}")
    c3.metric("Median on-time", f"{scored['on_time_share'].median() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top 5 best routes**")
        st.dataframe(scored.head(5), use_container_width=True)
    with right:
        st.markdown("**Top 5 worst routes**")
        st.dataframe(scored.tail(5).sort_values("quality_score"), use_container_width=True)

    st.markdown("**Full leaderboard**")
    st.dataframe(scored, use_container_width=True)


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="Baseline — Starter Kit", layout="wide")
    st.title("Baseline")

    track = st.session_state.get("track", TRACK1_LABEL)
    track1_dir = st.session_state.get(
        "track1_dir",
        str(_KIT_ROOT / "tracks" / "referral-care-coordination" / "data"),
    )
    track2_dir = st.session_state.get(
        "track2_dir",
        str(_KIT_ROOT / "tracks" / "food-security-delivery" / "data"),
    )

    st.caption(f"Track: **{track}**")

    if track == TRACK1_LABEL:
        _render_track1(track1_dir)
    else:
        _render_track2(track2_dir)


if __name__ == "__main__":
    main()
