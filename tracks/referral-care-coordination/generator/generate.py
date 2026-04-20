#!/usr/bin/env python3
"""Track 1 Generator — Inter-Org Referral & Care Coordination.

Synthetic data generator for the BuildersVault Social Services Hackathon.
Grounded in HIFIS 4 + BC Coordinated Access + AIRS taxonomy + VI-SPDAT + PIPA/FOIPPA/OCAP.

Produces:
- Parquet files in ../data/raw/
- SQLite DB in ../data/raw/track1.sqlite with two enrichment views
- Sample CSVs (1000 rows each) in ../data/sample/
- Deterministic output via seed=42

Tables:
- organizations (9)
- clients (800)
- referrals (3,000)
- service_encounters (10,000)
- consent_records (5,000)
- data_sharing_agreements (4)
- duplicate_flags (~500)

Run:
    python generate.py
"""

from __future__ import annotations

import os
import random
import sqlite3
import string
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Constants & seeding
# ---------------------------------------------------------------------------

SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)
fake = Faker("en_CA")
Faker.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
SAMPLE_DIR = SCRIPT_DIR.parent / "data" / "sample"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = RAW_DIR / "track1.sqlite"

# Row counts (spec-locked)
N_ORGS = 9
N_CLIENTS = 800
N_REFERRALS = 3_000
N_ENCOUNTERS = 10_000
N_CONSENTS = 5_000
N_DUP_PAIRS = 40  # seeded near-duplicates (pairs)

# Time windows
TODAY = date(2026, 4, 19)
HORIZON_START = date(2023, 1, 1)  # ~3 years back

# OCAP-eligible nations
NATIONS = ["Songhees", "Esquimalt", "Tsawout", "Pauquachin"]

# AIRS-style taxonomy codes (illustrative; not the full AIRS catalog)
AIRS_CODES = {
    "shelter": "BH-180.1700",
    "outreach": "PH-6200.7000",
    "addictions": "RX-1700",
    "mental_health": "RP-1400",
    "legal_aid": "FT-6000",
    "food_bank": "BD-1800",
    "housing": "BH-180.1500",
    "other": "YZ-0000",
}

# Encounter duration ranges (minutes)
ENCOUNTER_DURATIONS = {
    "shelter_stay": (6 * 60, 14 * 60),
    "outreach_contact": (15, 45),
    "office_visit": (30, 90),
    "food_bank_visit": (10, 40),
    "phone_contact": (5, 20),
    "case_management": (30, 60),
    "intake_assessment": (45, 120),
    "crisis_intervention": (20, 90),
    "housing_search": (30, 75),
    "advocacy": (20, 60),
}

WEEKDAY_MULT = {0: 1.0, 1: 1.2, 2: 1.3, 3: 1.2, 4: 1.1, 5: 0.85, 6: 0.75}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def weighted_choice(choices: list[tuple[Any, float]]) -> Any:
    """Weighted random choice using the seeded rng."""
    values, weights = zip(*choices)
    return rng.choices(values, weights=weights, k=1)[0]


def victoria_postal() -> str:
    """Generate a plausible Victoria-area postal code (V8*/V9*)."""
    letter1 = "V"
    letter2 = rng.choice(["8", "9"])
    letter3 = rng.choice(string.ascii_uppercase)
    digit1 = rng.randint(0, 9)
    letter4 = rng.choice(string.ascii_uppercase)
    digit2 = rng.randint(0, 9)
    return f"{letter1}{letter2}{letter3} {digit1}{letter4}{digit2}"


def rand_date_between(start: date, end: date) -> date:
    """Uniform random date in [start, end]."""
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 0)))


def rand_datetime_between(start: date, end: date) -> datetime:
    """Uniform random timestamp within the date range, weighted by weekday."""
    for _ in range(10):
        d = rand_date_between(start, end)
        mult = WEEKDAY_MULT[d.weekday()]
        # Month-end spike
        if d.day >= 25:
            mult *= 1.3
        if rng.random() < mult / 1.69:  # 1.69 = max(1.3 * 1.3)
            break
    h = rng.randint(7, 22)
    m = rng.randint(0, 59)
    return datetime(d.year, d.month, d.day, h, m)


def add_minutes(ts: datetime, lo: int, hi: int) -> datetime:
    """Add a random minute offset between lo and hi."""
    return ts + timedelta(minutes=rng.randint(lo, hi))


def maybe_null(value: Any, p: float = 0.0) -> Any:
    """Return None with probability p, else the value."""
    return None if rng.random() < p else value


# ---------------------------------------------------------------------------
# SECTION 1: organizations
# ---------------------------------------------------------------------------


def build_organizations() -> pd.DataFrame:
    """Create the 9 seed organizations spread across 2 clusters.

    Distribution: 2 shelter, 1 outreach, 1 addictions, 1 mental_health,
    1 legal_aid, 1 food_bank, 1 housing, 1 other.
    """
    specs = [
        ("shelter", "Inner Harbour Emergency Shelter", "adults", "all", 19, 99, 75),
        ("shelter", "Rock Bay Youth Shelter", "youth", "all", 16, 24, 28),  # youth gate
        ("outreach", "Downtown Outreach Collective", "adults", "all", 19, 99, 0),
        ("addictions", "Pandora Recovery Services", "adults", "all", 19, 99, 40),
        ("mental_health", "Island Mental Health Partners", "all_ages", "all", 12, 99, 60),
        ("legal_aid", "Vancouver Island Legal Aid Clinic", "all_ages", "all", 0, 99, 0),
        ("food_bank", "Mustard Seed Food Bank", "all_ages", "all", 0, 99, 0),
        ("housing", "Pacifica Supportive Housing", "adults", "all", 19, 99, 120),
        ("other", "Our Place Community Hub", "all_ages", "all", 0, 99, 200),
    ]
    rows = []
    for i, (otype, name, pop, gender, amin, amax, cap) in enumerate(specs, 1):
        # 2 clusters; split ~4/5
        cluster_id = 1 if i <= 5 else 2
        occupied = int(cap * rng.uniform(0.7, 1.15)) if cap > 0 else 0  # up to 1.15x
        waitlist_flag = cap > 0 and rng.random() < 0.55
        rows.append(
            {
                "org_id": f"ORG-{i:04d}",
                "org_name": name,
                "org_type": otype,
                "cluster_id": cluster_id,
                "address_street": fake.street_address(),
                "address_city": "Victoria",
                "address_province": "BC",
                "address_postal": victoria_postal(),
                "service_taxonomy_code": AIRS_CODES.get(otype, AIRS_CODES["other"]),
                "target_population": pop,
                "genders_served": gender,
                "age_min": amin,
                "age_max": amax,
                "capacity_total_slots": cap,
                "capacity_occupied_slots": occupied,
                "waitlist_flag": waitlist_flag,
                "waitlist_size": rng.randint(1, 40) if waitlist_flag else 0,
                "waitlist_average_days": rng.randint(3, 120) if waitlist_flag else 0,
                "last_capacity_update": (
                    datetime.combine(TODAY, datetime.min.time())
                    - timedelta(hours=rng.randint(1, 72))
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 2: clients
# ---------------------------------------------------------------------------


def build_clients(orgs_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Create 800 clients with realistic demographic + HIFIS flag distributions.

    Seeds 40 near-duplicate pairs via name typos/address drift/phonetic substitution.
    Ensures OCAP-protected clients correlate with Indigenous identity.
    Enforces the youth-shelter age gate (16-24) when primary_org is ORG-0002.

    Returns: (clients_df, list of near-duplicate client_id pairs).
    """
    clients: list[dict[str, Any]] = []
    org_ids = orgs_df["org_id"].tolist()
    youth_org = "ORG-0002"  # age 16-24 only

    for i in range(1, N_CLIENTS + 1):
        primary_org = rng.choice(org_ids)
        if primary_org == youth_org:
            age = rng.randint(16, 24)
        else:
            age = rng.choices(
                [rng.randint(16, 24), rng.randint(25, 45), rng.randint(46, 65), rng.randint(66, 85)],
                weights=[0.12, 0.45, 0.30, 0.13],
                k=1,
            )[0]
        dob = date(TODAY.year - age, rng.randint(1, 12), rng.randint(1, 28))

        gender = weighted_choice(
            [("male", 0.55), ("female", 0.38), ("non_binary", 0.03), ("two_spirit", 0.02), ("unknown", 0.02)]
        )
        indigenous = weighted_choice(
            [
                ("first_nations", 0.15),
                ("metis", 0.04),
                ("inuit", 0.01),
                ("non_indigenous", 0.65),
                ("unknown", 0.10),
                ("prefer_not_to_answer", 0.05),
            ]
        )
        current_sleep = weighted_choice(
            [
                ("emergency_shelter", 0.30),
                ("unsheltered", 0.20),
                ("couch_surfing", 0.15),
                ("transitional_housing", 0.12),
                ("institution", 0.05),
                ("hotel_motel", 0.03),
                ("housed", 0.10),
                ("unknown", 0.05),
            ]
        )
        chronic = rng.random() < 0.35
        mh = rng.random() < 0.40
        sub = rng.random() < 0.35
        phys = rng.random() < 0.30
        dev = rng.random() < 0.08

        assessment_tool = weighted_choice(
            [
                ("VI_SPDAT", 0.50),
                ("TAY_VI_SPDAT", 0.10),
                ("family", 0.08),
                ("local", 0.15),
                ("none", 0.17),
            ]
        )
        # Score weighted toward 7-13 (mid band = acuity)
        score = rng.choices(
            list(range(0, 18)),
            weights=[2, 2, 3, 3, 4, 5, 6, 9, 10, 10, 10, 9, 8, 6, 5, 4, 3, 1],
            k=1,
        )[0]
        if assessment_tool == "none":
            score = None
            acuity = None
        else:
            if score <= 3:
                acuity = "low"
            elif score <= 7:
                acuity = "moderate"
            elif score <= 11:
                acuity = "high"
            else:
                acuity = "very_high"

        # CA / BNL coupling (constraint 6)
        ca_enrolled = rng.random() < 0.45
        bnl_active = ca_enrolled and rng.random() < 0.6
        ca_priority = rng.choice(["p1", "p2", "p3"]) if ca_enrolled else None
        bnl_status = rng.choice(["active", "inactive", "housed"]) if bnl_active else None

        # OCAP assignment (constraint 12)
        is_indigenous = indigenous in {"first_nations", "metis", "inuit"}
        if is_indigenous:
            ocap_protected = rng.random() < 0.75  # higher rate for Indigenous
        else:
            ocap_protected = False
        # Target overall ~15%: top up with some non-Indigenous protected flags
        if not is_indigenous and rng.random() < 0.02:
            ocap_protected = True

        ocap_nation = rng.choice(NATIONS) if ocap_protected and is_indigenous else None
        ocap_conditions = (
            "No secondary use; annual review; nation data steward approval"
            if ocap_protected
            else None
        )

        first_homeless = rand_date_between(HORIZON_START - timedelta(days=365 * 5), TODAY)
        current_ep_start = rand_date_between(first_homeless, TODAY)
        days_homeless = rng.randint(30, 1095)

        first_name = fake.first_name_male() if gender == "male" else fake.first_name_female()
        last_name = fake.last_name()

        clients.append(
            {
                "client_id": f"CLI-{i:04d}",
                "primary_org_id": primary_org,
                "first_name": first_name,
                "last_name": last_name,
                "middle_name": fake.first_name() if rng.random() < 0.4 else None,
                "aliases": fake.first_name() if rng.random() < 0.15 else None,
                "dob": dob,
                "age": age,
                "gender": gender,
                "indigenous_identity": indigenous,
                "citizenship_status": rng.choices(
                    ["canadian_citizen", "permanent_resident", "refugee_claimant", "other", "unknown"],
                    weights=[0.78, 0.08, 0.04, 0.03, 0.07],
                    k=1,
                )[0],
                "veteran_status": rng.random() < 0.05,
                "primary_language": rng.choices(
                    ["english", "french", "indigenous_language", "other"],
                    weights=[0.88, 0.04, 0.04, 0.04],
                    k=1,
                )[0],
                "current_sleeping_location": current_sleep,
                "housing_status": (
                    "housed" if current_sleep == "housed" else "homeless"
                ),
                "first_homeless_date": first_homeless,
                "current_episode_start_date": current_ep_start,
                "days_homeless_past_3_years": days_homeless,
                "chronic_homeless_flag": chronic,
                "mental_health_flag": mh,
                "substance_use_flag": sub,
                "physical_health_flag": phys,
                "developmental_flag": dev,
                "bnl_active_flag": bnl_active,
                "bnl_status": bnl_status,
                "ca_enrolled_flag": ca_enrolled,
                "ca_priority_level": ca_priority,
                "assessment_tool": assessment_tool,
                "assessment_date": (
                    rand_date_between(HORIZON_START, TODAY) if assessment_tool != "none" else None
                ),
                "assessment_total_score": score,
                "assessment_acuity_level": acuity,
                "last_contact_date": rand_date_between(TODAY - timedelta(days=90), TODAY),
                "ocap_protected": ocap_protected,
                "ocap_governing_nation": ocap_nation,
                "ocap_data_use_conditions": ocap_conditions,
                "consent_coverage_level": rng.choice(["full", "partial", "minimal", "none"]),
                "default_sharing_scope": rng.choice(
                    ["all_dsa_agencies", "limited_agencies", "single_agency_only", "no_sharing"]
                ),
                "current_consent_id": None,  # backfilled after consent generation
            }
        )

    clients_df = pd.DataFrame(clients)

    # Seed 40 near-duplicate pairs (constraint 8)
    dup_pairs: list[tuple[str, str]] = []
    available_idx = list(range(len(clients_df)))
    rng.shuffle(available_idx)
    used: set[int] = set()
    attempts = 0
    while len(dup_pairs) < N_DUP_PAIRS and attempts < 5000:
        attempts += 1
        a = rng.choice(available_idx)
        if a in used:
            continue
        # Append a new duplicate row patterned on client a
        src = clients_df.iloc[a].to_dict()
        new_idx = len(clients_df)
        new_id = f"CLI-{new_idx + 1:04d}"
        dup = src.copy()
        dup["client_id"] = new_id
        # Near-dup perturbations: name typo, case shift, alias swap, phonetic tweak
        mode = rng.choice(["typo", "phonetic", "alias_only", "case"])
        if mode == "typo":
            fn = src["first_name"]
            if len(fn) > 3:
                pos = rng.randint(1, len(fn) - 2)
                dup["first_name"] = fn[:pos] + fn[pos + 1] + fn[pos] + fn[pos + 2 :]
        elif mode == "phonetic":
            mapping = {"ph": "f", "c": "k", "y": "i", "z": "s"}
            ln = src["last_name"].lower()
            for k, v in mapping.items():
                if k in ln:
                    ln = ln.replace(k, v, 1)
                    break
            dup["last_name"] = ln.capitalize()
        elif mode == "alias_only":
            dup["aliases"] = src["first_name"]
            dup["first_name"] = fake.first_name()
        elif mode == "case":
            dup["first_name"] = str(src["first_name"]).upper()
            dup["last_name"] = str(src["last_name"]).lower()
        # Address drift is implicit (addresses aren't on clients directly)
        clients_df = pd.concat([clients_df, pd.DataFrame([dup])], ignore_index=True)
        dup_pairs.append((src["client_id"], new_id))
        used.add(a)

    # Rewrite client_ids to stay sequential
    clients_df = clients_df.reset_index(drop=True)
    clients_df["client_id"] = [f"CLI-{i + 1:04d}" for i in range(len(clients_df))]
    # Remap dup pairs to the rewritten IDs — preserve positional pairs
    # (src rows stayed at their original index; dups were appended in order)
    remapped_pairs: list[tuple[str, str]] = []
    n_orig = N_CLIENTS
    for k, (src_id, _) in enumerate(dup_pairs):
        # src_id was CLI-xxxx; find its row by original order (positions preserved)
        src_pos = int(src_id.split("-")[1]) - 1
        new_pos = n_orig + k
        remapped_pairs.append(
            (clients_df.iloc[src_pos]["client_id"], clients_df.iloc[new_pos]["client_id"])
        )

    return clients_df, remapped_pairs


# ---------------------------------------------------------------------------
# SECTION 3: referrals
# ---------------------------------------------------------------------------


def build_referrals(clients_df: pd.DataFrame, orgs_df: pd.DataFrame) -> pd.DataFrame:
    """Create 3,000 referrals with strict lifecycle timestamp ordering.

    Enforces:
    - submitted < acknowledged < decision < started < completed (constraint 1)
    - status_reason only when declined/cancelled, from closed vocab (constraint 2)
    - declined allowed reasons: capacity_full|client_ineligible|out_of_catchment|
      consent_not_granted|other
    - receiving_org != referring_org
    """
    rows: list[dict[str, Any]] = []
    client_ids = clients_df["client_id"].tolist()
    org_ids = orgs_df["org_id"].tolist()

    decline_reasons = [
        "capacity_full",
        "client_ineligible",
        "out_of_catchment",
        "consent_not_granted",
        "other",
    ]
    cancel_reasons = ["client_withdrew", "duplicate_request", "no_longer_needed", "other"]

    for i in range(1, N_REFERRALS + 1):
        rid = f"REF-{i:05d}"
        client_id = rng.choice(client_ids)
        referring = rng.choice(org_ids)
        receiving = rng.choice([o for o in org_ids if o != referring])

        status = weighted_choice(
            [
                ("submitted", 0.10),
                ("pending", 0.15),
                ("accepted", 0.35),
                ("declined", 0.10),
                ("cancelled", 0.05),
                ("completed", 0.25),
            ]
        )
        priority = weighted_choice([("routine", 0.60), ("urgent", 0.30), ("crisis", 0.10)])

        submitted_at = rand_datetime_between(HORIZON_START, TODAY)
        acknowledged_at = None
        decision_at = None
        started_at = None
        completed_at = None
        cancelled_at = None
        status_reason = None
        related_enc = None

        if status == "submitted":
            pass  # only submitted_at populated
        elif status == "pending":
            acknowledged_at = add_minutes(submitted_at, 30, 60 * 48)
        elif status in ("accepted", "completed"):
            acknowledged_at = add_minutes(submitted_at, 30, 60 * 48)
            decision_at = add_minutes(acknowledged_at, 60, 60 * 72)
            started_at = add_minutes(decision_at, 30, 60 * 24 * 7)
            if status == "completed":
                completed_at = add_minutes(started_at, 60, 60 * 24 * 30)
                # related_service_encounter_id populated post-encounter-gen
                related_enc = "__PLACEHOLDER__"
        elif status == "declined":
            acknowledged_at = add_minutes(submitted_at, 30, 60 * 48)
            decision_at = add_minutes(acknowledged_at, 60, 60 * 72)
            status_reason = rng.choice(decline_reasons)
        elif status == "cancelled":
            acknowledged_at = add_minutes(submitted_at, 30, 60 * 48)
            cancelled_at = add_minutes(acknowledged_at, 30, 60 * 24 * 7)
            status_reason = rng.choice(cancel_reasons)

        rows.append(
            {
                "referral_id": rid,
                "client_id": client_id,
                "referring_org_id": referring,
                "receiving_org_id": receiving,
                "referral_source_module": rng.choice(
                    ["HIFIS_Referral", "HIFIS_Intake", "CA_Module", "External_Portal"]
                ),
                "referral_type": rng.choices(
                    ["internal", "external", "coordinated_access", "self_referral"],
                    weights=[0.30, 0.35, 0.25, 0.10],
                    k=1,
                )[0],
                "referral_reason": rng.choice(
                    [
                        "shelter_bed_needed",
                        "mh_assessment",
                        "detox_request",
                        "housing_support",
                        "food_security",
                        "legal_assistance",
                        "case_management",
                        "crisis_support",
                    ]
                ),
                "referral_priority": priority,
                "status": status,
                "status_reason": status_reason,
                "submitted_at": submitted_at,
                "acknowledged_at": acknowledged_at,
                "decision_at": decision_at,
                "started_at": started_at,
                "completed_at": completed_at,
                "cancelled_at": cancelled_at,
                "consent_record_id": None,  # backfilled later
                "related_service_encounter_id": related_enc,
                "referred_by_worker_name": fake.name(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 4: service_encounters
# ---------------------------------------------------------------------------


def build_encounters(
    clients_df: pd.DataFrame, orgs_df: pd.DataFrame, referrals_df: pd.DataFrame
) -> pd.DataFrame:
    """Create 10,000 service encounters with realistic durations and weekday weighting.

    Enforces:
    - constraint 9: realistic duration per encounter_type
    - constraint 5: encounters linked to completed referrals must start >= decision_at
    - constraint 7: youth shelter encounters (ORG-0002) only for age 16-24 clients
    """
    rows: list[dict[str, Any]] = []
    client_ids = clients_df["client_id"].tolist()
    age_lookup = dict(zip(clients_df["client_id"], clients_df["age"]))
    org_ids = orgs_df["org_id"].tolist()
    youth_org = "ORG-0002"

    completed_refs = referrals_df[referrals_df["status"] == "completed"].copy()
    completed_ref_iter = completed_refs.sample(frac=1, random_state=SEED).to_dict("records")
    ref_to_assign = {r["referral_id"]: r for r in completed_ref_iter}

    encounter_types = list(ENCOUNTER_DURATIONS.keys())

    # First, force an encounter for every completed referral (constraint 5)
    for ref in ref_to_assign.values():
        etype = rng.choice(encounter_types)
        lo, hi = ENCOUNTER_DURATIONS[etype]
        start = ref["decision_at"] + timedelta(minutes=rng.randint(30, 60 * 48))
        end = start + timedelta(minutes=rng.randint(lo, hi))
        # Youth gate
        org_choice = ref["receiving_org_id"]
        if org_choice == youth_org and age_lookup.get(ref["client_id"], 30) > 24:
            org_choice = rng.choice([o for o in org_ids if o != youth_org])
        rows.append(
            _encounter_row(
                len(rows) + 1,
                ref["client_id"],
                org_choice,
                etype,
                start,
                end,
                related_referral_id=ref["referral_id"],
            )
        )

    # Fill remaining with random encounters
    while len(rows) < N_ENCOUNTERS:
        cid = rng.choice(client_ids)
        age = age_lookup.get(cid, 30)
        org_choice = rng.choice(org_ids)
        if org_choice == youth_org and not (16 <= age <= 24):
            org_choice = rng.choice([o for o in org_ids if o != youth_org])  # constraint 7
        etype = rng.choice(encounter_types)
        lo, hi = ENCOUNTER_DURATIONS[etype]
        start = rand_datetime_between(HORIZON_START, TODAY)
        end = start + timedelta(minutes=rng.randint(lo, hi))
        rows.append(
            _encounter_row(len(rows) + 1, cid, org_choice, etype, start, end, related_referral_id=None)
        )

    enc_df = pd.DataFrame(rows)

    # Backfill referrals.related_service_encounter_id for completed refs
    completed_map: dict[str, str] = {}
    for _, enc in enc_df[enc_df["related_referral_id"].notna()].iterrows():
        completed_map[enc["related_referral_id"]] = enc["encounter_id"]
    referrals_df["related_service_encounter_id"] = referrals_df.apply(
        lambda r: completed_map.get(r["referral_id"])
        if r["status"] == "completed"
        else None,
        axis=1,
    )

    return enc_df


def _encounter_row(
    idx: int,
    client_id: str,
    org_id: str,
    etype: str,
    start: datetime,
    end: datetime,
    related_referral_id: str | None,
) -> dict[str, Any]:
    """Build one encounter row with realistic outcome and goods/services."""
    outcome = weighted_choice(
        [
            ("completed", 0.75),
            ("no_show", 0.10),
            ("cancelled_by_client", 0.08),
            ("cancelled_by_provider", 0.07),
        ]
    )
    diversion = rng.random() < 0.08 and etype in {"intake_assessment", "crisis_intervention"}
    return {
        "encounter_id": f"ENC-{idx:05d}",
        "client_id": client_id,
        "org_id": org_id,
        "encounter_type": etype,
        "reason_for_service": rng.choice(
            [
                "housing_need",
                "food_need",
                "mental_health_support",
                "substance_use_support",
                "legal_issue",
                "basic_needs",
                "case_coordination",
                "crisis",
            ]
        ),
        "encounter_start": start,
        "encounter_end": end,
        "location_type": rng.choice(["site", "community", "phone", "home_visit"]),
        "staff_role": rng.choice(
            ["outreach_worker", "case_manager", "intake_worker", "clinician", "peer_support"]
        ),
        "staff_initials": "".join(rng.choices(string.ascii_uppercase, k=2)),
        "related_referral_id": related_referral_id,
        "service_taxonomy_code": rng.choice(list(AIRS_CODES.values())),
        "units_of_service": max(1, int((end - start).total_seconds() // 3600)),
        "outcome_flag": outcome,
        "goods_services_provided": rng.choice(
            ["meal", "bed_night", "hygiene_kit", "transport_voucher", "counselling_session", "none"]
        ),
        "diversion_flag": diversion,
        "diversion_destination": (
            rng.choice(["family_reunification", "self_resolved", "other_shelter", "unknown"])
            if diversion
            else None
        ),
    }


# ---------------------------------------------------------------------------
# SECTION 5: consent_records (9 red-flag patterns)
# ---------------------------------------------------------------------------


def build_consents(
    clients_df: pd.DataFrame, orgs_df: pd.DataFrame
) -> pd.DataFrame:
    """Create 5,000 consent records with 9 seeded governance red-flag patterns.

    Red flags:
    1. 4% active-but-expired (expiry_date past but status=active)
    2. 3% scope mismatch (single_agency_only but client has cross-org encounters)
    3. 15 clients with partial withdrawal not propagated
    4. 2% orphan consents (client_id FK violation)
    5. 8 OCAP clients with sharing_scope=all_dsa_agencies (override ignored)
    6. 10 stale summary rollups (client.current_consent_id points to expired)
    7. 5 youth clients with inherited consent past age 19
    8. implied consent on data_categories containing "mental_health"/"substance_use"
    9. 4% foippa_statutory with no purpose_codes
    """
    rows: list[dict[str, Any]] = []
    client_ids = clients_df["client_id"].tolist()
    org_ids = orgs_df["org_id"].tolist()

    # Target ~5000 consents with per-client distribution
    consent_idx = 1
    client_consents: dict[str, list[int]] = {}

    for cid in client_ids:
        n = weighted_choice([(1, 0.70), (2, 0.20), (3, 0.07), (4, 0.02), (5, 0.01)])
        client_consents[cid] = []
        ocap_row = clients_df[clients_df["client_id"] == cid].iloc[0]
        is_ocap = bool(ocap_row["ocap_protected"])
        age = int(ocap_row["age"])

        for k in range(n):
            if consent_idx > N_CONSENTS:
                break
            cons_id = f"CNS-{consent_idx:05d}"
            consent_idx += 1

            status = weighted_choice(
                [
                    ("active", 0.68),
                    ("expired", 0.08),
                    ("withdrawn", 0.07),
                    ("superseded", 0.15),
                    ("pending", 0.02),
                ]
            )
            given = rand_date_between(HORIZON_START, TODAY)
            effective = given
            expiry = given + timedelta(days=rng.randint(180, 365 * 2))
            withdrawal = None
            superseded = None
            if status == "withdrawn":
                withdrawal = rand_date_between(given, TODAY)
            if status == "superseded":
                superseded = rand_date_between(given, TODAY)
            if status == "expired":
                expiry = rand_date_between(given, TODAY - timedelta(days=1))

            legal_basis = weighted_choice(
                [
                    ("pipa", 0.40),
                    ("foippa", 0.20),
                    ("pipa_and_foippa", 0.25),
                    ("foippa_statutory", 0.10),
                    ("indigenous_governance", 0.05),
                ]
            )
            # Constraint: OCAP client must have indigenous_governance on at least one active
            if is_ocap and k == 0:
                legal_basis = "indigenous_governance"
                status = "active"

            consent_type = weighted_choice(
                [
                    ("explicit", 0.55),
                    ("coordinated_access", 0.15),
                    ("inherited", 0.08),
                    ("declined_anonymous", 0.04),
                    ("implied", 0.15),
                    ("opt_out", 0.03),
                ]
            )

            purposes = rng.sample(
                ["case_mgmt", "referrals", "reporting", "research", "coordinated_access"],
                k=rng.randint(1, 3),
            )
            data_cats = rng.sample(
                [
                    "demographics",
                    "housing_status",
                    "mental_health",
                    "substance_use",
                    "medical",
                    "financial",
                    "family",
                ],
                k=rng.randint(1, 4),
            )
            scope = weighted_choice(
                [
                    ("all_dsa_agencies", 0.45),
                    ("limited_agencies", 0.25),
                    ("single_agency_only", 0.20),
                    ("no_sharing", 0.07),
                    ("anonymous_only", 0.03),
                ]
            )
            scope_agencies = (
                ",".join(rng.sample(org_ids, k=rng.randint(2, 4)))
                if scope == "limited_agencies"
                else None
            )

            rows.append(
                {
                    "consent_id": cons_id,
                    "client_id": cid,
                    "collecting_org_id": rng.choice(org_ids),
                    "dsa_id": rng.randint(1, 4),
                    "consent_type": consent_type,
                    "status": status,
                    "legal_basis": legal_basis,
                    "purpose_codes": ",".join(purposes),
                    "data_categories": ",".join(data_cats),
                    "sharing_scope_type": scope,
                    "sharing_scope_agency_ids": scope_agencies,
                    "given_date": given,
                    "effective_date": effective,
                    "expiry_date": expiry,
                    "withdrawal_date": withdrawal,
                    "superseded_date": superseded,
                    "consent_source": rng.choice(["paper", "digital_form", "verbal_logged", "portal"]),
                    "obtained_by_user_id": f"USR-{rng.randint(1, 40):03d}",
                    "witness_user_id": (
                        f"USR-{rng.randint(1, 40):03d}" if rng.random() < 0.3 else None
                    ),
                    "consent_document_ref": f"DOC-{rng.randint(10000, 99999)}",
                    "notes": None,
                    "_age_at_consent": age,  # internal only; dropped at end
                }
            )
            client_consents[cid].append(len(rows) - 1)
        if consent_idx > N_CONSENTS:
            break

    # If short of N_CONSENTS (unlikely), pad with extra active records
    while len(rows) < N_CONSENTS:
        cid = rng.choice(client_ids)
        cons_id = f"CNS-{len(rows) + 1:05d}"
        rows.append(
            {
                "consent_id": cons_id,
                "client_id": cid,
                "collecting_org_id": rng.choice(org_ids),
                "dsa_id": rng.randint(1, 4),
                "consent_type": "explicit",
                "status": "active",
                "legal_basis": "pipa",
                "purpose_codes": "case_mgmt",
                "data_categories": "demographics",
                "sharing_scope_type": "all_dsa_agencies",
                "sharing_scope_agency_ids": None,
                "given_date": rand_date_between(HORIZON_START, TODAY),
                "effective_date": rand_date_between(HORIZON_START, TODAY),
                "expiry_date": TODAY + timedelta(days=365),
                "withdrawal_date": None,
                "superseded_date": None,
                "consent_source": "digital_form",
                "obtained_by_user_id": f"USR-{rng.randint(1, 40):03d}",
                "witness_user_id": None,
                "consent_document_ref": f"DOC-{rng.randint(10000, 99999)}",
                "notes": None,
                "_age_at_consent": 30,
            }
        )
        client_consents.setdefault(cid, []).append(len(rows) - 1)

    consents = pd.DataFrame(rows[:N_CONSENTS])

    # --- Red flag pattern injection ------------------------------------
    n = len(consents)

    # (1) 4% active expired
    candidates = consents[consents["status"] == "active"].sample(frac=0.04, random_state=SEED).index
    consents.loc[candidates, "expiry_date"] = TODAY - timedelta(days=30)

    # (2) 3% scope mismatch flagged in notes (cross-org encounter will be checked downstream)
    candidates = consents.sample(frac=0.03, random_state=SEED + 1).index
    consents.loc[candidates, "sharing_scope_type"] = "single_agency_only"
    consents.loc[candidates, "notes"] = "RED_FLAG: scope_mismatch_seeded"

    # (3) 15 clients with partial withdrawal not propagated
    pw_clients = rng.sample(client_ids, 15)
    for cid in pw_clients:
        idxs = consents[consents["client_id"] == cid].index.tolist()
        if len(idxs) >= 2:
            consents.loc[idxs[0], "status"] = "withdrawn"
            consents.loc[idxs[0], "withdrawal_date"] = TODAY - timedelta(days=rng.randint(10, 90))
            # Leave the second row still active -> not propagated
            consents.loc[idxs[1], "status"] = "active"
            consents.loc[idxs[1], "notes"] = "RED_FLAG: partial_withdrawal_not_propagated"

    # (4) 2% orphan consents (FK violation)
    orphan_idx = consents.sample(frac=0.02, random_state=SEED + 2).index
    consents.loc[orphan_idx, "client_id"] = [
        f"CLI-{rng.randint(9000, 9999):04d}" for _ in orphan_idx
    ]

    # (5) 8 OCAP clients with sharing_scope=all_dsa_agencies (override ignored)
    ocap_clients = clients_df[clients_df["ocap_protected"]].client_id.tolist()
    if len(ocap_clients) >= 8:
        for cid in rng.sample(ocap_clients, 8):
            idxs = consents[consents["client_id"] == cid].index.tolist()
            if idxs:
                consents.loc[idxs[0], "sharing_scope_type"] = "all_dsa_agencies"
                consents.loc[idxs[0], "notes"] = "RED_FLAG: ocap_override_ignored"

    # (6) 10 stale summary rollups — flagged later when we point clients.current_consent_id
    #     at expired consents (applied below).

    # (7) 5 youth with inherited consent past age 19
    youth_now = clients_df[clients_df["age"] >= 19].client_id.tolist()
    if len(youth_now) >= 5:
        for cid in rng.sample(youth_now, 5):
            idxs = consents[consents["client_id"] == cid].index.tolist()
            if idxs:
                consents.loc[idxs[0], "consent_type"] = "inherited"
                consents.loc[idxs[0], "notes"] = "RED_FLAG: youth_inherited_past_19"

    # (8) implied consent on sensitive data categories
    implied_sensitive_idx = consents[consents["consent_type"] == "implied"].sample(
        frac=0.25, random_state=SEED + 3
    ).index
    consents.loc[implied_sensitive_idx, "data_categories"] = "mental_health,substance_use"
    consents.loc[implied_sensitive_idx, "notes"] = "RED_FLAG: implied_consent_sensitive"

    # (9) 4% foippa_statutory with no purpose_codes
    stat_idx = consents[consents["legal_basis"] == "foippa_statutory"].sample(
        frac=0.40, random_state=SEED + 4
    ).index
    consents.loc[stat_idx, "purpose_codes"] = None
    consents.loc[stat_idx, "notes"] = "RED_FLAG: foippa_statutory_missing_purpose"

    consents = consents.drop(columns=["_age_at_consent"])
    return consents


# ---------------------------------------------------------------------------
# Helpers post-consent
# ---------------------------------------------------------------------------


def assign_current_consent(
    clients_df: pd.DataFrame, consents_df: pd.DataFrame
) -> pd.DataFrame:
    """Point each client.current_consent_id to their most recent active consent,
    but for 10 clients seed a stale rollup (pointing at expired consent, red flag #6).
    """
    # Most recent active consent per client
    active = consents_df[consents_df["status"] == "active"].copy()
    active = active.sort_values("effective_date").drop_duplicates("client_id", keep="last")
    mapping = dict(zip(active["client_id"], active["consent_id"]))

    clients_df["current_consent_id"] = clients_df["client_id"].map(mapping)

    # Seed 10 stale rollups: point at an expired consent
    expired = consents_df[consents_df["status"] == "expired"].copy()
    stale_targets = rng.sample(list(expired.index), min(10, len(expired)))
    for tgt in stale_targets:
        cid = expired.loc[tgt, "client_id"]
        mask = clients_df["client_id"] == cid
        if mask.any():
            clients_df.loc[mask, "current_consent_id"] = expired.loc[tgt, "consent_id"]

    return clients_df


def link_referrals_consents(
    referrals_df: pd.DataFrame, consents_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach a consent_record_id to each referral where possible."""
    consent_by_client = (
        consents_df[consents_df["status"].isin(["active", "superseded"])]
        .groupby("client_id")["consent_id"]
        .apply(list)
        .to_dict()
    )

    def pick(cid: str) -> str | None:
        options = consent_by_client.get(cid, [])
        return rng.choice(options) if options else None

    referrals_df["consent_record_id"] = referrals_df["client_id"].apply(pick)
    return referrals_df


# ---------------------------------------------------------------------------
# SECTION 6: data_sharing_agreements
# ---------------------------------------------------------------------------


def build_dsas() -> pd.DataFrame:
    """Four seed DSAs: HIFIS common, FOIPPA ISA, bilateral, indigenous governance."""
    rows = [
        {
            "dsa_id": 1,
            "dsa_name": "HIFIS Common DSA (BC Cluster)",
            "dsa_type": "hifis_common",
            "governing_statute": "PIPA",
            "effective_date": date(2024, 1, 1),
            "expiry_date": date(2027, 12, 31),
            "signatories": "ORG-0001,ORG-0002,ORG-0003,ORG-0004,ORG-0008,ORG-0009",
        },
        {
            "dsa_id": 2,
            "dsa_name": "FOIPPA Information Sharing Agreement (Public Bodies)",
            "dsa_type": "foippa_isa",
            "governing_statute": "FOIPPA",
            "effective_date": date(2024, 3, 15),
            "expiry_date": date(2029, 3, 14),
            "signatories": "ORG-0005,ORG-0006,ORG-0008",
        },
        {
            "dsa_id": 3,
            "dsa_name": "Bilateral Housing-Addictions Coordination Agreement",
            "dsa_type": "bilateral",
            "governing_statute": "PIPA",
            "effective_date": date(2025, 6, 1),
            "expiry_date": date(2028, 5, 31),
            "signatories": "ORG-0004,ORG-0008",
        },
        {
            "dsa_id": 4,
            "dsa_name": "Indigenous Data Governance Agreement (OCAP-aligned)",
            "dsa_type": "indigenous_governance",
            "governing_statute": "OCAP/Nation-level",
            "effective_date": date(2024, 9, 30),
            "expiry_date": date(2029, 9, 29),
            "signatories": "ORG-0001,ORG-0003,ORG-0008,ORG-0009",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 7: duplicate_flags
# ---------------------------------------------------------------------------


def build_duplicate_flags(
    clients_df: pd.DataFrame, dup_pairs: list[tuple[str, str]], orgs_df: pd.DataFrame
) -> pd.DataFrame:
    """Produce ~500 duplicate flag rows: 300 TP (against seeded pairs; many appear
    multiple times) + 200 FP decoys.
    """
    rows: list[dict[str, Any]] = []
    org_ids = orgs_df["org_id"].tolist()

    reasons_tp = [
        "exact_name_dob_match",
        "phonetic_name_match",
        "phone_only_match",
        "address_drift_match",
        "alias_cross_ref",
    ]
    reasons_fp = [
        "same_first_name_only",
        "shared_address_but_different_person",
        "family_member_overlap",
        "loose_phonetic",
    ]

    # 300 TPs — over-sample from dup_pairs
    for i in range(300):
        pa, pb = rng.choice(dup_pairs)
        score = round(rng.uniform(0.75, 0.99), 2)
        status = weighted_choice(
            [("unreviewed", 0.50), ("confirmed_duplicate", 0.30), ("not_duplicate", 0.15), ("merged", 0.05)]
        )
        rows.append(
            {
                "duplicate_flag_id": f"DUP-{i + 1:04d}",
                "client_id_primary": pa,
                "client_id_secondary": pb,
                "match_score": score,
                "possible_duplicate_reason": rng.choice(reasons_tp),
                "review_status": status,
                "review_decision_date": (
                    rand_date_between(HORIZON_START, TODAY)
                    if status not in {"unreviewed"}
                    else None
                ),
                "reviewer_org_id": rng.choice(org_ids),
            }
        )

    # 200 FPs — random non-dup pairs
    all_ids = clients_df["client_id"].tolist()
    dup_set = set(tuple(sorted(p)) for p in dup_pairs)
    for i in range(300, 500):
        while True:
            a, b = rng.sample(all_ids, 2)
            if tuple(sorted((a, b))) not in dup_set:
                break
        score = round(rng.uniform(0.55, 0.80), 2)
        status = weighted_choice(
            [("unreviewed", 0.50), ("confirmed_duplicate", 0.05), ("not_duplicate", 0.40), ("merged", 0.05)]
        )
        rows.append(
            {
                "duplicate_flag_id": f"DUP-{i + 1:04d}",
                "client_id_primary": a,
                "client_id_secondary": b,
                "match_score": score,
                "possible_duplicate_reason": rng.choice(reasons_fp),
                "review_status": status,
                "review_decision_date": (
                    rand_date_between(HORIZON_START, TODAY)
                    if status not in {"unreviewed"}
                    else None
                ),
                "reviewer_org_id": rng.choice(org_ids),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 8: messiness injection
# ---------------------------------------------------------------------------


def messy_null() -> Any:
    """Apply the null-representation distribution:
    60% true null, 25% empty string, 10% 'Unknown'/'N/A', 5% 999.
    """
    r = rng.random()
    if r < 0.60:
        return None
    if r < 0.85:
        return ""
    if r < 0.95:
        return rng.choice(["Unknown", "N/A"])
    return 999


def messy_phone() -> str:
    """Generate a phone in one of 5 formats with the specified distribution."""
    digits = f"250{rng.randint(2000000, 9999999)}"
    r = rng.random()
    if r < 0.50:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    if r < 0.75:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:]}"
    if r < 0.90:
        return digits
    if r < 0.97:
        return f"{digits[0:3]}.{digits[3:6]}.{digits[6:]}"
    return f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:]}"


def messy_date(d: Any) -> Any:
    """Rewrite a date as a mixed-format string with spec distribution."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return d
    if isinstance(d, pd.Timestamp):
        d = d.date()
    if isinstance(d, datetime):
        d = d.date()
    if not isinstance(d, date):
        return d
    r = rng.random()
    if r < 0.75:
        return d.isoformat()
    if r < 0.87:
        return d.strftime("%m/%d/%Y")
    if r < 0.95:
        return d.strftime("%d/%m/%Y")
    return d.strftime("%B %d, %Y")


def messy_case(s: Any) -> Any:
    """Apply name-case distribution: 70% Title, 15% UPPER, 10% lower, 5% mixed."""
    if not isinstance(s, str) or not s:
        return s
    r = rng.random()
    if r < 0.70:
        return s.title()
    if r < 0.85:
        return s.upper()
    if r < 0.95:
        return s.lower()
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in s)


def inject_messiness(
    clients_df: pd.DataFrame,
    referrals_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    consents_df: pd.DataFrame,
    orgs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply messiness after clean generation:
    - null-representation distribution
    - phone + date + name-case formatting chaos
    - missingness rates (dob 5%, postal 8%, phone 12%, email 30%)
    - 0.8% orphan referrals, 0.3% blank referral_ids
    """
    # --- clients ---
    clients_df["phone"] = [messy_phone() for _ in range(len(clients_df))]
    clients_df["email"] = [fake.email() for _ in range(len(clients_df))]

    # Missingness
    dob_mask = clients_df.sample(frac=0.05, random_state=SEED).index
    clients_df.loc[dob_mask, "dob"] = [messy_null() for _ in dob_mask]

    phone_mask = clients_df.sample(frac=0.12, random_state=SEED + 1).index
    clients_df.loc[phone_mask, "phone"] = [messy_null() for _ in phone_mask]

    email_mask = clients_df.sample(frac=0.30, random_state=SEED + 2).index
    clients_df.loc[email_mask, "email"] = [messy_null() for _ in email_mask]

    # Name case chaos
    clients_df["first_name"] = clients_df["first_name"].apply(messy_case)
    clients_df["last_name"] = clients_df["last_name"].apply(messy_case)

    # Date formatting chaos on select date columns
    for col in ["dob", "first_homeless_date", "current_episode_start_date", "assessment_date", "last_contact_date"]:
        clients_df[col] = clients_df[col].apply(messy_date)

    # --- organizations ---
    postal_mask = orgs_df.sample(frac=0.08, random_state=SEED + 10).index
    orgs_df.loc[postal_mask, "address_postal"] = [messy_null() for _ in postal_mask]

    # --- referrals ---
    # 0.8% orphan referring_org_id
    orphan_idx = referrals_df.sample(frac=0.008, random_state=SEED + 3).index
    referrals_df.loc[orphan_idx, "referring_org_id"] = [
        f"ORG-{rng.randint(9000, 9999):04d}" for _ in orphan_idx
    ]

    # 0.3% blank referral_ids
    blank_idx = referrals_df.sample(frac=0.003, random_state=SEED + 4).index
    referrals_df.loc[blank_idx, "referral_id"] = ""

    # --- encounters: small amount of null injection on reason_for_service ---
    reason_mask = encounters_df.sample(frac=0.05, random_state=SEED + 5).index
    encounters_df.loc[reason_mask, "reason_for_service"] = [messy_null() for _ in reason_mask]

    # --- consents: null injection on witness_user_id and notes ---
    wm = consents_df.sample(frac=0.20, random_state=SEED + 6).index
    consents_df.loc[wm, "witness_user_id"] = [messy_null() for _ in wm]

    return clients_df, referrals_df, encounters_df, consents_df, orgs_df


# ---------------------------------------------------------------------------
# SECTION 9: write CSVs + Parquet + SQLite + views
# ---------------------------------------------------------------------------


def write_outputs(
    orgs_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    referrals_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    consents_df: pd.DataFrame,
    dsas_df: pd.DataFrame,
    dups_df: pd.DataFrame,
) -> None:
    """Write parquet + sqlite + sample CSVs and build the two enrichment views."""
    tables = {
        "organizations": orgs_df,
        "clients": clients_df,
        "referrals": referrals_df,
        "service_encounters": encounters_df,
        "consent_records": consents_df,
        "data_sharing_agreements": dsas_df,
        "duplicate_flags": dups_df,
    }

    # Messiness injection can leave object columns with mixed Python types
    # (str, int sentinels like 999, None). PyArrow rejects mixed-type object
    # columns. Coerce every object column to string (nulls preserved as None).
    def _coerce_objects_to_str(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include="object").columns:
            out[col] = out[col].apply(
                lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v)))
                else (v if isinstance(v, str) else str(v))
            )
        return out

    # Parquet (raw)
    for name, df in tables.items():
        _coerce_objects_to_str(df).to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    # Sample CSVs (first 1000 rows)
    for name, df in tables.items():
        df.head(1000).to_csv(SAMPLE_DIR / f"{name}_sample.csv", index=False)

    # SQLite
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        for name, df in tables.items():
            # sqlite-safe: stringify problematic objects (datetimes, dates already OK)
            df.to_sql(name, conn, if_exists="replace", index=False)

        # current_consent join: use a subquery because we need the active consent row
        conn.executescript(
            """
            DROP VIEW IF EXISTS v_referrals_enriched;
            CREATE VIEW v_referrals_enriched AS
            SELECT
                r.*,
                c.first_name        AS client_first_name,
                c.last_name         AS client_last_name,
                c.age               AS client_age,
                c.indigenous_identity AS client_indigenous_identity,
                c.ocap_protected    AS client_ocap_protected,
                c.current_consent_id AS client_current_consent_id,
                ro.org_name         AS referring_org_name,
                ro.org_type         AS referring_org_type,
                ro.cluster_id       AS referring_cluster_id,
                vo.org_name         AS receiving_org_name,
                vo.org_type         AS receiving_org_type,
                vo.cluster_id       AS receiving_cluster_id,
                cc.status           AS current_consent_status,
                cc.legal_basis      AS current_consent_legal_basis,
                cc.sharing_scope_type AS current_consent_sharing_scope,
                cc.expiry_date      AS current_consent_expiry_date
            FROM referrals r
            LEFT JOIN clients       c  ON c.client_id          = r.client_id
            LEFT JOIN organizations ro ON ro.org_id            = r.referring_org_id
            LEFT JOIN organizations vo ON vo.org_id            = r.receiving_org_id
            LEFT JOIN consent_records cc ON cc.consent_id      = c.current_consent_id;

            DROP VIEW IF EXISTS v_client_timeline;
            CREATE VIEW v_client_timeline AS
            SELECT
                client_id,
                'encounter'        AS event_type,
                encounter_id       AS event_id,
                encounter_start    AS event_ts,
                org_id             AS event_org_id,
                encounter_type     AS event_subtype,
                outcome_flag       AS event_outcome
            FROM service_encounters
            UNION ALL
            SELECT
                client_id,
                'referral'         AS event_type,
                referral_id        AS event_id,
                submitted_at       AS event_ts,
                referring_org_id   AS event_org_id,
                referral_type      AS event_subtype,
                status             AS event_outcome
            FROM referrals
            ORDER BY client_id, event_ts;
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SECTION 10: summary
# ---------------------------------------------------------------------------


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print row counts and null-representation stats across all tables."""
    print("\n" + "=" * 72)
    print("Track 1 Generator — Summary")
    print("=" * 72)
    print(f"{'table':<28}{'rows':>10}{'null_rep_pct':>16}")
    print("-" * 72)

    total_cells = 0
    total_flagged = 0
    for name, df in tables.items():
        rows = len(df)
        flagged = 0
        cells = df.size
        for col in df.columns:
            series = df[col]
            # Count representations of nulls beyond true NaN
            flagged += int(series.isna().sum())
            flagged += int((series.astype(str) == "").sum())
            flagged += int(series.isin(["Unknown", "N/A"]).sum())
            flagged += int((series.astype(str) == "999").sum())
        pct = (flagged / cells * 100) if cells else 0
        total_cells += cells
        total_flagged += flagged
        print(f"{name:<28}{rows:>10,}{pct:>15.2f}%")

    print("-" * 72)
    overall = (total_flagged / total_cells * 100) if total_cells else 0
    print(f"{'TOTAL':<28}{sum(len(df) for df in tables.values()):>10,}{overall:>15.2f}%")
    print("=" * 72)
    print(f"Parquet:  {RAW_DIR}")
    print(f"SQLite:   {SQLITE_PATH}")
    print(f"Samples:  {SAMPLE_DIR}")
    print("Views:    v_referrals_enriched, v_client_timeline")
    print("Seed:     42 (deterministic)\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full generator pipeline end-to-end."""
    print("[1/9] organizations ...")
    orgs_df = build_organizations()

    print("[2/9] clients (+ near-dup seeding + OCAP) ...")
    clients_df, dup_pairs = build_clients(orgs_df)

    print("[3/9] referrals (lifecycle-enforced) ...")
    referrals_df = build_referrals(clients_df, orgs_df)

    print("[4/9] service_encounters (duration + weekday weighting) ...")
    encounters_df = build_encounters(clients_df, orgs_df, referrals_df)

    print("[5/9] consent_records (+ 9 red-flag patterns) ...")
    consents_df = build_consents(clients_df, orgs_df)

    print("[5b/9] link referrals <-> consents, clients.current_consent_id ...")
    referrals_df = link_referrals_consents(referrals_df, consents_df)
    clients_df = assign_current_consent(clients_df, consents_df)

    print("[6/9] data_sharing_agreements ...")
    dsas_df = build_dsas()

    print("[7/9] duplicate_flags (TP + FP decoys) ...")
    dups_df = build_duplicate_flags(clients_df, dup_pairs, orgs_df)

    print("[8/9] messiness injection pass ...")
    clients_df, referrals_df, encounters_df, consents_df, orgs_df = inject_messiness(
        clients_df, referrals_df, encounters_df, consents_df, orgs_df
    )

    print("[9/9] writing parquet + sqlite + CSV samples + views ...")
    write_outputs(
        orgs_df=orgs_df,
        clients_df=clients_df,
        referrals_df=referrals_df,
        encounters_df=encounters_df,
        consents_df=consents_df,
        dsas_df=dsas_df,
        dups_df=dups_df,
    )

    print_summary(
        {
            "organizations": orgs_df,
            "clients": clients_df,
            "referrals": referrals_df,
            "service_encounters": encounters_df,
            "consent_records": consents_df,
            "data_sharing_agreements": dsas_df,
            "duplicate_flags": dups_df,
        }
    )


if __name__ == "__main__":
    main()
