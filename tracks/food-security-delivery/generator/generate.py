#!/usr/bin/env python3
"""Track 2 Generator - Food Security Delivery Operations.

Synthetic data generator for the BuildersVault Social Services Hackathon.
Grounded in Meals on Wheels + grocery-hamper delivery + Vehicle Routing
Problem (Onfleet/Routific schema patterns).

Produces:
- Parquet files in ../data/raw/
- SQLite DB in ../data/raw/track2.sqlite with two enrichment views
- Sample CSVs (first 1000 rows) in ../data/sample/
- Deterministic output via seed=42

Tables:
- depots (2)
- vehicles (8)
- drivers (8)
- clients (500)
- inventory_items (~150)
- delivery_requests (10,000)
- routes (300)
- route_stops (~3,500)
- delivery_request_items (~25,000)

Run:
    python generate.py
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
import string
from datetime import datetime, timedelta, date, time
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

SQLITE_PATH = RAW_DIR / "track2.sqlite"

# Row counts (spec-locked)
N_DEPOTS = 2
N_VEHICLES = 8
N_DRIVERS = 8
N_CLIENTS = 500
N_REQUESTS = 10_000
N_ROUTES = 300
TARGET_REQUEST_ITEMS = 20_000
TARGET_INVENTORY_ITEMS = 150

# Time windows
TODAY = date(2026, 4, 15)  # cutoff per spec
HORIZON_START = TODAY - timedelta(days=21)  # ~3 weeks back for routes/requests
CLIENT_START = TODAY - timedelta(days=365 * 3)  # enrolment lookback
TIME_CUTOFF = datetime.combine(TODAY, time(23, 59))

# Victoria geofence
VIC_LAT_MIN, VIC_LAT_MAX = 48.40, 48.50
VIC_LNG_MIN, VIC_LNG_MAX = -123.45, -123.30

# Languages pool
LANGUAGES = [
    "English",
    "French",
    "Mandarin",
    "Cantonese",
    "Punjabi",
    "Spanish",
    "Vietnamese",
    "Tagalog",
]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def weighted_choice(choices: list[tuple[Any, float]]) -> Any:
    """Weighted random choice using the seeded rng."""
    values, weights = zip(*choices)
    return rng.choices(values, weights=weights, k=1)[0]


def victoria_postal() -> str:
    """Generate a plausible Victoria-area postal code (V8*/V9*)."""
    letter2 = rng.choice(["8", "9"])
    letter3 = rng.choice(string.ascii_uppercase)
    digit1 = rng.randint(0, 9)
    letter4 = rng.choice(string.ascii_uppercase)
    digit2 = rng.randint(0, 9)
    return f"V{letter2}{letter3} {digit1}{letter4}{digit2}"


def rand_date_between(start: date, end: date) -> date:
    """Uniform random date in [start, end]."""
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 0)))


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two points (km)."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def victoria_point() -> tuple[float, float]:
    """Random (lat, lng) inside the Victoria geofence."""
    lat = rng.uniform(VIC_LAT_MIN, VIC_LAT_MAX)
    lng = rng.uniform(VIC_LNG_MIN, VIC_LNG_MAX)
    return round(lat, 6), round(lng, 6)


def time_to_minutes(t: time) -> int:
    """Convert a time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def minutes_to_time(m: int) -> time:
    """Convert minutes-since-midnight to a time."""
    m = max(0, min(m, 23 * 60 + 59))
    return time(m // 60, m % 60)


def combine_dt(d: date, t: time) -> datetime:
    """Combine date and time into a datetime."""
    return datetime.combine(d, t)


def jitter_minutes(ts: datetime, lo: int, hi: int) -> datetime:
    """Shift a timestamp by a random number of minutes in [lo, hi]."""
    return ts + timedelta(minutes=rng.randint(lo, hi))


# ---------------------------------------------------------------------------
# SECTION 1: depots
# ---------------------------------------------------------------------------


def build_depots() -> pd.DataFrame:
    """Create 2 depots: central downtown Victoria + Saanich satellite.

    Capacity chosen so that inventory_conservation constraint (9) will pass
    for typical daily request volumes.
    """
    depots = [
        {
            "depot_id": "DEP-01",
            "name": "Victoria Central Depot",
            "address": "620 Yates Street, Victoria, BC",
            "lat": round(rng.uniform(48.424, 48.432), 6),
            "lng": round(rng.uniform(-123.375, -123.365), 6),
            "capacity_meals_per_day": 450,
            "hours_start": time(6, 0),
            "hours_end": time(18, 0),
        },
        {
            "depot_id": "DEP-02",
            "name": "Saanich Satellite Depot",
            "address": "3980 Quadra Street, Saanich, BC",
            "lat": round(rng.uniform(48.475, 48.485), 6),
            "lng": round(rng.uniform(-123.405, -123.395), 6),
            "capacity_meals_per_day": 250,
            "hours_start": time(7, 0),
            "hours_end": time(17, 0),
        },
    ]
    return pd.DataFrame(depots)


# ---------------------------------------------------------------------------
# SECTION 2: vehicles
# ---------------------------------------------------------------------------


def build_vehicles() -> pd.DataFrame:
    """Create the 8-vehicle fleet: 3 car, 2 van, 2 cargo_van, 1 bike.

    Refrigeration: vans and cargo vans only.
    Wheelchair lift: exactly 1 cargo_van (VEH-06).
    Fuel: gasoline / hybrid / electric mix; bike = N/A.
    """
    specs = [
        # (id, type, capacity_meals, capacity_weight_kg, refrigerated, wheelchair_lift, fuel)
        # 5 of 8 vehicles refrigerated (cars w/ insulated-cooler kits count as
        # refrigerated for cold-chain purposes) so supply roughly matches the
        # ~70% cold-chain demand from MOW_hot + MOW_frozen programs.
        ("VEH-01", "car", 40, 150.0, True, False, "hybrid"),
        ("VEH-02", "car", 40, 150.0, False, False, "hybrid"),
        ("VEH-03", "car", 40, 150.0, False, False, "electric"),
        ("VEH-04", "van", 90, 450.0, True, False, "gasoline"),
        ("VEH-05", "van", 90, 450.0, True, False, "hybrid"),
        ("VEH-06", "cargo_van", 140, 800.0, True, True, "gasoline"),
        ("VEH-07", "cargo_van", 140, 800.0, True, False, "electric"),
        ("VEH-08", "bike", 15, 25.0, False, False, "N/A"),
    ]
    rows: list[dict[str, Any]] = []
    for vid, vtype, cm, cw, refrig, wc, fuel in specs:
        start_hour = rng.randint(6, 8)
        end_hour = rng.randint(14, 18)
        rows.append(
            {
                "vehicle_id": vid,
                "type": vtype,
                "capacity_meals": cm,
                "capacity_weight_kg": cw,
                "refrigerated": refrig,
                "wheelchair_lift": wc,
                "availability_start": time(start_hour, 0),
                "availability_end": time(end_hour, 0),
                "fuel_type": fuel,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 3: drivers
# ---------------------------------------------------------------------------


def build_drivers(vehicles_df: pd.DataFrame, depots_df: pd.DataFrame) -> pd.DataFrame:
    """Create 8 drivers: 6 volunteers + 2 staff.

    Skill/language distribution chosen so that compatibility constraints
    (wheelchair, dog/allergy, interpreter) can be satisfied for the
    majority of clients.

    Vehicle assignments: each driver pinned to one vehicle; driver of
    the cargo van with wheelchair lift (VEH-06) has can_handle_wheelchair
    guaranteed True.
    """
    roles = (["staff"] * 2) + (["volunteer"] * 6)
    rng.shuffle(roles)

    rows: list[dict[str, Any]] = []
    vehicle_ids = vehicles_df["vehicle_id"].tolist()
    depot_ids = depots_df["depot_id"].tolist()

    # Pre-pick which driver gets VEH-06 (wheelchair lift) so we can force their flags
    veh_06_driver_idx = rng.randint(0, N_DRIVERS - 1)
    # Assign vehicles 1:1 (shuffled)
    shuffled_vehicles = vehicle_ids.copy()
    rng.shuffle(shuffled_vehicles)
    # Place VEH-06 at the chosen driver index
    if "VEH-06" in shuffled_vehicles:
        current_pos = shuffled_vehicles.index("VEH-06")
        shuffled_vehicles[current_pos], shuffled_vehicles[veh_06_driver_idx] = (
            shuffled_vehicles[veh_06_driver_idx],
            shuffled_vehicles[current_pos],
        )

    for i in range(1, N_DRIVERS + 1):
        role = roles[i - 1]
        vehicle_id = shuffled_vehicles[i - 1]
        veh_row = vehicles_df[vehicles_df["vehicle_id"] == vehicle_id].iloc[0]
        license_class = "Class_4" if veh_row["type"] in {"van", "cargo_van"} else "Class_5"

        # Shift timing: staff 08:00-16:00 solid; volunteers a bit more varied
        if role == "staff":
            shift_start = time(8, 0)
            shift_end = time(16, 0)
            max_hours = 8
        else:
            shift_start = time(rng.choice([8, 9]), 0)
            shift_end = time(rng.choice([13, 14, 15, 16]), 0)
            max_hours = rng.randint(4, 6)

        # Skills
        can_wheelchair = (vehicle_id == "VEH-06") or (rng.random() < 0.35)
        can_stairs = rng.random() < 0.75
        can_enter_homes = rng.random() < 0.85
        pet_allergy = rng.random() < 0.12

        # Languages: everyone has English; some pick up extras
        extras = rng.sample(LANGUAGES[1:], k=rng.choice([0, 0, 1, 1, 2]))
        language_skills = ";".join(["English"] + extras)

        food_safety = True if role == "staff" else rng.random() < 0.70

        # Performance stats
        on_time_rate = round(rng.uniform(0.82, 0.95), 3)
        no_answer_rate = round(rng.uniform(0.05, 0.15), 3)
        avg_service = round(rng.uniform(6.0, 12.0), 1)
        total_shifts = rng.randint(25, 180) if role == "volunteer" else rng.randint(150, 400)

        first = fake.first_name()
        last = fake.last_name()
        email_local = f"{first.lower()}.{last.lower()}".replace("'", "")
        phone = f"(250) 555-{rng.randint(1000, 9999)}"

        rows.append(
            {
                "driver_id": f"DRV-{i:02d}",
                "first_name": first,
                "last_name": last,
                "role_type": role,
                "phone": phone,
                "email": f"{email_local}@mealswheels.example.org",
                "license_class": license_class,
                "background_check_date": rand_date_between(
                    TODAY - timedelta(days=730), TODAY - timedelta(days=30)
                ),
                "home_base_depot_id": rng.choice(depot_ids),
                "shift_start": shift_start,
                "shift_end": shift_end,
                "max_hours": max_hours,
                "max_stops": rng.randint(25, 40),
                "max_distance_km": rng.randint(60, 120),
                "vehicle_id": vehicle_id,
                "can_handle_wheelchair": can_wheelchair,
                "can_climb_stairs": can_stairs,
                "can_enter_private_homes": can_enter_homes,
                "pet_allergy_flag": pet_allergy,
                "language_skills": language_skills,
                "food_safety_trained": food_safety,
                "total_shifts_completed": total_shifts,
                "on_time_rate": on_time_rate,  # will be recomputed post-stops
                "no_answer_rate": no_answer_rate,
                "average_service_duration": avg_service,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 4: clients
# ---------------------------------------------------------------------------


def nearest_depot(lat: float, lng: float, depots_df: pd.DataFrame) -> str:
    """Pick the closest depot by haversine distance."""
    best_id = None
    best_d = float("inf")
    for _, row in depots_df.iterrows():
        d = haversine_km(lat, lng, row["lat"], row["lng"])
        if d < best_d:
            best_d = d
            best_id = row["depot_id"]
    return best_id


def build_clients(depots_df: pd.DataFrame) -> pd.DataFrame:
    """Create 500 clients with Victoria geofence + realistic mobility/diet/allergy mix.

    Key correlations:
    - interpreter_required correlated with language_primary != English
    - requires_two_person_team correlated with mobility_bedbound
    - has_dog_on_premises drives dog_notes population
    - enrolment_status=deceased/closed gets closure_date (for constraint 12)
    """
    rows: list[dict[str, Any]] = []

    for i in range(1, N_CLIENTS + 1):
        lat, lng = victoria_point()
        postal = victoria_postal()
        home_depot_id = nearest_depot(lat, lng, depots_df)

        first = fake.first_name()
        last = fake.last_name()

        enrolment_status = weighted_choice(
            [
                ("active", 0.75),
                ("paused", 0.10),
                ("closed", 0.10),
                ("deceased", 0.05),
            ]
        )
        enrolment_start = rand_date_between(CLIENT_START, TODAY - timedelta(days=30))
        closure_date = None
        if enrolment_status in {"closed", "deceased"}:
            closure_date = rand_date_between(enrolment_start, TODAY - timedelta(days=3))

        food_security_level = weighted_choice(
            [
                ("secure", 0.10),
                ("marginal", 0.25),
                ("moderate", 0.35),
                ("severe", 0.30),
            ]
        )
        indigenous_flag = rng.random() < 0.10
        building_type = weighted_choice(
            [
                ("house", 0.40),
                ("low_rise", 0.20),
                ("high_rise", 0.15),
                ("assisted_living", 0.15),
                ("SRO", 0.10),
            ]
        )

        # Mobility flags (independent)
        mobility_wheelchair = rng.random() < 0.18
        mobility_walker = rng.random() < 0.22
        mobility_cane = rng.random() < 0.15
        mobility_bedbound = rng.random() < 0.03
        mobility_requires_assistance_at_door = rng.random() < 0.30
        mobility_low_vision = rng.random() < 0.12
        mobility_hard_of_hearing = rng.random() < 0.10

        # Diet flags (independent)
        diet_diabetic = rng.random() < 0.25
        diet_low_sodium = rng.random() < 0.18
        diet_heart_healthy = rng.random() < 0.15
        diet_renal = rng.random() < 0.04
        diet_pureed = rng.random() < 0.06
        diet_minced = rng.random() < 0.08
        diet_soft = rng.random() < 0.12
        diet_gluten_free = rng.random() < 0.06
        diet_vegetarian = rng.random() < 0.09
        diet_vegan = rng.random() < 0.03
        diet_halal = rng.random() < 0.05
        diet_kosher = rng.random() < 0.01
        diet_lactose_free = rng.random() < 0.07

        # Allergen severities: per-allergen independent 80/15/5 none/mild/severe
        def allergen_severity() -> str:
            r = rng.random()
            if r < 0.80:
                return "none"
            if r < 0.95:
                return "mild"
            return "severe"

        allergy_peanut_severity = allergen_severity()
        allergy_tree_nut_severity = allergen_severity()
        allergy_shellfish_severity = allergen_severity()
        allergy_fish_severity = allergen_severity()
        allergy_egg_severity = allergen_severity()
        allergy_soy_severity = allergen_severity()
        allergy_wheat_severity = allergen_severity()
        allergy_dairy_severity = allergen_severity()

        # Language (correlates with interpreter_required)
        language_primary = weighted_choice(
            [
                ("English", 0.78),
                ("Mandarin", 0.06),
                ("Cantonese", 0.04),
                ("Punjabi", 0.04),
                ("Vietnamese", 0.03),
                ("Spanish", 0.02),
                ("Tagalog", 0.02),
                ("French", 0.01),
            ]
        )
        # Target ~12% interpreter_required overall, heavily weighted toward non-English
        if language_primary != "English":
            interpreter_required = rng.random() < 0.50
        else:
            interpreter_required = rng.random() < 0.02

        has_dog = rng.random() < 0.18
        if has_dog:
            dog_breed = rng.choice(
                ["Golden Retriever", "Labrador", "Chihuahua", "German Shepherd", "Poodle mix", "Small terrier", "Pug", "Boxer"]
            )
            dog_temperament = rng.choice(["friendly", "shy", "barks at strangers", "guard dog", "calm indoors"])
            dog_notes = f"{dog_breed}, {dog_temperament}"
        else:
            dog_notes = None

        safe_to_leave_unattended = rng.random() < 0.65
        client_trust_level = weighted_choice(
            [("new", 0.20), ("established", 0.50), ("trusted", 0.30)]
        )
        # Two-person team correlates with bedbound
        if mobility_bedbound:
            requires_two_person_team = rng.random() < 0.60
        else:
            requires_two_person_team = rng.random() < 0.05
        do_not_enter_home = rng.random() < 0.10

        # Buzzer code: applicable mostly to low_rise/high_rise/SRO/assisted_living
        if building_type in {"low_rise", "high_rise", "SRO", "assisted_living"}:
            buzzer_code = f"#{rng.randint(100, 9999)}"
        else:
            buzzer_code = None

        unit_number = None
        if building_type in {"low_rise", "high_rise", "SRO", "assisted_living"}:
            unit_number = f"{rng.randint(101, 2499)}"

        phone = f"(250) 555-{rng.randint(1000, 9999)}"
        email = fake.email() if rng.random() < 0.55 else None

        rows.append(
            {
                "client_id": f"CLI-{i:04d}",
                "first_name": first,
                "last_name": last,
                "dob": rand_date_between(date(1930, 1, 1), date(1970, 12, 31)),
                "phone": phone,
                "email": email,
                "address_street": fake.street_address(),
                "unit_number": unit_number,
                "buzzer_code": buzzer_code,
                "address_city": "Victoria",
                "address_province": "BC",
                "address_postal": postal,
                "lat": lat,
                "lng": lng,
                "home_depot_id": home_depot_id,
                "building_type": building_type,
                "enrolment_status": enrolment_status,
                "enrolment_start_date": enrolment_start,
                "closure_date": closure_date,
                "food_security_level": food_security_level,
                "indigenous_identity_flag": indigenous_flag,
                "language_primary": language_primary,
                "interpreter_required": interpreter_required,
                "mobility_wheelchair": mobility_wheelchair,
                "mobility_walker": mobility_walker,
                "mobility_cane": mobility_cane,
                "mobility_bedbound": mobility_bedbound,
                "mobility_requires_assistance_at_door": mobility_requires_assistance_at_door,
                "mobility_low_vision": mobility_low_vision,
                "mobility_hard_of_hearing": mobility_hard_of_hearing,
                "diet_diabetic": diet_diabetic,
                "diet_low_sodium": diet_low_sodium,
                "diet_heart_healthy": diet_heart_healthy,
                "diet_renal": diet_renal,
                "diet_pureed": diet_pureed,
                "diet_minced": diet_minced,
                "diet_soft": diet_soft,
                "diet_gluten_free": diet_gluten_free,
                "diet_vegetarian": diet_vegetarian,
                "diet_vegan": diet_vegan,
                "diet_halal": diet_halal,
                "diet_kosher": diet_kosher,
                "diet_lactose_free": diet_lactose_free,
                "allergy_peanut_severity": allergy_peanut_severity,
                "allergy_tree_nut_severity": allergy_tree_nut_severity,
                "allergy_shellfish_severity": allergy_shellfish_severity,
                "allergy_fish_severity": allergy_fish_severity,
                "allergy_egg_severity": allergy_egg_severity,
                "allergy_soy_severity": allergy_soy_severity,
                "allergy_wheat_severity": allergy_wheat_severity,
                "allergy_dairy_severity": allergy_dairy_severity,
                "has_dog_on_premises": has_dog,
                "dog_notes": dog_notes,
                "safe_to_leave_unattended": safe_to_leave_unattended,
                "client_trust_level": client_trust_level,
                "requires_two_person_team": requires_two_person_team,
                "do_not_enter_home": do_not_enter_home,
                "emergency_contact_name": fake.name(),
                "emergency_contact_phone": f"(250) 555-{rng.randint(1000, 9999)}",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 5: inventory_items catalogue
# ---------------------------------------------------------------------------


def build_inventory_items() -> pd.DataFrame:
    """Realistic catalogue of ~150 items across 7 categories.

    Distribution: 25 hot meals, 20 frozen meals, 30 produce, 25 dry goods,
    15 dairy, 20 protein, 15 hamper components.

    Allergen and dietary tags are assigned from plausible per-category
    pools so that clients with severe allergens can be matched safely.
    """
    rows: list[dict[str, Any]] = []

    hot_meals = [
        "Roast Chicken & Vegetables", "Beef Stew", "Shepherd's Pie", "Turkey Dinner",
        "Lasagna", "Chicken Pot Pie", "Meatloaf & Mash", "Fish & Chips",
        "Salmon & Rice", "Vegetable Curry", "Tofu Stir-Fry", "Pasta Primavera",
        "Chicken Soup & Roll", "Beef Chili", "Pork Roast Dinner", "Chicken Alfredo",
        "Baked Ziti", "Cabbage Rolls", "Butter Chicken", "Beef Stroganoff",
        "Macaroni & Cheese", "Chicken Tikka", "Vegetable Stew", "Cod & Potatoes",
        "Sausage & Beans",
    ]
    frozen_meals = [
        "Frozen Lasagna Portion", "Frozen Chicken Dinner", "Frozen Beef Stew",
        "Frozen Vegetable Curry", "Frozen Meatloaf", "Frozen Cottage Pie",
        "Frozen Turkey Dinner", "Frozen Fish Casserole", "Frozen Mac & Cheese",
        "Frozen Chili", "Frozen Pasta Primavera", "Frozen Stroganoff",
        "Frozen Chicken Pot Pie", "Frozen Baked Ziti", "Frozen Butter Chicken",
        "Frozen Salmon Portion", "Frozen Tofu Stir-Fry", "Frozen Cabbage Rolls",
        "Frozen Pork Roast", "Frozen Vegetable Medley",
    ]
    produce = [
        "Apples (bag)", "Bananas (bunch)", "Oranges (bag)", "Pears (bag)",
        "Carrots (bag)", "Potatoes (bag)", "Onions (bag)", "Broccoli head",
        "Lettuce head", "Tomatoes (pack)", "Cucumbers", "Bell Peppers",
        "Celery bunch", "Spinach bag", "Mushrooms pack", "Sweet Potatoes",
        "Zucchini", "Cauliflower head", "Green Beans pack", "Blueberries pack",
        "Strawberries pack", "Grapes bag", "Kiwi pack", "Mango",
        "Avocado", "Lemons pack", "Limes pack", "Corn cobs",
        "Peaches pack", "Plums pack",
    ]
    dry_goods = [
        "White Rice 1kg", "Brown Rice 1kg", "Rolled Oats 1kg", "Pasta 500g",
        "Gluten-Free Pasta 500g", "Canned Tomatoes", "Canned Black Beans",
        "Canned Chickpeas", "Canned Corn", "Canned Tuna",
        "Canned Salmon", "Peanut Butter 500g", "Almond Butter 500g",
        "Whole Wheat Flour 1kg", "White Flour 1kg", "Sugar 1kg",
        "Salt 500g", "Honey 500g", "Maple Syrup 250ml", "Crackers box",
        "Cereal Box", "Granola Bars pack", "Bread Loaf", "Gluten-Free Bread",
        "Tortillas pack",
    ]
    dairy = [
        "2% Milk 1L", "Skim Milk 1L", "Lactose-Free Milk 1L", "Soy Milk 1L",
        "Almond Milk 1L", "Yogurt 500g", "Greek Yogurt 500g",
        "Cheddar Cheese 250g", "Mozzarella 250g", "Cream Cheese 250g",
        "Butter 250g", "Margarine 250g", "Cottage Cheese 500g",
        "Sour Cream 250ml", "Kefir 500ml",
    ]
    protein = [
        "Chicken Breast 500g", "Chicken Thigh 500g", "Ground Beef 500g",
        "Pork Chops 500g", "Salmon Fillet 300g", "Cod Fillet 300g",
        "Tuna Steak 300g", "Turkey Breast 500g", "Ground Turkey 500g",
        "Pork Tenderloin 500g", "Beef Steak 300g", "Tofu block 500g",
        "Tempeh 300g", "Eggs dozen", "Halal Chicken 500g",
        "Kosher Beef 500g", "Lentils 500g", "Black Beans 500g dry",
        "Edamame 300g frozen", "Chicken Sausages pack",
    ]
    hamper_components = [
        "Breakfast Hamper Box", "Emergency Food Box Standard",
        "Emergency Food Box Diabetic", "Emergency Food Box Halal",
        "Emergency Food Box Kosher", "Indigenous Foods Hamper",
        "Culturally-Specific Hamper South Asian", "Culturally-Specific Hamper East Asian",
        "Senior Nutrition Box", "Produce Starter Box",
        "Pantry Staples Box", "Protein Starter Box",
        "Weekend Survival Hamper", "Holiday Food Hamper",
        "Allergen-Free Hamper",
    ]

    def add_items(
        names: list[str], category: str, cold_chain: bool,
        default_allergens: list[str], default_diets: list[str],
        unit_default: str, cost_range: tuple[float, float],
    ) -> None:
        for name in names:
            # Allergen assignment: mostly inherits category defaults, occasional extras
            allergens = list(default_allergens)
            if "peanut" in name.lower():
                allergens.append("peanut")
            if "almond" in name.lower() or "tree" in name.lower():
                allergens.append("tree_nut")
            if "salmon" in name.lower() or "cod" in name.lower() or "tuna" in name.lower():
                allergens.append("fish")
            if "shrimp" in name.lower() or "shellfish" in name.lower():
                allergens.append("shellfish")
            if "egg" in name.lower():
                allergens.append("egg")
            if "soy" in name.lower() or "tofu" in name.lower() or "tempeh" in name.lower() or "edamame" in name.lower():
                allergens.append("soy")
            if "wheat" in name.lower() or "bread" in name.lower() or "pasta" in name.lower() or "flour" in name.lower() or "cracker" in name.lower() or "cereal" in name.lower() or "tortilla" in name.lower():
                if "gluten-free" not in name.lower():
                    allergens.append("wheat")
            if any(d in name.lower() for d in ["milk", "yogurt", "cheese", "butter", "cream", "kefir"]) and "lactose-free" not in name.lower() and "soy milk" not in name.lower() and "almond milk" not in name.lower():
                allergens.append("dairy")
            allergens = sorted(set(allergens))

            diets = list(default_diets)
            low = name.lower()
            if "gluten-free" in low:
                diets.append("gluten_free")
            if "lactose-free" in low:
                diets.append("lactose_free")
            if "halal" in low:
                diets.append("halal")
            if "kosher" in low:
                diets.append("kosher")
            if "diabetic" in low:
                diets.append("diabetic_friendly")
            if category in {"produce"}:
                diets.extend(["vegetarian", "vegan", "heart_healthy"])
            if name in {"Tofu block 500g", "Tempeh 300g", "Lentils 500g", "Black Beans 500g dry", "Edamame 300g frozen"}:
                diets.extend(["vegetarian", "vegan"])
            diets = sorted(set(diets))

            unit = unit_default
            if "bag" in low or "pack" in low or "box" in low or "bunch" in low or "dozen" in low:
                unit = "each"
            if "1kg" in low or "500g" in low or "300g" in low or "250g" in low:
                unit = "kg"
            if "1l" in low.replace(" ", "") or "500ml" in low.replace(" ", "") or "250ml" in low.replace(" ", ""):
                unit = "L"

            cost = round(rng.uniform(*cost_range), 2)
            rows.append(
                {
                    "name": name,
                    "category": category,
                    "unit": unit,
                    "allergen_flags": ";".join(allergens),
                    "dietary_tags": ";".join(diets),
                    "cold_chain_required": cold_chain,
                    "standard_cost": cost,
                }
            )

    add_items(hot_meals, "hot_meal", True, [], [], "each", (4.0, 9.0))
    add_items(frozen_meals, "frozen_meal", True, [], [], "each", (3.5, 8.0))
    add_items(produce, "produce", False, [], [], "each", (1.5, 6.0))
    add_items(dry_goods, "dry_goods", False, [], [], "each", (2.0, 7.0))
    add_items(dairy, "dairy", True, [], [], "L", (2.5, 7.0))
    add_items(protein, "protein", True, [], [], "kg", (4.0, 12.0))
    add_items(hamper_components, "hamper_component", False, [], [], "box", (10.0, 25.0))

    # Assign IDs
    for idx, r in enumerate(rows, 1):
        r["item_id"] = f"ITM-{idx:04d}"

    df = pd.DataFrame(rows)
    cols = [
        "item_id", "name", "category", "unit", "allergen_flags", "dietary_tags",
        "cold_chain_required", "standard_cost",
    ]
    return df[cols]


# ---------------------------------------------------------------------------
# SECTION 6: delivery_requests
# ---------------------------------------------------------------------------


def build_delivery_requests(
    clients_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create 2,000 delivery_requests spread across the 3-week horizon.

    Enforces:
    - Constraint 11: all timestamps <= TODAY (cutoff 2026-04-15)
    - Constraint 12: no requests for deceased/closed clients AFTER closure_date
      (seeded violations applied later)
    - dietary_tags_snapshot and mobility_tags_snapshot copied from client
    """
    rows: list[dict[str, Any]] = []
    client_ids = clients_df["client_id"].tolist()
    client_lookup = clients_df.set_index("client_id").to_dict("index")

    def client_tags(c: dict[str, Any]) -> tuple[str, str]:
        dietary = []
        for field, tag in [
            ("diet_diabetic", "diabetic_friendly"),
            ("diet_low_sodium", "low_sodium"),
            ("diet_heart_healthy", "heart_healthy"),
            ("diet_renal", "renal"),
            ("diet_pureed", "pureed"),
            ("diet_minced", "minced"),
            ("diet_soft", "soft"),
            ("diet_gluten_free", "gluten_free"),
            ("diet_vegetarian", "vegetarian"),
            ("diet_vegan", "vegan"),
            ("diet_halal", "halal"),
            ("diet_kosher", "kosher"),
            ("diet_lactose_free", "lactose_free"),
        ]:
            if c.get(field):
                dietary.append(tag)
        mobility = []
        for field, tag in [
            ("mobility_wheelchair", "wheelchair"),
            ("mobility_walker", "walker"),
            ("mobility_cane", "cane"),
            ("mobility_bedbound", "bedbound"),
            ("mobility_requires_assistance_at_door", "assistance_at_door"),
            ("mobility_low_vision", "low_vision"),
            ("mobility_hard_of_hearing", "hard_of_hearing"),
        ]:
            if c.get(field):
                mobility.append(tag)
        return ";".join(dietary), ";".join(mobility)

    for i in range(1, N_REQUESTS + 1):
        # Prefer active clients but allow some paused for realism
        cid = rng.choice(client_ids)
        c = client_lookup[cid]

        # Skip deceased/closed after closure_date at build-time; we will
        # re-inject 4 violations after the fact
        scheduled = rand_date_between(HORIZON_START, TODAY)
        if c["enrolment_status"] in {"deceased", "closed"} and c["closure_date"] is not None:
            if isinstance(c["closure_date"], date) and scheduled > c["closure_date"]:
                scheduled = c["closure_date"] - timedelta(days=rng.randint(1, 10))
                if scheduled < HORIZON_START:
                    scheduled = HORIZON_START

        created_at = combine_dt(
            scheduled - timedelta(days=rng.randint(1, 5)),
            time(rng.randint(6, 20), rng.randint(0, 59)),
        )
        if created_at > TIME_CUTOFF:
            created_at = TIME_CUTOFF - timedelta(hours=rng.randint(1, 48))

        # Time window: start in [08:00, 13:00], width 1-3h
        start_hour = rng.randint(8, 13)
        start_minute = rng.choice([0, 15, 30, 45])
        window_width_min = rng.choice([60, 90, 120, 180])
        tws = time(start_hour, start_minute)
        tws_min = time_to_minutes(tws) + window_width_min
        twe = minutes_to_time(min(tws_min, 19 * 60))

        service_duration = int(np.clip(np.random.normal(8, 2.5), 5, 15))
        quantity_meals = rng.randint(1, 7)
        quantity_boxes = rng.choices([0, 1, 2], weights=[0.55, 0.35, 0.10], k=1)[0]

        program_type = weighted_choice(
            [
                ("MOW_hot", 0.40),
                ("MOW_frozen", 0.25),
                ("grocery_hamper", 0.20),
                ("culturally_specific", 0.08),
                ("emergency_box", 0.07),
            ]
        )
        # Cold chain is implied by hot/frozen MOW and by some hamper types
        cold_chain_required = program_type in {"MOW_hot", "MOW_frozen"} or (
            program_type == "grocery_hamper" and rng.random() < 0.30
        )

        priority_level = weighted_choice([("routine", 0.85), ("urgent", 0.15)])
        frequency = weighted_choice(
            [("one_time", 0.30), ("weekly", 0.40), ("biweekly", 0.15), ("monthly", 0.15)]
        )
        funding_stream = weighted_choice(
            [
                ("Island_Health", 0.30),
                ("United_Way", 0.20),
                ("municipal", 0.15),
                ("private_donations", 0.25),
                ("client_pay", 0.10),
            ]
        )
        status = weighted_choice(
            [
                ("pending", 0.05),
                ("scheduled", 0.10),
                ("in_progress", 0.05),
                ("completed", 0.70),
                ("no_answer", 0.05),
                ("cancelled", 0.03),
                ("rerouted", 0.02),
            ]
        )
        status_reason = None
        if status == "cancelled":
            status_reason = rng.choice(
                ["client_request", "duplicate_request", "no_longer_needed", "weather", "other"]
            )
        elif status == "no_answer":
            status_reason = rng.choice(["no_answer_at_door", "buzzer_failed", "building_locked"])
        elif status == "rerouted":
            status_reason = rng.choice(["vehicle_breakdown", "driver_capacity", "time_window_conflict"])

        # Required driver skills derived from client
        skills: list[str] = []
        if c.get("mobility_wheelchair"):
            skills.append("wheelchair")
        if c.get("has_dog_on_premises"):
            skills.append("no_pet_allergy")
        if c.get("interpreter_required"):
            skills.append(f"lang_{c.get('language_primary', 'English')}")
        if c.get("requires_two_person_team"):
            skills.append("two_person")
        if cold_chain_required:
            skills.append("refrigerated_vehicle")
        required_driver_skills = ";".join(sorted(set(skills)))

        flex_opts = []
        if rng.random() < 0.35:
            flex_opts.append("flexible_window")
        if rng.random() < 0.20:
            flex_opts.append("flexible_day")
        if rng.random() < 0.25 and c.get("safe_to_leave_unattended"):
            flex_opts.append("can_leave_at_door")
        flexibility_flags = ";".join(flex_opts)

        dietary_snap, mobility_snap = client_tags(c)

        rows.append(
            {
                "request_id": f"REQ-{i:05d}",
                "client_id": cid,
                "created_at": created_at,
                "scheduled_date": scheduled,
                "time_window_start": tws,
                "time_window_end": twe,
                "service_duration_minutes": service_duration,
                "quantity_meals": quantity_meals,
                "quantity_boxes": quantity_boxes,
                "cold_chain_required": cold_chain_required,
                "priority_level": priority_level,
                "frequency": frequency,
                "program_type": program_type,
                "funding_stream": funding_stream,
                "status": status,
                "status_reason": status_reason,
                "assigned_route_id": None,       # backfilled after route assignment
                "assigned_stop_sequence": None,  # backfilled after route assignment
                "required_driver_skills": required_driver_skills,
                "flexibility_flags": flexibility_flags,
                "dietary_tags_snapshot": dietary_snap,
                "mobility_tags_snapshot": mobility_snap,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 7: routes
# ---------------------------------------------------------------------------


def build_routes(
    drivers_df: pd.DataFrame, vehicles_df: pd.DataFrame, depots_df: pd.DataFrame
) -> pd.DataFrame:
    """Create 300 routes across the ~3-week horizon.

    Driver-vehicle pairing respects the driver.vehicle_id assignment (some
    routes randomly use the driver's assigned vehicle; occasionally a
    different compatible one).

    Constraint 11: actual timestamps <= TODAY cutoff.
    """
    rows: list[dict[str, Any]] = []
    driver_rows = drivers_df.to_dict("records")
    depot_ids = depots_df["depot_id"].tolist()

    for i in range(1, N_ROUTES + 1):
        service_date = rand_date_between(HORIZON_START, TODAY)
        driver = rng.choice(driver_rows)
        # 90% of the time use the driver's assigned vehicle
        vehicle_id = driver["vehicle_id"]
        if rng.random() < 0.10:
            vehicle_id = rng.choice(vehicles_df["vehicle_id"].tolist())
        start_depot = driver["home_base_depot_id"]
        end_depot = rng.choice(depot_ids)

        planned_start_minutes = rng.randint(8 * 60, 9 * 60 + 30)
        planned_start = combine_dt(service_date, minutes_to_time(planned_start_minutes))
        planned_time = rng.randint(150, 360)  # 2.5-6 hours
        planned_end = planned_start + timedelta(minutes=planned_time)

        # Actual jitters (constraint 1: actual_start >= planned_start, some lateness)
        actual_start = planned_start + timedelta(minutes=rng.randint(-5, 25))
        actual_duration = int(planned_time * rng.uniform(0.90, 1.20))
        actual_end = actual_start + timedelta(minutes=actual_duration)

        planned_stops = rng.randint(25, 45)
        actual_stops = planned_stops  # stop-gen sets skips; we keep planned==actual row-count

        planned_distance = round(rng.uniform(18.0, 70.0), 1)
        actual_distance = round(planned_distance * rng.uniform(0.90, 1.10), 1)

        # Enforce cutoff
        if actual_end > TIME_CUTOFF:
            actual_end = TIME_CUTOFF
            if actual_start > actual_end:
                actual_start = actual_end - timedelta(minutes=30)

        route_status = weighted_choice(
            [
                ("planned", 0.05),
                ("in_progress", 0.03),
                ("completed", 0.75),
                ("partially_completed", 0.12),
                ("cancelled", 0.05),
            ]
        )

        # Planned vs actual meals (will be adjusted after stops)
        meals_planned = rng.randint(20, 70)
        meals_delivered = meals_planned  # patched after stops land

        # If route hasn't started yet (planned) or was cancelled, blank actuals
        actual_start_val: datetime | None = actual_start
        actual_end_val: datetime | None = actual_end
        actual_distance_val: float | None = actual_distance
        actual_time_val: int | None = actual_duration
        actual_stops_val: int | None = actual_stops
        if route_status == "planned":
            actual_start_val = None
            actual_end_val = None
            actual_distance_val = None
            actual_time_val = None
            actual_stops_val = 0
            meals_delivered = 0
        if route_status == "cancelled":
            actual_start_val = None
            actual_end_val = None
            actual_distance_val = None
            actual_time_val = None
            actual_stops_val = 0
            meals_delivered = 0

        rows.append(
            {
                "route_id": f"RTE-{i:04d}",
                "service_date": service_date,
                "driver_id": driver["driver_id"],
                "vehicle_id": vehicle_id,
                "start_depot_id": start_depot,
                "end_depot_id": end_depot,
                "planned_start_time": planned_start,
                "planned_end_time": planned_end,
                "actual_start_time": actual_start_val,
                "actual_end_time": actual_end_val,
                "planned_distance_km": planned_distance,
                "actual_distance_km": actual_distance_val,
                "planned_time_minutes": planned_time,
                "actual_time_minutes": actual_time_val,
                "planned_stops": planned_stops,
                "actual_stops": actual_stops_val,
                "meals_planned": meals_planned,
                "meals_delivered": meals_delivered,
                "route_status": route_status,
                "on_time_rate": None,          # derived later
                "no_answer_count": 0,          # derived later
                "rerouted_stops_count": 0,     # derived later
                "volunteer_hours_recorded": round(planned_time / 60.0, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 8 & 10: route_stops (with sequencing + time windows + assignment)
# ---------------------------------------------------------------------------


def driver_vehicle_compatible(
    client: dict[str, Any],
    driver: dict[str, Any],
    vehicle: dict[str, Any],
    request: dict[str, Any],
) -> bool:
    """Check constraints 5 + 6 (skill + cold-chain compatibility)."""
    if client.get("mobility_wheelchair") and not driver.get("can_handle_wheelchair"):
        return False
    if client.get("has_dog_on_premises") and driver.get("pet_allergy_flag"):
        return False
    if client.get("interpreter_required"):
        langs = str(driver.get("language_skills") or "").split(";")
        if client.get("language_primary") not in langs:
            return False
    if request.get("cold_chain_required") and not vehicle.get("refrigerated"):
        return False
    return True


def assign_requests_and_build_stops(
    routes_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    clients_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    depots_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign requests to routes obeying compatibility, then build route_stops.

    Returns: (routes_df, requests_df, stops_df) — routes get on_time_rate,
    no_answer_count, rerouted_stops_count; requests get assigned_route_id
    and assigned_stop_sequence.
    """
    stops: list[dict[str, Any]] = []
    client_lookup = clients_df.set_index("client_id").to_dict("index")
    driver_lookup = drivers_df.set_index("driver_id").to_dict("index")
    vehicle_lookup = vehicles_df.set_index("vehicle_id").to_dict("index")
    depot_lookup = depots_df.set_index("depot_id").to_dict("index")

    # Eligible requests = not cancelled and not pending
    eligible_statuses = {"scheduled", "in_progress", "completed", "no_answer", "rerouted"}
    eligible = requests_df[requests_df["status"].isin(eligible_statuses)].copy()
    # Index requests by scheduled_date for fast lookup
    by_date: dict[date, list[int]] = {}
    for idx, r in eligible.iterrows():
        by_date.setdefault(r["scheduled_date"], []).append(idx)

    # Track total meals per depot per day to enforce constraint 9
    depot_day_meals: dict[tuple[str, date], int] = {}

    stop_counter = 1
    # Patch route totals after we fill stops
    route_stop_counts: dict[str, int] = {}
    route_completed_on_time: dict[str, int] = {}
    route_completed_total: dict[str, int] = {}
    route_no_answer: dict[str, int] = {}
    route_rerouted: dict[str, int] = {}
    route_meals_delivered: dict[str, int] = {}
    route_meals_planned_actual: dict[str, int] = {}
    skipped_violations_seeded = 0

    # For each route, we try to consume requests from that date
    for route_idx, route in routes_df.iterrows():
        route_id = route["route_id"]
        service_date = route["service_date"]
        driver = driver_lookup[route["driver_id"]]
        vehicle = vehicle_lookup[route["vehicle_id"]]
        start_depot = depot_lookup.get(route["start_depot_id"])
        planned_stops = int(route["planned_stops"])
        capacity_meals = int(vehicle["capacity_meals"])
        capacity_weight = float(vehicle["capacity_weight_kg"])

        candidates = [eligible.loc[i] for i in by_date.get(service_date, [])]
        # Shuffle candidates then filter for compatibility
        rng.shuffle(candidates)
        chosen: list[pd.Series] = []
        meals_loaded = 0
        weight_loaded = 0.0

        for cand in candidates:
            if len(chosen) >= planned_stops:
                break
            client = client_lookup.get(cand["client_id"])
            if client is None:
                continue
            # Already assigned?
            if requests_df.at[cand.name, "assigned_route_id"] is not None:
                continue
            # Capacity check (constraint 3)
            ql = int(cand["quantity_meals"])
            wl = ql * 0.7 + int(cand["quantity_boxes"]) * 2.0
            if meals_loaded + ql > capacity_meals:
                continue
            if weight_loaded + wl > capacity_weight:
                continue
            # Skill/cold-chain check (constraints 5 + 6)
            if not driver_vehicle_compatible(client, driver, vehicle, cand.to_dict()):
                continue
            # Depot daily meals cap (constraint 9)
            depot_key = (route["start_depot_id"], service_date)
            depot_cap = depot_lookup[route["start_depot_id"]]["capacity_meals_per_day"]
            if depot_day_meals.get(depot_key, 0) + ql > depot_cap:
                continue
            # Accept
            chosen.append(cand)
            meals_loaded += ql
            weight_loaded += wl
            depot_day_meals[depot_key] = depot_day_meals.get(depot_key, 0) + ql

        # Now build stops for the chosen requests, respecting sequence + timing
        if not chosen:
            route_stop_counts[route_id] = 0
            continue

        # Sequence = route order (simple nearest-neighbor heuristic from depot)
        def key_fn(c: pd.Series) -> float:
            cli = client_lookup[c["client_id"]]
            if start_depot is None:
                return 0.0
            return haversine_km(start_depot["lat"], start_depot["lng"], cli["lat"], cli["lng"])

        chosen.sort(key=key_fn)

        # Planned arrival/departure sequencing
        planned_start_dt = route["planned_start_time"]
        current_time = planned_start_dt
        prev_lat = start_depot["lat"] if start_depot else chosen[0]["client_id"]
        prev_lng = start_depot["lng"] if start_depot else 0.0

        actual_current = route["actual_start_time"]
        route_status = route["route_status"]
        # Decide if this route is "early" or "late" for realism
        tardiness_bias = rng.choice([-3, 0, 0, 2, 5, 10])

        completed_on_time_count = 0
        completed_total_count = 0
        no_answer_count = 0
        rerouted_count = 0
        meals_delivered_here = 0
        meals_planned_here = 0

        for seq_idx, cand in enumerate(chosen, 1):
            cli = client_lookup[cand["client_id"]]
            dist_km = haversine_km(prev_lat, prev_lng, cli["lat"], cli["lng"]) if seq_idx > 1 or start_depot else 0.5
            # Urban speed ~ 25 km/h
            travel_min = max(2, int(dist_km / 25.0 * 60))
            planned_arrival = current_time + timedelta(minutes=travel_min)
            service_min = int(cand["service_duration_minutes"])
            planned_departure = planned_arrival + timedelta(minutes=service_min)

            # Compute actual arrival/departure based on status
            actual_arrival: datetime | None = None
            actual_departure: datetime | None = None
            wait_time = 0

            # Determine stop status, honoring request-level status + route-level status
            if route_status == "cancelled":
                stop_status = "cancelled"
            elif route_status == "planned":
                stop_status = "skipped"  # not executed yet — planned only
            else:
                # Weighted status: completed 82% / no_answer 8% / skipped 4% / cancelled 3% / rerouted 3%
                stop_status = weighted_choice(
                    [
                        ("completed", 0.82),
                        ("no_answer", 0.08),
                        ("skipped", 0.04),
                        ("cancelled", 0.03),
                        ("rerouted", 0.03),
                    ]
                )
                # partially_completed must include some skipped/rerouted (constraint 2)
                if route_status == "partially_completed" and seq_idx == 1 and rng.random() < 0.3:
                    stop_status = rng.choice(["skipped", "rerouted"])
                # Two-person requirement unmet (constraint 5)
                if cli.get("requires_two_person_team") and not driver.get("role_type") == "staff":
                    # 5% of such stops fail with specific reason
                    if rng.random() < 0.05:
                        stop_status = "skipped"
                        cand = cand.copy()
                        cand["_failure_reason_override"] = "requires_two_person_unavailable"

            if stop_status in {"completed", "no_answer"}:
                if actual_current is None:
                    actual_current = planned_start_dt
                actual_arrival = actual_current + timedelta(minutes=travel_min + tardiness_bias + rng.randint(-3, 5))
                # 82% in-window, 18% late (constraint 4)
                window_start_dt = combine_dt(cand["scheduled_date"], cand["time_window_start"])
                window_end_dt = combine_dt(cand["scheduled_date"], cand["time_window_end"])
                on_time = window_start_dt <= actual_arrival <= window_end_dt
                if not on_time and rng.random() < 0.70:
                    # Pull back inside the window for ~70% of misses
                    actual_arrival = window_start_dt + timedelta(minutes=rng.randint(0, max(1, int((window_end_dt - window_start_dt).total_seconds() / 60 - 1))))
                    on_time = True
                if actual_arrival > TIME_CUTOFF:
                    actual_arrival = TIME_CUTOFF - timedelta(minutes=rng.randint(5, 60))
                # Wait if early
                if actual_arrival < window_start_dt:
                    wait_time = int((window_start_dt - actual_arrival).total_seconds() / 60)
                # Service duration
                svc = max(3, int(np.random.normal(service_min, 2)))
                actual_departure = actual_arrival + timedelta(minutes=svc)
                if actual_departure > TIME_CUTOFF:
                    actual_departure = TIME_CUTOFF
                actual_current = actual_departure

                if stop_status == "completed":
                    completed_total_count += 1
                    if on_time:
                        completed_on_time_count += 1
                    meals_delivered_here += int(cand["quantity_meals"])
                else:
                    no_answer_count += 1
            else:
                # skipped / cancelled / rerouted -> actuals null
                actual_arrival = None
                actual_departure = None
                if stop_status == "rerouted":
                    rerouted_count += 1

            meals_planned_here += int(cand["quantity_meals"])

            # Failure reason
            failure_reason = None
            if stop_status == "no_answer":
                failure_reason = rng.choice(["no_answer_at_door", "buzzer_failed", "building_locked"])
            elif stop_status == "skipped":
                failure_reason = cand.get("_failure_reason_override") or rng.choice(
                    ["weather", "time_budget_exceeded", "client_not_home_preknown"]
                )
            elif stop_status == "cancelled":
                failure_reason = rng.choice(["route_cancelled", "client_cancel"])
            elif stop_status == "rerouted":
                failure_reason = rng.choice(["vehicle_capacity", "time_window_slip", "wrong_vehicle_type"])

            left_at_door = False
            if stop_status == "completed":
                if cli.get("safe_to_leave_unattended") and "can_leave_at_door" in str(cand["flexibility_flags"]):
                    left_at_door = rng.random() < 0.35
            signature_captured = False
            if stop_status == "completed" and not left_at_door:
                signature_captured = rng.random() < 0.60

            driver_notes = None
            if rng.random() < 0.12:
                driver_notes = rng.choice(
                    [
                        "Client chatty today",
                        "Left meal on counter",
                        "Delivered to neighbor per instructions",
                        "Client using walker, door ajar",
                        "Note: dog barking but friendly",
                        "Called ahead, no answer",
                    ]
                )

            stops.append(
                {
                    "route_stop_id": f"STP-{stop_counter:05d}",
                    "route_id": route_id,
                    "request_id": cand["request_id"],
                    "client_id": cand["client_id"],
                    "sequence_index": seq_idx,
                    "planned_arrival": planned_arrival,
                    "planned_departure": planned_departure,
                    "actual_arrival": actual_arrival,
                    "actual_departure": actual_departure,
                    "wait_time_minutes": wait_time,
                    "service_duration_minutes": service_min,
                    "distance_from_prev_km": round(dist_km, 2),
                    "travel_time_from_prev_minutes": travel_min,
                    "status": stop_status,
                    "failure_reason": failure_reason,
                    "left_at_door": left_at_door,
                    "signature_captured": signature_captured,
                    "driver_notes": driver_notes,
                }
            )

            # Backfill request assignment
            requests_df.at[cand.name, "assigned_route_id"] = route_id
            requests_df.at[cand.name, "assigned_stop_sequence"] = seq_idx

            current_time = planned_departure
            prev_lat = cli["lat"]
            prev_lng = cli["lng"]
            stop_counter += 1

        # Record route-level stats
        route_stop_counts[route_id] = len(chosen)
        route_completed_on_time[route_id] = completed_on_time_count
        route_completed_total[route_id] = completed_total_count
        route_no_answer[route_id] = no_answer_count
        route_rerouted[route_id] = rerouted_count
        route_meals_delivered[route_id] = meals_delivered_here
        route_meals_planned_actual[route_id] = meals_planned_here

        # partially_completed should have >=1 skipped or rerouted (constraint 2)
        if route_status == "partially_completed":
            if rerouted_count == 0 and all(
                s.get("status") not in {"skipped", "rerouted"}
                for s in stops
                if s["route_id"] == route_id
            ):
                # Force one stop to skipped
                for s in stops:
                    if s["route_id"] == route_id and s["status"] == "completed":
                        s["status"] = "rerouted"
                        s["actual_arrival"] = None
                        s["actual_departure"] = None
                        s["failure_reason"] = "vehicle_capacity"
                        route_rerouted[route_id] = route_rerouted.get(route_id, 0) + 1
                        break

    # Update routes_df with derived stats
    def _update(rid: str, key: str, default: Any = 0) -> Any:
        return {
            "actual_stops": route_stop_counts.get(rid, 0),
            "no_answer_count": route_no_answer.get(rid, 0),
            "rerouted_stops_count": route_rerouted.get(rid, 0),
            "meals_delivered": route_meals_delivered.get(rid, 0),
            "on_time_rate": (
                round(route_completed_on_time[rid] / route_completed_total[rid], 3)
                if route_completed_total.get(rid, 0) > 0
                else None
            ),
        }

    def _route_patch(row: pd.Series) -> pd.Series:
        rid = row["route_id"]
        patch = _update(rid, "")
        for k, v in patch.items():
            row[k] = v
        # Status coherence (constraint 2): if completed route has any skipped/cancelled stops,
        # flip to partially_completed
        if row["route_status"] == "completed":
            rstops = [s for s in stops if s["route_id"] == rid]
            if any(s["status"] in {"skipped", "rerouted", "cancelled"} for s in rstops):
                row["route_status"] = "partially_completed"
        return row

    routes_df = routes_df.apply(_route_patch, axis=1)
    stops_df = pd.DataFrame(stops)
    return routes_df, requests_df, stops_df


# ---------------------------------------------------------------------------
# SECTION 9: delivery_request_items
# ---------------------------------------------------------------------------


def build_request_items(
    requests_df: pd.DataFrame, clients_df: pd.DataFrame, items_df: pd.DataFrame
) -> pd.DataFrame:
    """Create ~4,000 request-item lines (1-4 items per request).

    Enforces constraint 7 (dietary safety): no severe-allergen item lands
    on a request whose client has that allergen flagged as severe.
    Three intentional violations are seeded via driver_notes marker.
    """
    lines: list[dict[str, Any]] = []
    client_lookup = clients_df.set_index("client_id").to_dict("index")

    # Category weights by program type
    cat_pool_by_program = {
        "MOW_hot": ["hot_meal", "hot_meal", "hot_meal", "dairy", "produce"],
        "MOW_frozen": ["frozen_meal", "frozen_meal", "dairy", "dry_goods"],
        "grocery_hamper": ["produce", "produce", "dairy", "dry_goods", "protein", "hamper_component"],
        "culturally_specific": ["hamper_component", "dry_goods", "protein", "produce"],
        "emergency_box": ["hamper_component", "dry_goods", "dairy"],
    }

    severe_allergen_map = {
        "peanut": "allergy_peanut_severity",
        "tree_nut": "allergy_tree_nut_severity",
        "shellfish": "allergy_shellfish_severity",
        "fish": "allergy_fish_severity",
        "egg": "allergy_egg_severity",
        "soy": "allergy_soy_severity",
        "wheat": "allergy_wheat_severity",
        "dairy": "allergy_dairy_severity",
    }

    items_by_cat: dict[str, list[dict[str, Any]]] = {}
    for _, it in items_df.iterrows():
        items_by_cat.setdefault(it["category"], []).append(it.to_dict())

    line_counter = 1
    violation_slots = 3
    violations_seeded = 0

    # Target ~4000 lines across 2000 requests -> mean 2 per request
    for _, req in requests_df.iterrows():
        client = client_lookup.get(req["client_id"])
        if client is None:
            continue
        # Skip cancelled requests more often (fewer lines)
        if req["status"] == "cancelled" and rng.random() < 0.6:
            continue
        n_items = rng.randint(1, 4)
        program = req["program_type"]
        pool_cats = cat_pool_by_program.get(program, list(items_by_cat.keys()))

        banned_allergens = {
            alg for alg, field in severe_allergen_map.items()
            if client.get(field) == "severe"
        }

        picked: list[dict[str, Any]] = []
        attempts = 0
        while len(picked) < n_items and attempts < 40:
            attempts += 1
            cat = rng.choice(pool_cats)
            bucket = items_by_cat.get(cat, [])
            if not bucket:
                continue
            it = rng.choice(bucket)
            item_allergens = set(
                a for a in str(it.get("allergen_flags") or "").split(";") if a
            )
            if item_allergens & banned_allergens:
                continue
            picked.append(it)

        # If we couldn't find 1 item (rare), fall back
        if not picked:
            picked = [rng.choice(list(items_df.to_dict("records")))]

        # Seed 3 intentional violations: pick one severe-allergen client and
        # insert a conflicting item with the DATA_QUALITY_ISSUE marker
        if violations_seeded < violation_slots and banned_allergens and rng.random() < 0.02:
            # Find an item that DOES match a banned allergen
            for it in items_df.to_dict("records"):
                item_allergens = set(
                    a for a in str(it.get("allergen_flags") or "").split(";") if a
                )
                if item_allergens & banned_allergens:
                    picked.append(it)
                    violations_seeded += 1
                    lines.append(
                        {
                            "line_id": f"LIN-{line_counter:05d}",
                            "request_id": req["request_id"],
                            "item_id": it["item_id"],
                            "quantity": rng.randint(1, 3),
                            "notes": "DATA_QUALITY_ISSUE: allergen conflict",
                        }
                    )
                    line_counter += 1
                    break

        for it in picked:
            lines.append(
                {
                    "line_id": f"LIN-{line_counter:05d}",
                    "request_id": req["request_id"],
                    "item_id": it["item_id"],
                    "quantity": rng.randint(1, 5),
                    "notes": rng.choice(
                        [None, None, None, "Per client request", "Extra portion",
                         "Substitute if unavailable", "Frozen preferred"]
                    ),
                }
            )
            line_counter += 1

    return pd.DataFrame(lines)


# ---------------------------------------------------------------------------
# SECTION 11: derive KPIs - driver + route
# ---------------------------------------------------------------------------


def derive_kpis(
    drivers_df: pd.DataFrame, routes_df: pd.DataFrame, stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Recompute driver.on_time_rate and no_answer_rate from the actual stops.

    constraint 10: driver.on_time_rate = completed_on_time / total_completed
    """
    # For each driver, sum completed and on-time completed across all their routes
    route_driver = routes_df[["route_id", "driver_id"]]
    merged = stops_df.merge(route_driver, on="route_id", how="left")
    completed = merged[merged["status"] == "completed"].copy()
    no_answer = merged[merged["status"] == "no_answer"].copy()

    # On-time: actual_arrival within request window -- we re-derive from actuals
    # Since we already shifted arrivals to window, treat any completed row with
    # actual_arrival not null as the base; re-infer on-time from actual_arrival <=
    # request.time_window_end would require the request join. Easier path: use
    # stop signature that we already gated in build_stops. Approximate with
    # random-backed bias via existing on_time_rate per route.
    total_by_driver = completed.groupby("driver_id").size().to_dict()
    na_by_driver = no_answer.groupby("driver_id").size().to_dict()

    # On-time uses route-level on_time_rate weighted by stops
    completed_with_rate = completed.merge(routes_df[["route_id", "on_time_rate"]], on="route_id", how="left")
    completed_with_rate["on_time_rate"] = completed_with_rate["on_time_rate"].fillna(0.85)
    # Expected on-time count per driver
    ot_count_by_driver = completed_with_rate.groupby("driver_id")["on_time_rate"].sum().to_dict()

    def _compute(row: pd.Series) -> pd.Series:
        did = row["driver_id"]
        tot = total_by_driver.get(did, 0)
        if tot > 0:
            row["on_time_rate"] = round(ot_count_by_driver.get(did, 0) / tot, 3)
        na = na_by_driver.get(did, 0)
        denom = tot + na
        if denom > 0:
            row["no_answer_rate"] = round(na / denom, 3)
        return row

    drivers_df = drivers_df.apply(_compute, axis=1)
    return drivers_df


# ---------------------------------------------------------------------------
# SECTION 12: messiness injection
# ---------------------------------------------------------------------------


def messy_null() -> Any:
    """60% true null, 25% empty string, 10% 'Unknown'/'N/A', 5% 999."""
    r = rng.random()
    if r < 0.60:
        return None
    if r < 0.85:
        return ""
    if r < 0.95:
        return rng.choice(["Unknown", "N/A"])
    return 999


def messy_phone() -> str:
    """Phone in one of 5 formats with spec distribution."""
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
    """Apply the 75/12/8/5 date-format distribution as strings."""
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
    """Name case: 70% Title, 15% UPPER, 10% lower, 5% mIxEd."""
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
    requests_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared messiness config: phones, dates, cases, missingness, orphans.

    Also seeds the 4 deceased/closed-client violations (constraint 12) and
    keeps the allergen-conflict markers placed by build_request_items.
    """
    # --- clients ---
    clients_df["phone"] = clients_df["phone"].apply(lambda _: messy_phone())
    # Email already sometimes None — top up to 30% missingness
    email_fill_mask = clients_df["email"].isna()
    target_missing = int(len(clients_df) * 0.30)
    current_missing = int(email_fill_mask.sum())
    if current_missing < target_missing:
        # Drop more emails
        to_null = clients_df[~email_fill_mask].sample(
            n=target_missing - current_missing, random_state=SEED
        ).index
        clients_df.loc[to_null, "email"] = [messy_null() for _ in to_null]

    phone_mask = clients_df.sample(frac=0.12, random_state=SEED + 1).index
    clients_df.loc[phone_mask, "phone"] = [messy_null() for _ in phone_mask]

    # buzzer_code 40% missing (on top of the structural None for houses)
    buzzer_mask = clients_df.sample(frac=0.40, random_state=SEED + 2).index
    clients_df.loc[buzzer_mask, "buzzer_code"] = [messy_null() for _ in buzzer_mask]

    # Name case chaos
    clients_df["first_name"] = clients_df["first_name"].apply(messy_case)
    clients_df["last_name"] = clients_df["last_name"].apply(messy_case)

    # Date formatting chaos on a subset of date columns
    for col in ["dob", "enrolment_start_date", "closure_date"]:
        clients_df[col] = clients_df[col].apply(messy_date)

    # --- drivers: phone chaos ---
    drivers_df["phone"] = drivers_df["phone"].apply(lambda _: messy_phone())

    # --- requests: 0.3% blank request_ids ---
    blank_idx = requests_df.sample(frac=0.003, random_state=SEED + 3).index
    requests_df.loc[blank_idx, "request_id"] = ""

    # --- routes: 0.8% orphan driver_id (FK violation) ---
    orphan_idx = routes_df.sample(frac=0.008, random_state=SEED + 4).index
    routes_df.loc[orphan_idx, "driver_id"] = [
        f"DRV-{rng.randint(90, 99):02d}" for _ in orphan_idx
    ]

    # Null injection on some nullable fields
    for col in ["driver_notes"]:
        null_mask = stops_df.sample(frac=0.10, random_state=SEED + 5).index
        stops_df.loc[null_mask, col] = [messy_null() for _ in null_mask]

    # Seed 4 deceased/closed constraint violations (constraint 12).
    # Earlier versions picked any request for the closed client, but many such
    # requests never got assigned to a route, so the "delivered after closure"
    # tag landed on 0-2 stops instead of 4. We now restrict to closed clients
    # whose requests actually produced a route_stop (ensuring the tag lands).
    closed_clients = clients_df[
        clients_df["enrolment_status"].isin(["deceased", "closed"])
    ].copy()
    # Build the set of client_ids that have at least one request with a stop
    assigned_request_ids = set(stops_df["request_id"].dropna().unique())
    assigned_reqs = requests_df[requests_df["request_id"].isin(assigned_request_ids)]
    clients_with_stops = set(assigned_reqs["client_id"].unique())
    seedable = closed_clients[closed_clients["client_id"].isin(clients_with_stops)].copy()
    seeded = 0
    if len(seedable) >= 4:
        sample = seedable.sample(n=4, random_state=SEED + 6)
        for _, cli in sample.iterrows():
            # Prefer a request belonging to this client that DOES have a stop
            reqs_for_client = requests_df[
                (requests_df["client_id"] == cli["client_id"])
                & (requests_df["request_id"].isin(assigned_request_ids))
            ]
            if reqs_for_client.empty:
                continue
            # Parse closure_date if it's been messied to string
            closure = cli["closure_date"]
            if isinstance(closure, str):
                try:
                    closure = date.fromisoformat(closure)
                except Exception:
                    closure = TODAY - timedelta(days=10)
            if not isinstance(closure, date):
                closure = TODAY - timedelta(days=10)
            shifted = closure + timedelta(days=rng.randint(2, 15))
            if shifted > TODAY:
                shifted = TODAY
            ridx = reqs_for_client.index[0]
            requests_df.at[ridx, "scheduled_date"] = shifted
            # Mark related stop with a driver_note so audits can detect it
            stop_hits = stops_df[stops_df["request_id"] == requests_df.at[ridx, "request_id"]]
            if not stop_hits.empty:
                sidx = stop_hits.index[0]
                stops_df.at[sidx, "driver_notes"] = "DATA_QUALITY_ISSUE: delivered after closure"
                seeded += 1

    return clients_df, requests_df, routes_df, stops_df, drivers_df, items_df


# ---------------------------------------------------------------------------
# SECTION 13: write outputs (parquet + sqlite + views + CSVs)
# ---------------------------------------------------------------------------


def _to_sqlite_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce datetime/time/date columns so sqlite3 doesn't choke."""
    out = df.copy()
    for col in out.columns:
        sample = next((v for v in out[col] if v is not None and not (isinstance(v, float) and math.isnan(v))), None)
        if isinstance(sample, time):
            out[col] = out[col].apply(lambda v: v.strftime("%H:%M:%S") if isinstance(v, time) else v)
        elif isinstance(sample, datetime):
            out[col] = out[col].apply(lambda v: v.isoformat() if isinstance(v, datetime) else v)
        elif isinstance(sample, date):
            out[col] = out[col].apply(lambda v: v.isoformat() if isinstance(v, date) else v)
    return out


def _coerce_objects_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Messiness injection leaves object columns with mixed Python types (str,
    int sentinels like 999, None). PyArrow rejects mixed-type object columns,
    so coerce every object column to string (nulls preserved as None)."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].apply(
            lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v)))
            else (v if isinstance(v, str) else str(v))
        )
    return out


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write parquet + sqlite + sample CSVs and create the two enrichment views."""
    # Parquet
    for name, df in tables.items():
        _coerce_objects_to_str(df).to_parquet(RAW_DIR / f"{name}.parquet", index=False)

    # Sample CSVs
    for name, df in tables.items():
        df.head(1000).to_csv(SAMPLE_DIR / f"{name}_sample.csv", index=False)

    # SQLite
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    conn = sqlite3.connect(SQLITE_PATH)
    try:
        for name, df in tables.items():
            _to_sqlite_safe(df).to_sql(name, conn, if_exists="replace", index=False)

        conn.executescript(
            """
            DROP VIEW IF EXISTS v_route_performance;
            CREATE VIEW v_route_performance AS
            SELECT
                r.route_id,
                r.service_date,
                r.driver_id,
                r.vehicle_id,
                r.route_status,
                r.planned_stops,
                r.actual_stops,
                r.planned_distance_km,
                r.actual_distance_km,
                r.planned_time_minutes,
                r.actual_time_minutes,
                r.meals_planned,
                r.meals_delivered,
                r.on_time_rate,
                r.no_answer_count,
                r.rerouted_stops_count,
                agg.stops_total        AS stops_total_observed,
                agg.stops_completed    AS stops_completed_observed,
                agg.stops_no_answer    AS stops_no_answer_observed,
                agg.stops_skipped      AS stops_skipped_observed,
                agg.stops_cancelled    AS stops_cancelled_observed,
                agg.stops_rerouted     AS stops_rerouted_observed,
                agg.avg_service_duration
            FROM routes r
            LEFT JOIN (
                SELECT
                    route_id,
                    COUNT(*) AS stops_total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS stops_completed,
                    SUM(CASE WHEN status = 'no_answer' THEN 1 ELSE 0 END) AS stops_no_answer,
                    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS stops_skipped,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS stops_cancelled,
                    SUM(CASE WHEN status = 'rerouted' THEN 1 ELSE 0 END) AS stops_rerouted,
                    AVG(service_duration_minutes) AS avg_service_duration
                FROM route_stops
                GROUP BY route_id
            ) agg ON agg.route_id = r.route_id;

            DROP VIEW IF EXISTS v_stop_details;
            CREATE VIEW v_stop_details AS
            SELECT
                s.route_stop_id,
                s.route_id,
                s.sequence_index,
                s.planned_arrival,
                s.actual_arrival,
                s.status        AS stop_status,
                s.failure_reason,
                s.left_at_door,
                s.signature_captured,
                r.request_id,
                r.scheduled_date,
                r.time_window_start,
                r.time_window_end,
                r.program_type,
                r.priority_level,
                r.cold_chain_required,
                r.quantity_meals,
                c.client_id,
                c.first_name    AS client_first_name,
                c.last_name     AS client_last_name,
                c.address_street,
                c.address_postal,
                c.building_type,
                c.enrolment_status,
                c.food_security_level,
                c.mobility_wheelchair,
                c.has_dog_on_premises,
                c.interpreter_required,
                c.requires_two_person_team,
                c.client_trust_level,
                (
                    SELECT GROUP_CONCAT(i.name, '; ')
                    FROM delivery_request_items dri
                    LEFT JOIN inventory_items i ON i.item_id = dri.item_id
                    WHERE dri.request_id = s.request_id
                ) AS item_names,
                (
                    SELECT SUM(dri.quantity)
                    FROM delivery_request_items dri
                    WHERE dri.request_id = s.request_id
                ) AS total_item_quantity
            FROM route_stops s
            LEFT JOIN delivery_requests r ON r.request_id = s.request_id
            LEFT JOIN clients c           ON c.client_id  = s.client_id;
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SECTION 14: summary + KPI preview
# ---------------------------------------------------------------------------


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Row counts, null-rep stats, and a sample KPI preview."""
    print("\n" + "=" * 78)
    print("Track 2 Generator - Food Security Delivery Operations - Summary")
    print("=" * 78)
    print(f"{'table':<28}{'rows':>12}{'null_rep_pct':>18}")
    print("-" * 78)

    total_cells = 0
    total_flagged = 0
    for name, df in tables.items():
        rows = len(df)
        flagged = 0
        cells = df.size
        for col in df.columns:
            series = df[col]
            flagged += int(series.isna().sum())
            flagged += int((series.astype(str) == "").sum())
            flagged += int(series.isin(["Unknown", "N/A"]).sum())
            flagged += int((series.astype(str) == "999").sum())
        pct = (flagged / cells * 100) if cells else 0
        total_cells += cells
        total_flagged += flagged
        print(f"{name:<28}{rows:>12,}{pct:>17.2f}%")

    print("-" * 78)
    overall = (total_flagged / total_cells * 100) if total_cells else 0
    total_rows = sum(len(df) for df in tables.values())
    print(f"{'TOTAL':<28}{total_rows:>12,}{overall:>17.2f}%")
    print("=" * 78)

    # KPI preview
    stops_df = tables["route_stops"]
    routes_df = tables["routes"]
    drivers_df = tables["drivers"]
    print("\nKPI Preview")
    print("-" * 78)
    total_stops = len(stops_df)
    completed_stops = (stops_df["status"] == "completed").sum()
    no_answer_stops = (stops_df["status"] == "no_answer").sum()
    print(f"  total route_stops              : {total_stops:,}")
    print(f"  completed stops                : {completed_stops:,} ({completed_stops/total_stops*100:.1f}%)")
    print(f"  no-answer stops                : {no_answer_stops:,} ({no_answer_stops/total_stops*100:.1f}%)")
    ot_vals = routes_df["on_time_rate"].dropna()
    if len(ot_vals):
        print(f"  mean route on_time_rate        : {ot_vals.mean():.3f}")
    print(f"  routes completed               : {(routes_df['route_status']=='completed').sum()}")
    print(f"  routes partially_completed     : {(routes_df['route_status']=='partially_completed').sum()}")
    print(f"  mean driver on_time_rate       : {drivers_df['on_time_rate'].mean():.3f}")
    print(f"  mean driver no_answer_rate     : {drivers_df['no_answer_rate'].mean():.3f}")
    print("-" * 78)
    print(f"Parquet:  {RAW_DIR}")
    print(f"SQLite:   {SQLITE_PATH}")
    print(f"Samples:  {SAMPLE_DIR}")
    print("Views:    v_route_performance, v_stop_details")
    print("Seed:     42 (deterministic)")
    print(f"Cutoff:   {TODAY.isoformat()}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Track 2 generator pipeline end-to-end."""
    print("[1/10] depots ...")
    depots_df = build_depots()

    print("[2/10] vehicles ...")
    vehicles_df = build_vehicles()

    print("[3/10] drivers ...")
    drivers_df = build_drivers(vehicles_df, depots_df)

    print("[4/10] clients (Victoria geofence + mobility/diet/allergy) ...")
    clients_df = build_clients(depots_df)

    print("[5/10] inventory_items catalogue ...")
    items_df = build_inventory_items()

    print("[6/10] delivery_requests ...")
    requests_df = build_delivery_requests(clients_df)

    print("[7/10] routes ...")
    routes_df = build_routes(drivers_df, vehicles_df, depots_df)

    print("[8/10] assign requests -> routes and build route_stops (skill + cold-chain + capacity) ...")
    routes_df, requests_df, stops_df = assign_requests_and_build_stops(
        routes_df, requests_df, clients_df, drivers_df, vehicles_df, depots_df
    )

    print("[8b/10] delivery_request_items (allergen-safe + 3 seeded violations) ...")
    request_items_df = build_request_items(requests_df, clients_df, items_df)

    print("[9/10] derive KPIs (driver.on_time_rate, driver.no_answer_rate) ...")
    drivers_df = derive_kpis(drivers_df, routes_df, stops_df)

    print("[9b/10] messiness injection pass + 4 closure violations ...")
    clients_df, requests_df, routes_df, stops_df, drivers_df, items_df = inject_messiness(
        clients_df, requests_df, routes_df, stops_df, drivers_df, items_df
    )

    print("[10/10] writing parquet + sqlite + CSV samples + views ...")
    tables = {
        "depots": depots_df,
        "vehicles": vehicles_df,
        "drivers": drivers_df,
        "clients": clients_df,
        "inventory_items": items_df,
        "delivery_requests": requests_df,
        "routes": routes_df,
        "route_stops": stops_df,
        "delivery_request_items": request_items_df,
    }
    write_outputs(tables)
    print_summary(tables)


if __name__ == "__main__":
    main()
