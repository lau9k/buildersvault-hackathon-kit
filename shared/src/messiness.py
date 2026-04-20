"""Controlled messiness helpers for synthetic-data generation.

Hackathon data needs to mirror the rough edges of real government and
social-services systems: inconsistent phone formats, multiple date styles,
mixed name casing, and a mess of null representations. These helpers give
generators a single, auditable place to sprinkle that chaos so both tracks
stay consistent and the messiness ratios are easy to tune.

All functions accept a ``numpy.random.Generator`` (or ``random.Random``)
so generation remains deterministic under a fixed seed.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol


class _RNGLike(Protocol):
    """Minimal protocol for the RNG-like object the helpers expect."""

    def random(self) -> float: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Messiness ratios are expressed as probabilities per cell. They are intentionally
# small: the point is to make cleaning a real task, not to flood the dataset
# with noise. Ratios per sub-format sum to 1.0 within each family.
MESSINESS_CONFIG: dict[str, Any] = {
    # Overall probability that a given cell is rewritten in a non-canonical format.
    "phone_format_mix_rate": 0.65,
    "date_format_mix_rate": 0.40,
    "name_case_mix_rate": 0.30,
    "null_injection_rate": 0.04,
    # Sub-format mix for phone numbers. Must sum to 1.0.
    "phone_formats": {
        "parens_space": 0.35,      # (250) 555-0143
        "dashes": 0.25,            # 250-555-0143
        "dots": 0.10,              # 250.555.0143
        "plain_digits": 0.20,      # 2505550143
        "e164": 0.10,              # +12505550143
    },
    # Sub-format mix for dates. Must sum to 1.0.
    "date_formats": {
        "iso": 0.55,               # 2026-04-15
        "slash_ymd": 0.15,         # 2026/04/15
        "slash_dmy": 0.20,         # 15/04/2026
        "text_long": 0.10,         # April 15, 2026
    },
    # Sub-format mix for names. Must sum to 1.0.
    "name_cases": {
        "title": 0.55,             # Sarah Thompson
        "upper": 0.15,             # SARAH THOMPSON
        "lower": 0.15,             # sarah thompson
        "weird": 0.15,             # sArAh thompson / trailing whitespace
    },
    # Sub-format mix for null-like values. Must sum to 1.0.
    "null_values": {
        "python_none": 0.50,       # actual None
        "empty_string": 0.20,      # ""
        "literal_unknown": 0.20,   # "Unknown"
        "sentinel_999": 0.10,      # 999 or "999"
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _digits_only(phone: str) -> str:
    """Strip everything except digits from a phone string."""
    return "".join(ch for ch in str(phone) if ch.isdigit())


def _pick(rng: _RNGLike, weighted: dict[str, float]) -> str:
    """Pick a key from a weighted dict using the provided RNG.

    The weights must sum to 1.0 (we do not normalise here to keep the
    configuration honest and catch drift early).
    """
    r = rng.random()
    cumulative = 0.0
    for key, weight in weighted.items():
        cumulative += weight
        if r <= cumulative:
            return key
    # Floating-point rounding guard: fall back to last key.
    return next(reversed(weighted))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def inject_phone_format(phone: str, rng: _RNGLike) -> str:
    """Return ``phone`` rewritten in a mixed format with probability.

    With probability ``phone_format_mix_rate`` the phone is reformatted into
    one of five styles according to ``phone_formats``. Otherwise the input
    is returned unchanged.
    """
    if phone is None or str(phone).strip() == "":
        return phone

    if rng.random() > MESSINESS_CONFIG["phone_format_mix_rate"]:
        return phone

    digits = _digits_only(phone)
    if len(digits) < 10:
        # Not a parseable NANP number; leave it alone to avoid inventing data.
        return phone

    # Normalise to a 10-digit North-American form before reformatting.
    area, prefix, line = digits[-10:-7], digits[-7:-4], digits[-4:]

    style = _pick(rng, MESSINESS_CONFIG["phone_formats"])
    if style == "parens_space":
        return f"({area}) {prefix}-{line}"
    if style == "dashes":
        return f"{area}-{prefix}-{line}"
    if style == "dots":
        return f"{area}.{prefix}.{line}"
    if style == "plain_digits":
        return f"{area}{prefix}{line}"
    if style == "e164":
        return f"+1{area}{prefix}{line}"
    return phone


def inject_date_format(value: Any, rng: _RNGLike) -> Any:
    """Return ``value`` rewritten in a mixed date format with probability.

    Accepts ``date``, ``datetime``, or ISO-formatted strings. Non-date inputs
    are returned unchanged. With probability ``date_format_mix_rate`` the
    canonical ISO string is replaced with one of four stylistic variants.
    """
    if value is None:
        return value

    if isinstance(value, datetime):
        d = value.date()
    elif isinstance(value, date):
        d = value
    else:
        # Try to parse an ISO string; otherwise leave the value untouched.
        try:
            d = date.fromisoformat(str(value)[:10])
        except ValueError:
            return value

    if rng.random() > MESSINESS_CONFIG["date_format_mix_rate"]:
        return d.isoformat()

    style = _pick(rng, MESSINESS_CONFIG["date_formats"])
    if style == "iso":
        return d.isoformat()
    if style == "slash_ymd":
        return f"{d.year:04d}/{d.month:02d}/{d.day:02d}"
    if style == "slash_dmy":
        return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
    if style == "text_long":
        return d.strftime("%B %d, %Y")
    return d.isoformat()


def inject_name_case(name: str, rng: _RNGLike) -> str:
    """Return ``name`` rewritten with mixed casing with probability.

    With probability ``name_case_mix_rate`` the name is rewritten in one of
    four casing styles. The ``weird`` bucket also randomly adds trailing
    whitespace to simulate data-entry artefacts.
    """
    if name is None or str(name).strip() == "":
        return name

    if rng.random() > MESSINESS_CONFIG["name_case_mix_rate"]:
        return name

    style = _pick(rng, MESSINESS_CONFIG["name_cases"])
    if style == "title":
        return str(name).title()
    if style == "upper":
        return str(name).upper()
    if style == "lower":
        return str(name).lower()
    if style == "weird":
        # Alternating-case + possible trailing whitespace. Common artefact
        # of copy-paste from PDFs and legacy terminals.
        weirded = "".join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(str(name)))
        if rng.random() < 0.5:
            weirded = weirded + "  "
        return weirded
    return name


def inject_null_representation(value: Any, rng: _RNGLike) -> Any:
    """Replace ``value`` with a null-like sentinel with probability.

    With probability ``null_injection_rate`` the value is replaced by one of
    four null representations drawn from ``null_values``. Otherwise the
    input is returned unchanged.
    """
    if rng.random() > MESSINESS_CONFIG["null_injection_rate"]:
        return value

    style = _pick(rng, MESSINESS_CONFIG["null_values"])
    if style == "python_none":
        return None
    if style == "empty_string":
        return ""
    if style == "literal_unknown":
        return "Unknown"
    if style == "sentinel_999":
        # Numeric sentinel when the source value is numeric, string "999" otherwise.
        return 999 if isinstance(value, (int, float)) else "999"
    return value
