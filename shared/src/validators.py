"""Lightweight schema and referential-integrity validators.

These helpers are intentionally permissive. They print human-readable
pass/fail summaries rather than raising, so notebooks can keep running
and participants can triage issues in bulk.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def check_schema(df: pd.DataFrame, expected_cols: Iterable[str], table_name: str) -> bool:
    """Verify that ``df`` contains every column in ``expected_cols``.

    Prints a compact pass/fail summary and returns True when all expected
    columns are present. Extra columns (not listed in ``expected_cols``) are
    reported as informational warnings but do not cause a failure.

    Args:
        df: DataFrame to inspect.
        expected_cols: Iterable of required column names.
        table_name: Human-readable table name for the log output.

    Returns:
        True when all expected columns are present, False otherwise.
    """
    expected = list(expected_cols)
    actual = list(df.columns)

    missing: List[str] = [c for c in expected if c not in actual]
    extra: List[str] = [c for c in actual if c not in expected]

    if not missing:
        print(f"[PASS] schema {table_name}: {len(expected)} expected columns present, {len(df):,} rows.")
        if extra:
            # Extra columns are common during early prototyping; surface them
            # as info so hackers can spot schema drift without being blocked.
            print(f"       info: {len(extra)} extra column(s) present: {extra[:8]}{'...' if len(extra) > 8 else ''}")
        return True

    print(f"[FAIL] schema {table_name}: missing {len(missing)} column(s): {missing}")
    if extra:
        print(f"       info: {len(extra)} extra column(s) present: {extra[:8]}{'...' if len(extra) > 8 else ''}")
    return False


def check_referential_integrity(
    df_child: pd.DataFrame,
    child_col: str,
    df_parent: pd.DataFrame,
    parent_col: str,
    name: str,
) -> int:
    """Report child rows whose foreign key has no matching parent row.

    Nulls in ``child_col`` are ignored (treated as legitimately unmapped).

    Args:
        df_child: Child DataFrame that holds the foreign key column.
        child_col: Foreign key column name in the child DataFrame.
        df_parent: Parent DataFrame that should contain the referenced rows.
        parent_col: Primary key column name in the parent DataFrame.
        name: Short label for the relationship, used in log output.

    Returns:
        Count of orphan rows found. Zero means referential integrity holds.
    """
    if child_col not in df_child.columns:
        print(f"[SKIP] FK check {name}: child column '{child_col}' not in DataFrame.")
        return 0
    if parent_col not in df_parent.columns:
        print(f"[SKIP] FK check {name}: parent column '{parent_col}' not in DataFrame.")
        return 0

    # Ignore null FKs so that optional relationships don't get flagged as orphans.
    child_non_null = df_child[df_child[child_col].notna()]
    parent_keys = set(df_parent[parent_col].dropna().unique())
    orphan_mask = ~child_non_null[child_col].isin(parent_keys)
    orphan_count = int(orphan_mask.sum())

    if orphan_count == 0:
        print(f"[PASS] FK {name}: all {len(child_non_null):,} non-null keys resolve.")
    else:
        sample = child_non_null.loc[orphan_mask, child_col].head(5).tolist()
        print(f"[WARN] FK {name}: {orphan_count:,} orphan row(s). Sample keys: {sample}")
    return orphan_count


def null_summary(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Return a per-column null summary.

    Args:
        df: DataFrame to summarise.
        table_name: Short label included in the output for multi-table reports.

    Returns:
        DataFrame with columns: table, column, null_count, null_pct, dtype.
        Sorted by null_pct descending so the worst offenders surface first.
    """
    if len(df) == 0:
        # Empty DataFrames break division; return an empty shape with the
        # same columns so callers can concat results across tables safely.
        return pd.DataFrame(columns=["table", "column", "null_count", "null_pct", "dtype"])

    nulls = df.isna().sum()
    summary = pd.DataFrame(
        {
            "table": table_name,
            "column": nulls.index,
            "null_count": nulls.values,
            "null_pct": (nulls.values / len(df) * 100).round(2),
            "dtype": [str(df[c].dtype) for c in nulls.index],
        }
    )
    return summary.sort_values("null_pct", ascending=False).reset_index(drop=True)
