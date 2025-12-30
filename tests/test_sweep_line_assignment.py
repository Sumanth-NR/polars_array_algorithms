from datetime import date, datetime

import polars as pl
import pytest
from polars.exceptions import ComputeError

import polars_array_algorithms as pl_alg


def test_basic_integer_sweep():
    """Test standard room allocation logic with integers."""
    df = pl.DataFrame({"start": [1, 2, 4, 8], "end": [4, 5, 8, 10]})

    # overlapping=False (Default): reuse at exact boundary
    res = df.with_columns(
        room_id=pl_alg.sweep_line_assignment("start", "end", overlapping=False)
    )
    assert res["room_id"].to_list() == [1, 2, 1, 1]


def test_overlap_true_boundary():
    """Verify that overlapping=True prevents reuse at the exact same tick."""
    df = pl.DataFrame({"start": [1, 4], "end": [4, 8]})

    res = df.with_columns(
        room_id=pl_alg.sweep_line_assignment("start", "end", overlapping=True)
    )
    assert res["room_id"].to_list() == [1, 2]


def test_datetime_dispatch():
    """Verify that to_physical_repr correctly handles Datetimes."""
    df = pl.DataFrame(
        {
            "start": [
                datetime(2025, 1, 1, 10, 0),
                datetime(2025, 1, 1, 11, 0),
            ],
            "end": [
                datetime(2025, 1, 1, 11, 0),
                datetime(2025, 1, 1, 12, 0),
            ],
        }
    )

    # Rust sees i64 ticks; reuse happens at 11:00
    res = df.with_columns(
        room_id=pl_alg.sweep_line_assignment("start", "end", overlapping=False)
    )
    assert res["room_id"].to_list() == [1, 1]
    assert res["room_id"].dtype == pl.UInt32


def test_date_dispatch():
    """Verify that to_physical_repr correctly handles Dates (Int32)."""
    df = pl.DataFrame(
        {
            "start": [date(2025, 1, 1), date(2025, 1, 1)],
            "end": [date(2025, 1, 2), date(2025, 1, 2)],
        }
    )

    res = df.select(pl_alg.sweep_line_assignment("start", "end", overlapping=False))
    assert res.to_series().to_list() == [1, 2]


def test_hard_interval_case_python():
    """A more complex case to ensure room recycling logic works."""
    # 1. 1-9 (R1)
    # 2. 2-9 (R2)
    # 3. 3-9 (R3)
    # 4. 10-19 (R1 reused)
    # 5. 5-35 (R4)
    starts = [1, 2, 3, 10, 5]
    ends = [9, 9, 9, 19, 35]

    df = pl.DataFrame({"s": starts, "e": ends})
    res = df.select(pl_alg.sweep_line_assignment("s", "e", overlapping=False))

    assert res.to_series().to_list() == [1, 2, 3, 1, 4]


def test_error_on_invalid_intervals():
    """Check that Rust's error propagates as a ComputeError."""
    df = pl.DataFrame({"s": [10], "e": [5]})

    with pytest.raises(ComputeError, match="End time before start time"):
        df.select(pl_alg.sweep_line_assignment("s", "e"))


def test_apply_over_groups():
    """Test that the algorithm works correctly when applied over groups."""
    df = pl.DataFrame(
        {
            "group_key": [1, 1, 2, 2, 3],
            "start": [1, 2, 3, 10, 5],
            "end": [9, 9, 9, 19, 35],
        }
    )

    res = df.with_columns(
        room_id=pl_alg.sweep_line_assignment("start", "end").over("group_key")
    )

    assert res["room_id"].to_list() == [1, 2, 1, 1, 1]


def test_huge_polars_native():
    """Test that the algorithm works correctly on a large dataset."""
    N_TOTAL = 500_000
    N_GROUPS = 10
    K_PEAK = 50  # Concurrency depth
    rows_per_group = N_TOTAL // N_GROUPS

    # 1. Create a base sequence [0, 1, 2, ..., rows_per_group-1]
    # 2. Duplicate it for each group
    # 3. Add a group_key
    df = (
        pl.select(group_key=pl.int_range(0, N_GROUPS, dtype=pl.UInt32))
        .join(
            pl.select(pl.int_range(0, rows_per_group, dtype=pl.UInt32).alias("start")),
            how="cross",
        )
        .with_columns(end=pl.col("start") + K_PEAK)
    )

    # Run the plugin
    res = df.with_columns(
        room_id=pl_alg.sweep_line_assignment("start", "end", overlapping=False).over(
            "group_key"
        )
    )

    # === Validation ===
    # Because we used a shifting block, every group's IDs MUST be 1, 2, ..., 50, 1, 2...
    # We can validate the entire 2 million rows by checking if (index % K) + 1 == room_id

    # We'll use a row_number() per group to check the mathematical pattern
    error_rows = res.filter(pl.col("room_id") != (pl.col("start") % K_PEAK) + 1)

    if error_rows.is_empty():
        print(f"✅ PERFECT: All {N_TOTAL:_} rows match the predicted pattern.")
    else:
        print(f"❌ FAILED: {error_rows.height} rows do not match the pattern.")
        print(error_rows.head())
