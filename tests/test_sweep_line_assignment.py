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
