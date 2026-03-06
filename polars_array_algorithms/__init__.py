"""Polars Array Algorithms - Efficient array algorithms as Polars expressions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_array_algorithms._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_array_algorithms.typing import IntoExprColumn

__all__ = ["sweep_line_assignment", "longest_increasing_subsequence", "__version__"]

LIB = Path(__file__).parent


def sweep_line_assignment(
    start: IntoExprColumn,
    end: IntoExprColumn,
    overlapping: bool = False,
) -> pl.Expr:
    """
    Assign minimum resources (rooms, seats, etc.) to intervals using a sweep-line algorithm.

    Parameters
    ----------
    start
        Start times/values of intervals. Accepts Polars expression, column name, or Series.
    end
        End times/values of intervals. Must have same type as `start`.
    overlapping
        If False (default), intervals [start, end) can share resources if endpoints touch.
        If True, intervals [start, end] need separate resources if endpoints touch.

    Returns
    -------
    Polars expression returning UInt32 Series with room IDs (1-indexed).

    Raises
    ------
    PolarsError
        If types don't match, end < start for any interval, or type is unsupported.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_array_algorithms as pl_alg
    >>> df = pl.DataFrame({
    ...     "start": [10, 20, 15],
    ...     "end": [20, 30, 28],
    ... })
    >>> df.select(
    ...     room=pl_alg.sweep_line_assignment("start", "end", overlapping=False)
    ... )
    shape: (3, 1)
    ┌──────┐
    │ room │
    │ ---  │
    │ u32  │
    ╞══════╡
    │ 1    │
    │ 1    │
    │ 2    │
    └──────┘

    Notes
    -----
    Time complexity: O(n log n) where n is the number of intervals.
    Space complexity: O(n).
    Always produces optimal (minimum) number of resources.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="sweep_line_assignment",
        args=[start, end],
        kwargs={"overlapping": overlapping},
        is_elementwise=False,
    )


def longest_increasing_subsequence(array: IntoExprColumn) -> pl.Expr:
    """
    Find the longest strictly increasing subsequence in an array.

    Parameters
    ----------
    array
        Input array to analyze. Accepts Polars expression, column name, or Series.
        Must be a numeric type (int8-64, uint8-64, float32/64).

    Returns
    -------
    Polars expression returning a Series of the same dtype containing the LIS elements
    in order. Returns empty Series if input is empty or all null.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_array_algorithms as pl_alg
    >>> df = pl.DataFrame({"values": [1, 3, 2, 4, 5]})
    >>> df.select(lis=pl_alg.longest_increasing_subsequence("values"))
    shape: (4, 1)
    ┌─────┐
    │ lis │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 1   │
    │ 2   │
    │ 4   │
    │ 5   │
    └─────┘

    Notes
    -----
    Time complexity: O(n log n). Space complexity: O(n).

    Result is always strictly increasing (no equal consecutive elements).
    Null values in the input are automatically skipped.

    When multiple LIS of the same length exist, the algorithm produces the one where
    the reverse of the subsequence is the lexicographically smallest.
    """
    return register_plugin_function(
        plugin_path=LIB,
        function_name="longest_increasing_subsequence",
        args=[array],
        is_elementwise=False,
        changes_length=True,
    )
