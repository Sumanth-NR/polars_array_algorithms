from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_array_algorithms._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_array_algorithms.typing import IntoExprColumn

LIB = Path(__file__).parent


def assign_rooms(
    start: IntoExprColumn,
    end: IntoExprColumn,
    overlapping: bool = False,
) -> pl.Expr:
    """
    Assigns the minimum number of resources (rooms/seats) using a sweep-line algorithm.

    Parameters
    ----------
    start : IntoExprColumn
        The start times/values of the intervals.
    end : IntoExprColumn
        The end times/values of the intervals.
    overlapping : bool
        If False (default), intervals are [start, end). Touching endpoints do NOT conflict.
        If True, intervals are [start, end]. Touching endpoints DO conflict.
    """
    return register_plugin_function(
        args=[start, end],
        plugin_path=LIB,
        function_name="sweep_line_assignment",
        kwargs={"overlapping": overlapping},
        is_elementwise=False,
    )
