"""Type definitions for polars_array_algorithms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys

    import polars as pl

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    from polars.datatypes import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
    PolarsDataType: TypeAlias = DataType | DataTypeClass
