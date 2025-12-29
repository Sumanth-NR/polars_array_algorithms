//! Polars Array Algorithms - Rust extension for Polars
//!
//! Provides specialized array algorithms exposed as Polars expression plugins.

use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod sweep_line;

/// PyO3 module definition exposing Rust functions to Python.
#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Global allocator for Polars compatibility.
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
