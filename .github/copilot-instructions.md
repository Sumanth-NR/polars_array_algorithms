# Polars Array Algorithms - Copilot Instructions

## Project Overview
This is a Rust + Python hybrid project implementing additional array algorithms for the Polars library. The project uses:
- **Rust**: Core algorithm implementations with PyO3 bindings via `pyo3-polars`
- **Python**: Plugin registration, type stubs, and tests using Pytest
- **Build System**: Maturin for compiling Rust to Python extension modules

## Architecture

### Rust Side (`src/`)
- **lib.rs**: Module root and PyO3 module definition with `#[pymodule]` decorator
- **expressions.rs**: Custom expression implementations using `#[polars_expr]` macro
  - Functions must accept `&[Series]` and return `PolarsResult<Series>`
  - Use `apply_into_string_amortized()` for string operations and similar chunked array methods for performance

### Python Side (`polars_array_algorithms/`)
- **__init__.py**: Plugin registration via `register_plugin_function()`
- **_internal.pyi**: Type stubs for Rust-compiled module
- **typing.py**: Custom type definitions like `IntoExprColumn`

### Testing & CI
- **tests/**: Pytest-based tests for all expressions
- **Makefile**: Build, install, and testing commands
- Pre-commit checks: `cargo fmt`, `cargo clippy`, `ruff`, `mypy`

## Development Guidelines

### Adding New Expressions
1. **Rust Implementation** (`src/expressions.rs`):
   - Use `#[polars_expr(output_type=<Type>)]` macro
   - Handle chunked arrays efficiently (avoid unnecessary allocations)
   - Use Polars' built-in methods for array operations

2. **Python Wrapper** (`polars_array_algorithms/__init__.py`):
   - Create a function that calls `register_plugin_function()`
   - Set `is_elementwise=True` for element-wise operations
   - Update type hints using `IntoExprColumn`

3. **Type Stub** (`polars_array_algorithms/_internal.pyi`):
   - Add function signature for the compiled Rust function

4. **Tests** (`tests/`):
   - Create test file following `test_<expression_name>.py` pattern
   - Use Pytest with Polars DataFrames for validation

### Code Quality
- Run `make pre-commit` before committing
- Rust formatting: `cargo +nightly fmt --all`
- Linting: `cargo clippy --all-features`
- Python formatting: `ruff` (enforces consistent style)
- Type checking: `mypy` (strict typing for Python)

### Building & Running
- **Development**: `make install` (builds with debug symbols)
- **Release**: `make install-release` (optimized build)
- **Tests**: `make test` (runs all Pytest tests)
- **Full run**: `make run` or `make run-release`

## Important Constraints

### 🚫 STRICTLY DO NOT CREATE NEW FILES
- **Do NOT create new files or directories unless explicitly prompted by the user**
- This includes:
  - Documentation files (Markdown, RST, etc.)
  - Configuration files
  - Test files
  - Source files
  - Any other files not explicitly requested
- Only modify existing files or create files when the user specifically asks for it

## Best Practices

- Keep algorithm implementations focused and modular
- Leverage Polars' chunked array API for performance
- Add comprehensive tests for edge cases
- Use descriptive variable names and comments for complex logic
- Document performance considerations for algorithms
- Maintain compatibility with Polars >= 0.52.0 and Python >= 3.8

## Common Tasks

| Task | Command |
|------|---------|
| Setup environment | `make venv && make install` |
| Run all checks | `make pre-commit` |
| Run tests | `make test` |
| Rebuild after changes | `make install` |
| Release build | `make install-release && make run-release` |

## Dependencies
- **Rust**: pyo3 (0.26.0), pyo3-polars (0.25.0), polars (0.52.0)
- **Python**: polars, maturin, pytest, ruff, mypy