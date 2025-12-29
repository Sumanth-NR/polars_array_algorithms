# Polars Array Algorithms

High-performance array algorithms for Polars, implemented in Rust with Python bindings.

## Features

- **Sweep-line algorithm** for interval scheduling and resource assignment
- Optimized Rust implementation with PyO3 bindings
- Full type hints and comprehensive tests
- O(n log n) time complexity

## Installation

```bash
pip install -e .
```

For optimized release build:
```bash
make install-release
```

## Quick Start

```python
import polars as pl
import polars_array_algorithms as pl_alg

df = pl.DataFrame({
    "start": [10, 20, 15, 30],
    "end": [25, 35, 28, 40],
})

result = df.select(
    "start", "end",
    room_id=pl_alg.sweep_line_assignment("start", "end", overlapping=False)
)
print(result)
```

Output:
```
shape: (4, 3)
┌───────┬─────┬─────────┐
│ start ┆ end ┆ room_id │
│ ---   ┆ --- ┆ ---     │
│ i64   ┆ i64 ┆ u32     │
╞═══════╪═════╪═════════╡
│ 10    ┆ 25  ┆ 1       │
│ 20    ┆ 35  ┆ 2       │
│ 15    ┆ 28  ┆ 2       │
│ 30    ┆ 40  ┆ 1       │
└───────┴─────┴─────────┘
```

## API Reference

### `sweep_line_assignment(start, end, overlapping=False) -> Expr`

Assigns the minimum number of resources to intervals using a sweep-line algorithm.

**Parameters:**
- `start`: Start times (expression, column name, or Series)
- `end`: End times (same type as start)
- `overlapping`: If False (default), intervals [start, end) can share resources if endpoints touch. If True, intervals [start, end] need separate resources if endpoints touch.

**Returns:** UInt32 Series with resource IDs (1-indexed)

**Supported types:** Any type which can be physically represented as a signed or unsigned integer.
(int8-int64, uint8-uint64, Date, Datetime)

**Complexity:** O(n log n) time, O(n) space

**Example:**
```python
import polars as pl
import polars_array_algorithms as pl_alg

# Non-overlapping intervals
df = pl.DataFrame({
    "start": [10, 20],
    "end": [20, 30],
})
df.select(room=pl_alg.sweep_line_assignment("start", "end", overlapping=False))
# Returns room=[1, 1] - both intervals can share the same room

# Overlapping intervals
df.select(room=pl_alg.sweep_line_assignment("start", "end", overlapping=True))
# Returns room=[1, 2] - intervals at same endpoint need different rooms
```

## Development

### Setup
```bash
make venv
make install
```

### Testing
```bash
make test
```

### Code Quality
```bash
make pre-commit
```

### Building
```bash
make install          # Debug build
make install-release  # Optimized build
```

## Project Structure

```
src/
├── lib.rs              # PyO3 module definition
└── sweep_line.rs       # Algorithm implementation

polars_array_algorithms/
├── __init__.py         # Python API
├── typing.py           # Type definitions
└── _internal.pyi       # Type stubs

tests/                  # Pytest tests
Makefile                # Build commands
```

## Algorithm Details

### Sweep-Line Algorithm

Solves the interval scheduling problem by finding the minimum number of resources needed so no two overlapping intervals share the same resource.

**How it works:**
1. Create events for interval starts and ends
2. Sort events by time (departures before arrivals in non-overlapping mode)
3. Process events, assigning lowest available resource ID or allocating new one
4. Return resource to pool when interval ends

**Tie-breaking:**
- Non-overlapping: departures at time T < arrivals at time T (allows reuse)
- Overlapping: arrivals at time T < departures at time T (requires separate resources)

## Performance

| Metric | Value |
|--------|-------|
| Time Complexity | O(n log n) |
| Space Complexity | O(n) |
| Optimality | Always minimum resources |

## Requirements

- Python >= 3.8
- Polars >= 0.52.0
- Rust (for building)

## License

MIT License - see [LICENSE](LICENSE) file for details.
