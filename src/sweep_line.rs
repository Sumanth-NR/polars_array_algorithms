use std::cmp::Reverse;
use std::collections::BinaryHeap;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct SweepLineKwargs {
    pub overlapping: bool,
}

/// Core algorithm logic.
/// We use a generic T to handle different bit-widths (32/64) of physical data.
fn assign<T>(
    ca_start: &ChunkedArray<T>,
    ca_end: &ChunkedArray<T>,
    overlapping: bool,
) -> PolarsResult<Vec<u32>>
where
    T: PolarsNumericType,
    T::Native: Ord,
{
    let n = ca_start.len();
    // overlapping=False means departure at t < arrival at t (0 < 1), so room is freed first.
    let (arrival_type, departure_type) = if overlapping { (0i8, 1i8) } else { (1i8, 0i8) };

    let mut events = Vec::with_capacity(n * 2);
    for (i, (s_opt, e_opt)) in ca_start.iter().zip(ca_end.iter()).enumerate() {
        if let (Some(s), Some(e)) = (s_opt, e_opt) {
            if e < s {
                return Err(PolarsError::ComputeError(
                    "End time before start time".into(),
                ));
            }
            events.push((s, arrival_type, i));
            events.push((e, departure_type, i));
        }
    }

    // Sort by Time, then Priority (Type), then Row Index
    events.sort_unstable();

    let mut assignments = vec![0u32; n];
    let mut free_rooms = BinaryHeap::new();
    let mut max_id = 0u32;

    for (_, event_type, idx) in events {
        if event_type == arrival_type {
            let id = free_rooms.pop().map(|Reverse(r)| r).unwrap_or_else(|| {
                max_id += 1;
                max_id
            });
            assignments[idx] = id;
        } else {
            free_rooms.push(Reverse(assignments[idx]));
        }
    }
    Ok(assignments)
}

/// The plugin entry point.
/// We mark it as 'pub' and ensure it's in the root of this module.
#[polars_expr(output_type=UInt32)]
pub fn sweep_line_assignment(inputs: &[Series], kwargs: SweepLineKwargs) -> PolarsResult<Series> {
    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError(
            "Required 2 arguments (start, end)".into(),
        ));
    }

    let s_start = inputs[0].rechunk();
    let s_end = inputs[1].rechunk();

    // Map logical (Datetime/Date) to physical (Int64/Int32)
    let p_start = s_start.to_physical_repr();
    let p_end = s_end.to_physical_repr();

    if p_start.dtype() != p_end.dtype() {
        return Err(PolarsError::ComputeError(
            "Physical dtypes must match".into(),
        ));
    }

    let res = match p_start.dtype() {
        DataType::Int64 => assign(p_start.i64()?, p_end.i64()?, kwargs.overlapping)?,
        DataType::Int32 => assign(p_start.i32()?, p_end.i32()?, kwargs.overlapping)?,
        DataType::Int16 => assign(p_start.i16()?, p_end.i16()?, kwargs.overlapping)?,
        DataType::Int8 => assign(p_start.i8()?, p_end.i8()?, kwargs.overlapping)?,
        DataType::UInt64 => assign(p_start.u64()?, p_end.u64()?, kwargs.overlapping)?,
        DataType::UInt32 => assign(p_start.u32()?, p_end.u32()?, kwargs.overlapping)?,
        DataType::UInt16 => assign(p_start.u16()?, p_end.u16()?, kwargs.overlapping)?,
        DataType::UInt8 => assign(p_start.u8()?, p_end.u8()?, kwargs.overlapping)?,
        _ => {
            return Err(PolarsError::ComputeError(
                "Unsupported physical type".into(),
            ))
        },
    };

    let ca = UInt32Chunked::from_vec(PlSmallStr::from_static("room_id"), res);
    Ok(ca.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_logic() {
        // [10, 20] and [20, 30]
        let start = Int64Chunked::from_slice(PlSmallStr::EMPTY, &[10, 20]);
        let end = Int64Chunked::from_slice(PlSmallStr::EMPTY, &[20, 30]);

        // Non-overlapping: reuse room at tick 20
        let res_f = assign(&start, &end, false).unwrap();
        assert_eq!(res_f, vec![1, 1]);

        // Overlapping: need new room at tick 20
        let res_t = assign(&start, &end, true).unwrap();
        assert_eq!(res_t, vec![1, 2]);
    }

    #[test]
    fn test_datetime_to_physical() {
        // Create Datetime Series
        let s_start = Series::new(PlSmallStr::from_static("s"), &[1735689600000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();
        let s_end = Series::new(PlSmallStr::from_static("e"), &[1735689601000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();

        // Manual simulation of the plugin flow:
        // 1. Get physical representation
        let p_start = s_start.to_physical_repr();
        let p_end = s_end.to_physical_repr();

        // 2. Unpack to i64 (this is what Datetime becomes)
        let ca_start = p_start.i64().unwrap();
        let ca_end = p_end.i64().unwrap();

        // 3. Call assign directly
        let res = assign(ca_start, ca_end, false).unwrap();
        assert_eq!(res, vec![1]);
    }

    #[test]
    fn test_hard_interval_case() {
        // 15 intervals with specific patterns to test room reuse efficiency.
        // Pattern: [0, 10], [10, 20], [20, 30] ... (Chainable)
        // Mixed with some long-running intervals that block rooms.
        let starts = vec![1, 2, 3, 10, 11, 12, 20, 21, 22, 5, 15, 25, 30, 31, 32];
        let ends = vec![9, 9, 9, 19, 19, 19, 29, 29, 29, 35, 35, 35, 40, 40, 40];

        let ca_start = UInt32Chunked::from_slice(PlSmallStr::EMPTY, &starts);
        let ca_end = UInt32Chunked::from_slice(PlSmallStr::EMPTY, &ends);

        // We use overlapping=false to see how well we recycle IDs.
        let res = assign(&ca_start, &ca_end, false).unwrap();

        assert_eq!(res.len(), 15);

        // Validation: No two intervals with same room ID should overlap.
        for i in 0..15 {
            for j in i + 1..15 {
                if res[i] == res[j] {
                    // Check for overlap: [s1, e1) and [s2, e2)
                    let overlap = starts[i].max(starts[j]) < ends[i].min(ends[j]);
                    assert!(
                        !overlap,
                        "Room {} shared by overlapping intervals {} and {}",
                        res[i], i, j
                    );
                }
            }
        }

        // Check that we didn't use an excessive number of rooms.
        let unique_rooms: std::collections::HashSet<_> = res.iter().collect();
        assert!(
            unique_rooms.len() <= 6,
            "Inefficient room allocation detected"
        );
    }
}
