//! Sweep-line algorithm for interval scheduling and resource assignment.
//!
//! Solves the problem of assigning the minimum number of resources (rooms, seats, etc.)
//! to a set of intervals such that no two overlapping intervals share the same resource.
//!
//! Algorithm: O(n log n) time, O(n) space
//! - Create events for interval starts/ends
//! - Sort events by time
//! - Process events maintaining a pool of available resource IDs
//! - Assign greedily: use lowest available ID or allocate new one

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// Configuration for sweep-line algorithm.
#[derive(Deserialize)]
pub struct SweepLineKwargs {
    /// If false: intervals [start, end) - endpoints touching don't conflict
    /// If true: intervals [start, end] - endpoints touching do conflict
    pub overlapping: bool,
}

/// Core algorithm for assigning resources to intervals.
///
/// Generic over numeric type T to handle different bit-widths (8/16/32/64).
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
    let (arrival_type, departure_type) = if overlapping { (0i8, 1i8) } else { (1i8, 0i8) };

    // Create events: (time, event_type, interval_index)
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

    // Sort by time, then event type (controls overlap semantics), then index
    events.sort_unstable();

    // Process events
    let mut assignments = vec![0u32; n];
    let mut free_rooms = BinaryHeap::new();
    let mut max_id = 0u32;

    for (_, event_type, idx) in events {
        if event_type == arrival_type {
            // Arrival: assign lowest available room or allocate new one
            let id = free_rooms.pop().map(|Reverse(r)| r).unwrap_or_else(|| {
                max_id += 1;
                max_id
            });
            assignments[idx] = id;
        } else {
            // Departure: return room to pool
            free_rooms.push(Reverse(assignments[idx]));
        }
    }
    Ok(assignments)
}

/// Polars expression plugin for interval-to-resource assignment.
///
/// # Arguments
/// - `inputs[0]`: start times (Series)
/// - `inputs[1]`: end times (Series)
/// - `kwargs.overlapping`: interval semantics
///
/// # Returns
/// `UInt32` Series with resource IDs (1-indexed)
///
/// # Errors
/// - If inputs.len() != 2
/// - If types don't match
/// - If type not supported
/// - If any end < start
#[polars_expr(output_type = UInt32)]
pub fn sweep_line_assignment(inputs: &[Series], kwargs: SweepLineKwargs) -> PolarsResult<Series> {
    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError(
            "Required 2 arguments (start, end)".into(),
        ));
    }

    let s_start = inputs[0].rechunk();
    let s_end = inputs[1].rechunk();

    // Convert to physical representation (handles Date/Datetime)
    let p_start = s_start.to_physical_repr();
    let p_end = s_end.to_physical_repr();

    if p_start.dtype() != p_end.dtype() {
        return Err(PolarsError::ComputeError(
            "Physical dtypes must match".into(),
        ));
    }

    // Dispatch to generic implementation based on physical type
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
        let start = Int64Chunked::from_slice(PlSmallStr::EMPTY, &[10, 20]);
        let end = Int64Chunked::from_slice(PlSmallStr::EMPTY, &[20, 30]);

        // Non-overlapping: [10, 20) and [20, 30) share room
        let res_f = assign(&start, &end, false).unwrap();
        assert_eq!(res_f, vec![1, 1]);

        // Overlapping: [10, 20] and [20, 30] need different rooms
        let res_t = assign(&start, &end, true).unwrap();
        assert_eq!(res_t, vec![1, 2]);
    }

    #[test]
    fn test_datetime_to_physical() {
        let s_start = Series::new(PlSmallStr::from_static("s"), &[1735689600000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();
        let s_end = Series::new(PlSmallStr::from_static("e"), &[1735689601000i64])
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap();

        let p_start = s_start.to_physical_repr();
        let p_end = s_end.to_physical_repr();

        let ca_start = p_start.i64().unwrap();
        let ca_end = p_end.i64().unwrap();

        let res = assign(ca_start, ca_end, false).unwrap();
        assert_eq!(res, vec![1]);
    }

    #[test]
    fn test_hard_interval_case() {
        let starts = vec![1, 2, 3, 10, 11, 12, 20, 21, 22, 5, 15, 25, 30, 31, 32];
        let ends = vec![9, 9, 9, 19, 19, 19, 29, 29, 29, 35, 35, 35, 40, 40, 40];

        let ca_start = UInt32Chunked::from_slice(PlSmallStr::EMPTY, &starts);
        let ca_end = UInt32Chunked::from_slice(PlSmallStr::EMPTY, &ends);

        let res = assign(&ca_start, &ca_end, false).unwrap();
        assert_eq!(res.len(), 15);

        // Verify: no two overlapping intervals share the same room
        for i in 0..15 {
            for j in i + 1..15 {
                if res[i] == res[j] {
                    let overlap = starts[i].max(starts[j]) < ends[i].min(ends[j]);
                    assert!(!overlap, "Overlapping intervals share room {}", res[i]);
                }
            }
        }

        // Verify efficiency
        let unique_rooms: std::collections::HashSet<_> = res.iter().collect();
        assert!(unique_rooms.len() <= 6, "Too many rooms allocated");
    }
}
