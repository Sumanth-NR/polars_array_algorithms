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
            // (time, event_type, other_time, index)
            events.push((s, arrival_type, e, i));
            events.push((e, departure_type, s, i));
        }
    }

    // Sort by time, then event type (controls overlap semantics), then other_time, then index
    events.sort_unstable();

    // Process events
    let mut assignments = vec![0u32; n];
    let mut free_rooms = BinaryHeap::new();
    let mut max_id = 0u32;

    for (_, event_type, _, idx) in events {
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

    if s_start.dtype() != s_end.dtype() {
        return Err(PolarsError::ComputeError(
            "Types of input sequences must be the same".into(),
        ));
    }

    // Convert to physical representation (handles Date/Datetime)
    let p_start = s_start.to_physical_repr();
    let p_end = s_end.to_physical_repr();

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

    fn brute_overlap_logic(start: &[i64], end: &[i64], overlapping: bool) -> Vec<u32> {
        // 1. Pair starts and ends with their original indices to maintain order
        let mut intervals: Vec<(i64, i64, usize)> = start
            .iter()
            .zip(end.iter())
            .enumerate()
            .map(|(i, (&s, &e))| (s, e, i))
            .collect();

        // 2. Sort by start time (standard sweep-line practice)
        intervals.sort_unstable();

        let mut assigned_room_ids = vec![0; start.len()];
        // Stores the end time of the meeting currently in each room
        // index 0 = Room 1, index 1 = Room 2, etc.
        let mut room_end_times: Vec<i64> = vec![];

        for (s, e, original_idx) in intervals {
            let mut placed = false;

            // 3. Brute force check: can we fit this in an existing room?
            for (i, room_end_time) in room_end_times.iter_mut().enumerate() {
                // Check if interval fits based on overlapping semantics
                let fits = if overlapping {
                    // overlapping=true: intervals [s, e] share room if they don't overlap
                    s > *room_end_time
                } else {
                    // overlapping=false: intervals [s, e) share room if s >= room_end_time
                    s >= *room_end_time
                };

                if fits {
                    *room_end_time = e;
                    assigned_room_ids[original_idx] = (i + 1) as u32;
                    placed = true;
                    break;
                }
            }

            // 4. If no rooms are free, open a new one
            if !placed {
                room_end_times.push(e);
                assigned_room_ids[original_idx] = room_end_times.len() as u32;
            }
        }

        assigned_room_ids
    }

    #[test]
    fn test_same_start_different_end() {
        let starts = vec![1, 1];
        let ends = vec![5, 4];

        let ca_start = Int64Chunked::from_slice(PlSmallStr::EMPTY, &starts);
        let ca_end = Int64Chunked::from_slice(PlSmallStr::EMPTY, &ends);

        let res = assign(&ca_start, &ca_end, false).unwrap();
        let brute = brute_overlap_logic(&starts, &ends, false);

        assert_eq!(res, brute, "Sweep-line vs brute force mismatch");
    }

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

    #[test]
    fn test_hard_interval_case_large_random() {
        use std::collections::HashSet;

        // Seed-based random generation for reproducibility
        let mut rng = 42u64;
        fn next_u64(rng: &mut u64) -> u64 {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *rng
        }

        // Generate 10,000 random intervals
        let n = 10_000;
        let mut starts = Vec::with_capacity(n);
        let mut ends = Vec::with_capacity(n);

        for _ in 0..n {
            let start = (next_u64(&mut rng) % 5000) as i64;
            let duration = (next_u64(&mut rng) % 200) + 1; // 1..=200
            let end = start + duration as i64;
            starts.push(start);
            ends.push(end);
        }

        // Test with overlapping=false
        let ca_start = Int64Chunked::from_slice(PlSmallStr::EMPTY, &starts);
        let ca_end = Int64Chunked::from_slice(PlSmallStr::EMPTY, &ends);

        let res = assign(&ca_start, &ca_end, false).unwrap();
        assert_eq!(res.len(), n);

        // Validate: all room IDs are positive
        assert!(
            res.iter().all(|&id| id > 0),
            "All room IDs should be positive"
        );

        // Validate: no two overlapping intervals share the same room
        for i in 0..n {
            for j in i + 1..n {
                if res[i] == res[j] {
                    // Same room, so intervals must not overlap with overlapping=false semantics
                    // [s1, e1) and [s2, e2) don't overlap if e1 <= s2 or e2 <= s1
                    let overlap = starts[i].max(starts[j]) < ends[i].min(ends[j]);
                    assert!(
                        !overlap,
                        "Overlapping intervals {} and {} share room {}",
                        i, j, res[i]
                    );
                }
            }
        }

        // Verify the brute force logic matches sweep-line for a sample
        let sample_size = 100.min(n);
        let starts_sample: Vec<_> = starts[..sample_size].to_vec();
        let ends_sample: Vec<_> = ends[..sample_size].to_vec();

        let res_sample = assign(
            &Int64Chunked::from_slice(PlSmallStr::EMPTY, &starts_sample),
            &Int64Chunked::from_slice(PlSmallStr::EMPTY, &ends_sample),
            false,
        )
        .unwrap();
        let brute_sample = brute_overlap_logic(&starts_sample, &ends_sample, false);

        assert_eq!(
            res_sample, brute_sample,
            "Sweep-line algorithm should match brute-force logic on sample"
        );

        println!(
            "✅ Hard interval test passed: {} intervals → {} rooms",
            n,
            res.iter().collect::<HashSet<_>>().len()
        );
    }
}
