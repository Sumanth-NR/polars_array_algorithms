//! Longest Increasing Subsequence (LIS) algorithm.
//!
//! Finds the longest strictly increasing subsequence in an array.
//! Time complexity: O(n log n), Space complexity: O(n).
//!
//! ## Tie-Breaking Behavior
//!
//! When multiple LIS of the same length exist, the algorithm produces the one where
//! the reverse of the subsequence is the lexicographically smallest.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Core algorithm for finding the longest increasing subsequence.
fn find_lis<T>(ca: &ChunkedArray<T>) -> PolarsResult<Vec<T::Native>>
where
    T: PolarsNumericType,
    T::Native: PartialOrd + Copy,
{
    let n = ca.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Collect non-null values
    let values: Vec<T::Native> = ca.iter().flatten().collect();

    if values.is_empty() {
        return Ok(Vec::new());
    }

    // tails[i] = smallest tail value for LIS of length i+1
    let mut tails: Vec<T::Native> = Vec::new();
    // lis_idx[i] = index in 'values' where LIS of length i+1 ends
    let mut lis_idx: Vec<usize> = Vec::new();
    // parent[i] = index in 'values' of previous element in LIS ending at values[i]
    let mut parent: Vec<Option<usize>> = vec![None; values.len()];

    // Process each element
    for (i, &val) in values.iter().enumerate() {
        // Find position where tails[pos] >= val
        let pos = tails.partition_point(|&x| x < val);

        if pos == tails.len() {
            tails.push(val);
            lis_idx.push(i);
        } else {
            tails[pos] = val;
            lis_idx[pos] = i;
        }

        if pos > 0 {
            parent[i] = Some(lis_idx[pos - 1]);
        }
    }

    // Reconstruct LIS by backtracking from the end
    let mut result = Vec::new();
    if let Some(&last_idx) = lis_idx.last() {
        let mut current = Some(last_idx);
        while let Some(idx) = current {
            result.push(values[idx]);
            current = parent[idx];
        }
        result.reverse();
    }

    Ok(result)
}

/// Find the longest strictly increasing subsequence in a Polars Series.
///
/// Returns a Series containing elements from the longest strictly increasing subsequence
/// in the order they appear in the input.
///
/// # Arguments
/// - `inputs[0]`: Series of numeric values (int8-64, uint8-64, float32/64)
///
/// # Returns
/// Series of the same dtype containing the LIS elements in order
///
/// # Examples
///
/// ```text
/// Input:  [1, 3, 2, 4, 5]      → Output: [1, 2, 4, 5]
/// Input:  [5, 4, 3, 2, 1]      → Output: [1] (any single element)
/// Input:  [1, None, 3, 2, 4]   → Output: [1, 2, 4] (nulls skipped)
/// ```
#[polars_expr(output_type = UInt32)]
pub fn longest_increasing_subsequence(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 {
        return Err(PolarsError::ComputeError(
            "longest_increasing_subsequence requires exactly 1 argument".into(),
        ));
    }

    let s = inputs[0].rechunk();
    let dtype = s.dtype();

    let result = match dtype {
        DataType::Int64 => {
            let ca = s.i64()?;
            let lis = find_lis(ca)?;
            Int64Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::Int32 => {
            let ca = s.i32()?;
            let lis = find_lis(ca)?;
            Int32Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::Int16 => {
            let ca = s.i16()?;
            let lis = find_lis(ca)?;
            Int16Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::Int8 => {
            let ca = s.i8()?;
            let lis = find_lis(ca)?;
            Int8Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::UInt64 => {
            let ca = s.u64()?;
            let lis = find_lis(ca)?;
            UInt64Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::UInt32 => {
            let ca = s.u32()?;
            let lis = find_lis(ca)?;
            UInt32Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::UInt16 => {
            let ca = s.u16()?;
            let lis = find_lis(ca)?;
            UInt16Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::UInt8 => {
            let ca = s.u8()?;
            let lis = find_lis(ca)?;
            UInt8Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::Float64 => {
            let ca = s.f64()?;
            let lis = find_lis(ca)?;
            Float64Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        DataType::Float32 => {
            let ca = s.f32()?;
            let lis = find_lis(ca)?;
            Float32Chunked::from_vec(PlSmallStr::EMPTY, lis).into_series()
        },
        _ => {
            return Err(PolarsError::ComputeError(
                format!(
                    "Unsupported dtype for longest_increasing_subsequence: {:?}",
                    dtype
                )
                .into(),
            ))
        },
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_lis() {
        let input = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert!(result.len() >= 4);
        for i in 0..result.len() - 1 {
            assert!(result[i] < result[i + 1]);
        }
    }

    #[test]
    fn test_empty_array() {
        let input: Vec<i32> = vec![];
        let ca = Int32Chunked::from_vec(PlSmallStr::EMPTY, input);
        let result = find_lis(&ca).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_element() {
        let input = vec![42];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 42);
    }

    #[test]
    fn test_already_sorted() {
        let input = vec![1, 2, 3, 4, 5];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted() {
        let input = vec![5, 4, 3, 2, 1];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_with_nulls() {
        let input = vec![Some(1), None, Some(3), Some(2), Some(4)];
        let ca = Int32Chunked::from_iter(input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result, vec![1, 2, 4]);
    }

    #[test]
    fn test_duplicates() {
        let input = vec![1, 2, 2, 3, 3, 3, 4];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_complex_case() {
        let input = vec![0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert!(result.len() >= 6);
        for i in 0..result.len() - 1 {
            assert!(result[i] < result[i + 1]);
        }
    }

    #[test]
    fn test_with_negative_numbers() {
        let input = vec![-5, -3, -4, 1, 2, 0, 3];
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result, vec![-5, -4, 1, 2, 3]);
    }

    #[test]
    fn test_float_values() {
        let input = vec![1.5, 2.5, 0.5, 3.5, 2.0];
        let ca = Float64Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_large_input() {
        let mut input = Vec::new();
        for i in 0..100 {
            input.push(i as i32);
            input.push(i as i32);
            input.push((i + 1) as i32);
        }
        let ca = Int32Chunked::from_slice(PlSmallStr::EMPTY, &input);
        let result = find_lis(&ca).unwrap();
        assert!(result.len() >= 90);
        for i in 0..result.len() - 1 {
            assert!(result[i] < result[i + 1]);
        }
    }
}
