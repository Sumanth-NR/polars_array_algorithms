"""Tests for longest_increasing_subsequence expression."""

import polars as pl

import polars_array_algorithms as pl_alg


class TestBasicFunctionality:
    """Tests for basic LIS computation."""

    def test_basic_computation(self):
        """Test basic LIS computation with integers."""
        df = pl.DataFrame({"values": [1, 3, 2, 4, 5]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 4, 5]

    def test_already_sorted(self):
        """Test when input is already sorted."""
        df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        """Test when input is reverse sorted."""
        df = pl.DataFrame({"values": [5, 4, 3, 2, 1]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert len(result) == 1
        assert result["lis"].to_list()[0] in [1, 2, 3, 4, 5]

    def test_single_element(self):
        """Test with single element."""
        df = pl.DataFrame({"values": [42]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [42]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty input."""
        df = pl.DataFrame({"values": pl.Series([], dtype=pl.Int64)})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert len(result) == 0

    def test_all_nulls(self):
        """Test with all null values."""
        df = pl.DataFrame({"values": [None, None, None]}, schema={"values": pl.Int32})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert len(result) == 0

    def test_with_nulls(self):
        """Test that null values are properly skipped."""
        df = pl.DataFrame({"values": [1, None, 3, 2, 4]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 4]

    def test_with_duplicates(self):
        """Test that duplicates are handled correctly (strictly increasing)."""
        df = pl.DataFrame({"values": [1, 2, 2, 3, 3, 3, 4]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 3, 4]


class TestDifferentPatterns:
    """Tests for different input patterns and distributions."""

    def test_with_negative_numbers(self):
        """Test LIS with negative numbers."""
        df = pl.DataFrame({"values": [-5, -3, -4, 1, 2, 0, 3]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        lis = result["lis"].to_list()

        assert len(lis) == 5
        for i in range(len(lis) - 1):
            assert lis[i] < lis[i + 1]
        assert lis[0] == -5
        assert lis[-1] == 3

    def test_complex_zigzag(self):
        """Test a more complex pattern."""
        df = pl.DataFrame({"values": [1, 3, 2, 4, 5, 3, 6, 7]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 4, 5, 6, 7]

    def test_complex_random_pattern(self):
        """Test a complex random-like pattern."""
        df = pl.DataFrame(
            {"values": [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]}
        )
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        lis = result["lis"].to_list()

        assert len(lis) >= 6
        for i in range(len(lis) - 1):
            assert lis[i] < lis[i + 1]


class TestDataTypes:
    """Tests for different numeric data types."""

    def test_int64_dtype(self):
        """Test with int64 dtype (default)."""
        df = pl.DataFrame({"values": [1, 3, 2, 4, 5]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].dtype == pl.Int64

    def test_uint32_dtype(self):
        """Test with unsigned integer dtype."""
        df = pl.DataFrame({"values": pl.Series([1, 3, 2, 4, 5], dtype=pl.UInt32)})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 4, 5]
        assert result["lis"].dtype == pl.UInt32

    def test_float32_dtype(self):
        """Test with float32 dtype."""
        df = pl.DataFrame({"values": pl.Series([1.5, 2.5, 0.5, 3.5], dtype=pl.Float32)})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert len(result) == 3
        assert result["lis"].dtype == pl.Float32

    def test_float64_dtype(self):
        """Test with float64 dtype."""
        df = pl.DataFrame({"values": [1.5, 2.5, 0.5, 3.5, 2.0]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        lis = result["lis"].to_list()
        for i in range(len(lis) - 1):
            assert lis[i] < lis[i + 1]


class TestInterfaceAndIntegration:
    """Tests for different interface usage patterns."""

    def test_with_expression(self):
        """Test using the function with Polars expressions."""
        df = pl.DataFrame({"values": [1, 3, 2, 4, 5]})
        result = df.select(lis=pl_alg.longest_increasing_subsequence(pl.col("values")))
        assert result["lis"].to_list() == [1, 2, 4, 5]

    def test_with_series(self):
        """Test using the function with a Series input via DataFrame."""
        series = pl.Series("values", [1, 3, 2, 4, 5])
        df = pl.DataFrame(series)
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result["lis"].to_list() == [1, 2, 4, 5]

    def test_return_type_matches_input(self):
        """Test that return type matches input type."""
        df_i64 = pl.DataFrame({"values": [1, 3, 2, 4]})
        result_i64 = df_i64.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result_i64["lis"].dtype == pl.Int64

        df_f64 = pl.DataFrame({"values": [1.0, 3.0, 2.0, 4.0]})
        result_f64 = df_f64.select(lis=pl_alg.longest_increasing_subsequence("values"))
        assert result_f64["lis"].dtype == pl.Float64


class TestLargeInputs:
    """Tests for performance with larger inputs."""

    def test_large_array(self):
        """Test with a larger array."""
        values = []
        for i in range(500):
            values.append(i)
            values.append(i)
            values.append(i + 1)

        df = pl.DataFrame({"values": values})
        result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
        lis = result["lis"].to_list()

        assert len(lis) > 400
        for i in range(len(lis) - 1):
            assert lis[i] < lis[i + 1]


class TestStrictlyIncreasing:
    """Tests that verify the strictly increasing property."""

    def test_output_is_strictly_increasing(self):
        """Verify that output is always strictly increasing."""
        test_cases = [
            [1, 3, 2, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1],
            [10, 20, 15, 25, 5, 30],
            [-5, 0, 5, -3, 2, 8],
        ]

        for values in test_cases:
            df = pl.DataFrame({"values": values})
            result = df.select(lis=pl_alg.longest_increasing_subsequence("values"))
            lis = result["lis"].to_list()

            for i in range(len(lis) - 1):
                assert lis[i] < lis[i + 1], f"Not strictly increasing in {lis}"
