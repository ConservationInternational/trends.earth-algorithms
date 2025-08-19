"""
Tests for te_algorithms.gdal.util_numba module.

This module tests the numba-optimized utility functions used for
raster data processing and analysis without requiring GDAL installation.
"""

import math

import pytest

# Skip all tests in this module if numpy or te_algorithms.gdal modules are not available
np = pytest.importorskip("numpy")

try:
    from te_algorithms.gdal.util_numba import (
        MASK_VALUE,
        NODATA_VALUE,
        _accumulate_dicts,
        _combine_dicts,
        bizonal_total,
        calc_cell_area,
        cast_numba_int_dict_list_to_cpython,
        cast_numba_int_dict_to_cpython,
        slice_area,
        zonal_total,
        zonal_total_weighted,
    )
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy and GDAL dependencies",
        allow_module_level=True,
    )


class TestSliceArea:
    """Test the slice_area function for calculating Earth surface area."""

    def test_slice_area_equator(self):
        """Test area calculation at the equator."""
        area = slice_area(0.0)  # Equator
        assert area == 0.0

    def test_slice_area_90_degrees(self):
        """Test area calculation at 90 degrees (pole)."""
        area = slice_area(math.pi / 2)  # 90 degrees in radians
        # The actual calculation results in a very large number
        # Test that it's positive and reasonable for a hemisphere
        assert area > 0
        assert area < 1e16  # Reasonable upper bound for Earth's surface area

    def test_slice_area_45_degrees(self):
        """Test area calculation at 45 degrees latitude."""
        area = slice_area(math.pi / 4)  # 45 degrees in radians
        assert area > 0
        assert area < slice_area(math.pi / 2)  # Should be less than pole area

    def test_slice_area_negative_latitude(self):
        """Test area calculation with negative latitude."""
        positive_area = slice_area(math.pi / 4)
        negative_area = slice_area(-math.pi / 4)
        assert negative_area == -positive_area  # Should be symmetric


class TestCellArea:
    """Test the calc_cell_area function for calculating raster cell area."""

    def test_cell_area_basic(self):
        """Test basic cell area calculation."""
        area = calc_cell_area(0.0, 1.0, 1.0)  # 1x1 degree cell at equator
        assert area > 0
        assert area < 15000000000  # Reasonable upper bound in m²

    def test_cell_area_zero_width(self):
        """Test cell area with zero width."""
        area = calc_cell_area(0.0, 1.0, 0.0)
        assert area == 0.0

    def test_cell_area_zero_height(self):
        """Test cell area with zero height."""
        area = calc_cell_area(0.0, 0.0, 1.0)
        assert area == 0.0

    def test_cell_area_swapped_coordinates(self):
        """Test that function handles swapped min/max coordinates."""
        area1 = calc_cell_area(0.0, 1.0, 1.0)
        area2 = calc_cell_area(1.0, 0.0, 1.0)  # Swapped ymin and ymax
        assert area1 == area2  # Should handle the swap internally

    def test_cell_area_different_latitudes(self):
        """Test that cells at different latitudes have different areas."""
        equator_area = calc_cell_area(-0.5, 0.5, 1.0)  # Near equator
        polar_area = calc_cell_area(89.0, 89.5, 1.0)  # Near pole
        assert equator_area > polar_area  # Equatorial cells are larger


class TestZonalTotal:
    """Test the zonal_total function for zonal statistics."""

    def test_zonal_total_basic(self):
        """Test basic zonal total calculation."""
        # Create simple 2x2 arrays
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = zonal_total(zones, data, mask)

        assert result[1] == 30.0  # Zone 1: 10 + 20
        assert result[2] == 70.0  # Zone 2: 30 + 40

    def test_zonal_total_with_mask(self):
        """Test zonal total with masked values."""
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        mask = np.array(
            [[True, False], [False, True]], dtype=bool
        )  # Mask first and last

        result = zonal_total(zones, data, mask)

        assert result[1] == 20.0  # Zone 1: only 20 (10 is masked)
        assert result[2] == 30.0  # Zone 2: only 30 (40 is masked)
        assert MASK_VALUE[0] in result  # Should have masked values

    def test_zonal_total_with_nodata(self):
        """Test zonal total with NODATA values."""
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)
        data = np.array([[10.0, NODATA_VALUE[0]], [30.0, 40.0]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = zonal_total(zones, data, mask)

        assert result[1] == 10.0  # Zone 1: only 10 (NODATA ignored)
        assert result[2] == 70.0  # Zone 2: 30 + 40

    def test_zonal_total_single_zone(self):
        """Test zonal total with single zone."""
        zones = np.array([[5, 5], [5, 5]], dtype=np.int16)
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = zonal_total(zones, data, mask)

        assert result[5] == 10.0  # All values sum to 10
        assert len(result) == 1  # Only one zone


class TestZonalTotalWeighted:
    """Test the zonal_total_weighted function."""

    def test_zonal_total_weighted_basic(self):
        """Test basic weighted zonal total calculation."""
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)
        data = np.array([[10, 20], [30, 40]], dtype=np.int16)
        weights = np.array([[0.5, 1.0], [2.0, 0.25]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = zonal_total_weighted(zones, data, weights, mask)

        expected_zone1 = 10 * 0.5 + 20 * 1.0  # 5 + 20 = 25
        expected_zone2 = 30 * 2.0 + 40 * 0.25  # 60 + 10 = 70

        assert result[1] == expected_zone1
        assert result[2] == expected_zone2

    def test_zonal_total_weighted_zero_weights(self):
        """Test weighted zonal total with zero weights."""
        zones = np.array([[1, 1]], dtype=np.int16)
        data = np.array([[10, 20]], dtype=np.int16)
        weights = np.array([[0.0, 0.0]], dtype=np.float64)
        mask = np.array([[False, False]], dtype=bool)

        result = zonal_total_weighted(zones, data, weights, mask)

        assert result[1] == 0.0  # Zero weights should give zero total


class TestBizonalTotal:
    """Test the bizonal_total function for cross-tabulation."""

    def test_bizonal_total_basic(self):
        """Test basic bizonal total calculation."""
        zones1 = np.array([[1, 1], [2, 2]], dtype=np.int16)
        zones2 = np.array([[10, 20], [10, 20]], dtype=np.int16)
        data = np.array([[5.0, 15.0], [25.0, 35.0]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = bizonal_total(zones1, zones2, data, mask)

        assert result[(1, 10)] == 5.0  # zones1=1, zones2=10
        assert result[(1, 20)] == 15.0  # zones1=1, zones2=20
        assert result[(2, 10)] == 25.0  # zones1=2, zones2=10
        assert result[(2, 20)] == 35.0  # zones1=2, zones2=20

    def test_bizonal_total_overlapping_keys(self):
        """Test bizonal total with overlapping zone combinations."""
        zones1 = np.array([[1, 1], [1, 1]], dtype=np.int16)
        zones2 = np.array([[10, 10], [10, 10]], dtype=np.int16)
        data = np.array([[5.0, 15.0], [25.0, 35.0]], dtype=np.float64)
        mask = np.array([[False, False], [False, False]], dtype=bool)

        result = bizonal_total(zones1, zones2, data, mask)

        assert result[(1, 10)] == 80.0  # All values sum: 5+15+25+35
        assert len(result) == 1  # Only one unique combination


class TestDictionaryUtilities:
    """Test dictionary manipulation utility functions."""

    def test_combine_dicts_basic(self):
        """Test basic dictionary combination."""
        dict1 = {1: 10.0, 2: 20.0}
        dict2 = {2: 5.0, 3: 15.0}

        result = _combine_dicts(dict1, dict2)

        assert result[1] == 10.0  # From dict1 only
        assert result[2] == 25.0  # Combined: 20.0 + 5.0
        assert result[3] == 15.0  # From dict2 only

    def test_combine_dicts_empty(self):
        """Test combining with empty dictionary."""
        dict1 = {1: 10.0, 2: 20.0}
        dict2 = {}

        result = _combine_dicts(dict1, dict2)

        assert result == dict1

    def test_accumulate_dicts_basic(self):
        """Test accumulating multiple dictionaries."""
        dicts = [{1: 10.0, 2: 20.0}, {1: 5.0, 3: 15.0}, {2: 10.0, 3: 5.0}]

        result = _accumulate_dicts(dicts)

        assert result[1] == 15.0  # 10 + 5
        assert result[2] == 30.0  # 20 + 10
        assert result[3] == 20.0  # 15 + 5

    def test_accumulate_dicts_single(self):
        """Test accumulating single dictionary."""
        dicts = [{1: 10.0, 2: 20.0}]

        result = _accumulate_dicts(dicts)

        assert result == dicts[0]


class TestNumbaTypeCasting:
    """Test functions for casting numba types to CPython types."""

    def test_cast_numba_dict_basic(self):
        """Test casting a numba dictionary to CPython types."""
        # Simulate numba dictionary with numpy types
        numba_dict = {np.int32(1): np.float64(10.5), np.int32(2): np.float64(20.5)}

        result = cast_numba_int_dict_to_cpython(numba_dict)

        assert isinstance(result[1], float)
        assert isinstance(result[2], float)
        assert result[1] == 10.5
        assert result[2] == 20.5

    def test_cast_numba_dict_list_basic(self):
        """Test casting a list of numba dictionaries."""
        numba_dicts = [{np.int32(1): np.float64(10.5)}, {np.int32(2): np.float64(20.5)}]

        result = cast_numba_int_dict_list_to_cpython(numba_dicts)

        assert len(result) == 2
        assert isinstance(result[0][1], float)
        assert isinstance(result[1][2], float)
        assert result[0][1] == 10.5
        assert result[1][2] == 20.5

    def test_cast_empty_dict(self):
        """Test casting empty dictionary."""
        result = cast_numba_int_dict_to_cpython({})
        assert result == {}

    def test_cast_empty_dict_list(self):
        """Test casting empty dictionary list."""
        result = cast_numba_int_dict_list_to_cpython([])
        assert result == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_arrays(self):
        """Test with larger arrays to ensure scalability."""
        size = 100
        zones = np.random.randint(1, 10, size=(size, size), dtype=np.int16)
        data = np.random.rand(size, size).astype(np.float64) * 100
        mask = np.random.choice([True, False], size=(size, size), p=[0.1, 0.9])

        result = zonal_total(zones, data, mask)

        # Basic sanity checks
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(k, (int, np.integer)) for k in result.keys())
        assert all(isinstance(v, (float, np.floating)) for v in result.values())

    def test_all_masked_data(self):
        """Test behavior when all data is masked."""
        zones = np.array([[1, 2], [3, 4]], dtype=np.int16)
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        mask = np.array([[True, True], [True, True]], dtype=bool)  # All masked

        result = zonal_total(zones, data, mask)

        # Should only contain the MASK_VALUE
        assert MASK_VALUE[0] in result
        # Original zone values should not be in result
        for zone in [1, 2, 3, 4]:
            assert zone not in result
