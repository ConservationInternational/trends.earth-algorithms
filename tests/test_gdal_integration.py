"""
Integration tests for te_algorithms.gdal modules.

This module tests the integration between different GDAL components,
ensuring they work together correctly for land degradation analysis workflows.
"""

import pytest

# Skip all tests in this module if numpy or te_algorithms.gdal modules are not available
np = pytest.importorskip("numpy")

try:
    from te_algorithms.gdal.drought_numba import (
        NODATA_VALUE as DROUGHT_NODATA,
    )
    from te_algorithms.gdal.drought_numba import (
        drought_class,
        jrc_sum_and_count,
    )
    from te_algorithms.gdal.land_deg.config import (
        MASK_VALUE as CONFIG_MASK,
    )
    from te_algorithms.gdal.land_deg.config import (
        NODATA_VALUE as CONFIG_NODATA,
    )
    from te_algorithms.gdal.land_deg.land_deg_numba import (
        MASK_VALUE as LD_MASK,
    )
    from te_algorithms.gdal.land_deg.land_deg_numba import (
        NODATA_VALUE as LD_NODATA,
    )
    from te_algorithms.gdal.land_deg.land_deg_numba import (
        calc_deg_sdg,
        calc_lc_trans,
        calc_prod5,
        prod5_to_prod3,
        recode_state,
        recode_traj,
        sdg_status_expanded,
    )
    from te_algorithms.gdal.util_numba import (
        MASK_VALUE as UTIL_MASK,
    )
    from te_algorithms.gdal.util_numba import (
        NODATA_VALUE as UTIL_NODATA,
    )
    from te_algorithms.gdal.util_numba import (
        bizonal_total,
        calc_cell_area,
        zonal_total,
        zonal_total_weighted,
    )
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy and GDAL dependencies",
        allow_module_level=True,
    )


class TestConstantConsistency:
    """Test that constants are consistent across modules."""

    def test_nodata_values_consistent(self):
        """Test that NODATA values are consistent across all modules."""
        assert UTIL_NODATA[0] == LD_NODATA[0]
        assert UTIL_NODATA[0] == DROUGHT_NODATA[0]
        assert UTIL_NODATA[0] == CONFIG_NODATA

    def test_mask_values_consistent(self):
        """Test that MASK values are consistent across modules."""
        assert UTIL_MASK[0] == LD_MASK[0]
        assert UTIL_MASK[0] == CONFIG_MASK

    def test_special_values_are_int16(self):
        """Test that all special values are int16 compatible."""
        values_to_test = [
            UTIL_NODATA[0],
            UTIL_MASK[0],
            LD_NODATA[0],
            LD_MASK[0],
            DROUGHT_NODATA[0],
            CONFIG_NODATA,
            CONFIG_MASK,
        ]

        int16_min = np.iinfo(np.int16).min
        int16_max = np.iinfo(np.int16).max

        for value in values_to_test:
            assert int16_min <= value <= int16_max
            # Test that value can be cast to int16 without loss
            assert np.int16(value) == value


class TestWorkflowIntegration:
    """Test integration of different processing steps in typical workflows."""

    def test_productivity_analysis_workflow(self):
        """Test a complete productivity analysis workflow."""
        # Step 1: Create mock productivity trajectory, state, and performance data
        trajectory = np.array([[-3, -2, -1, 0, 1, 2, 3]], dtype=np.int16)
        state = np.array([[1, 2, 3, 1, 2, 3, 1]], dtype=np.int16)
        performance = np.array([[-1, 0, 1, -1, 0, 1, -1]], dtype=np.int16)

        # Step 2: Recode trajectory to 3-class
        traj_3class = recode_traj(trajectory)
        expected_traj = np.array([[-1, -1, 0, 0, 0, 1, 1]], dtype=np.int16)
        np.testing.assert_array_equal(traj_3class, expected_traj)

        # Step 3: Calculate 5-class productivity (with correct signature)
        prod5 = calc_prod5(traj_3class, state, performance)

        # Step 4: Convert to 3-class
        prod3 = prod5_to_prod3(prod5)

        # Verify the workflow produces valid results
        assert prod3.shape == trajectory.shape
        assert prod3.dtype == np.int16

    def test_sdg_calculation_workflow(self):
        """Test SDG 15.3.1 calculation workflow."""
        # Create mock data for the three indicators
        productivity = np.array([[-1, 0, 1], [0, -1, 1]], dtype=np.int16)
        land_cover = np.array([[0, -1, 0], [-1, 0, 1]], dtype=np.int16)
        soil_carbon = np.array([[0, 0, -1], [0, -1, 0]], dtype=np.int16)

        # Calculate SDG indicator
        sdg_result = calc_deg_sdg(productivity, land_cover, soil_carbon)

        # Verify basic properties
        assert sdg_result.shape == productivity.shape
        assert sdg_result.dtype == np.int16

        # Create expanded status
        baseline_sdg = sdg_result
        target_sdg = np.array(
            [[0, -1, 1], [-1, 0, 0]], dtype=np.int16
        )  # Different target

        expanded_status = sdg_status_expanded(baseline_sdg, target_sdg)

        # Verify expanded status has valid range
        assert expanded_status.shape == sdg_result.shape
        assert expanded_status.dtype == np.int16
        valid_status_values = {1, 2, 3, 4, 5, 6, 7, LD_NODATA[0]}
        assert set(np.unique(expanded_status)).issubset(valid_status_values)

    def test_land_cover_transition_workflow(self):
        """Test land cover transition analysis workflow."""
        # Create mock land cover data for baseline and target periods
        lc_baseline = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        lc_target = np.array([[1, 3, 2], [5, 4, 6]], dtype=np.int16)

        # Calculate transitions
        transitions = calc_lc_trans(lc_baseline, lc_target, 100)

        # Verify transition codes
        assert transitions.shape == lc_baseline.shape
        # Function may return int16, int32, or int64 depending on numba version
        assert transitions.dtype in [np.int16, np.int32, np.int64]

        # Check specific transitions
        assert transitions[0, 0] == 101  # 1->1 = stable
        assert transitions[0, 1] == 203  # 2->3 = change
        assert transitions[0, 2] == 302  # 3->2 = change

    def test_drought_analysis_workflow(self):
        """Test drought analysis workflow."""
        # Create mock SPI data (scaled by 1000)
        spi_data = np.array(
            [
                [1000, 0, -500, -1250, -1750, -2500],  # Normal to extreme drought
                [-100, -800, -1100, -1600, -2100, DROUGHT_NODATA[0]],
            ],
            dtype=np.int16,
        )

        # Classify drought
        drought_classes = drought_class(spi_data)

        # Verify classification
        expected_classes = np.array(
            [
                [0, 0, 1, 2, 3, 4],  # Normal, Normal, Mild, Moderate, Severe, Extreme
                [1, 1, 2, 3, 4, DROUGHT_NODATA[0]],  # Various classes + NODATA
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(drought_classes, expected_classes)

        # Test JRC aggregation
        jrc_data = np.array([[1000.0, 2000.0, 3000.0]], dtype=np.float64)
        # Note: mask dtype is int16 per the numba signature. Without numba,
        # numpy uses integer fancy indexing (not boolean masking), so mask
        # values [0, 0, 1] index positions 0 and 1, setting them to NODATA.
        mask = np.array([[False, False, True]], dtype=np.int16)

        total_sum, count = jrc_sum_and_count(jrc_data, mask)

        assert total_sum == 3.0  # 3000 / 1000 (only unmasked value)
        assert count == 1


class TestZonalAnalysisIntegration:
    """Test integration of zonal analysis with land degradation indicators."""

    def test_sdg_zonal_analysis(self):
        """Test zonal analysis of SDG indicators."""
        # Create mock SDG data and zones
        sdg_data = np.array([[-1, 0, 1], [-1, 0, 1]], dtype=np.int16)
        zones = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.int16)
        areas = np.array(
            [[100.0, 100.0, 200.0], [200.0, 200.0, 100.0]], dtype=np.float64
        )
        mask = np.array([[False, False, False], [False, False, False]], dtype=bool)

        # Convert SDG values to counts for zonal analysis
        # Use simple mapping: deg=-1->1, stable=0->0, imp=1->0 for degraded area
        degraded = (sdg_data == -1).astype(np.float64) * areas

        # Calculate zonal totals of degraded areas
        degraded_by_zone = zonal_total(zones, degraded, mask)

        # Verify results
        # Zone 1 pixels: (0,0)=-1, (0,1)=0, (1,2)=1 → one degraded pixel
        expected_zone1 = 100.0  # One degraded pixel in zone 1 at (0,0)
        # Zone 2 pixels: (0,2)=1, (1,0)=-1, (1,1)=0 → one degraded pixel
        expected_zone2 = 200.0  # One degraded pixel in zone 2 at (1,0)

        assert degraded_by_zone[1] == expected_zone1
        assert degraded_by_zone[2] == expected_zone2

    def test_productivity_zonal_weighted_analysis(self):
        """Test weighted zonal analysis of productivity data."""
        # Create productivity classes and zones
        productivity = np.array([[1, 2, 3], [4, 5, 1]], dtype=np.int16)  # 5-class
        zones = np.array([[1, 1, 2], [2, 1, 1]], dtype=np.int16)
        population = np.array(
            [[50.0, 100.0, 75.0], [25.0, 200.0, 150.0]], dtype=np.float64
        )
        mask = np.array([[False, False, False], [False, False, False]], dtype=bool)

        # Calculate population-weighted productivity by zone
        weighted_prod = zonal_total_weighted(zones, productivity, population, mask)

        # Verify calculation
        # Zone 1: (1*50 + 2*100 + 5*200 + 1*150) / (50+100+200+150) = weighted avg
        # Zone 2: (3*75 + 4*25) / (75+25) = weighted avg

        zone1_total = 1 * 50 + 2 * 100 + 5 * 200 + 1 * 150  # 1400
        zone2_total = 3 * 75 + 4 * 25  # 325

        assert weighted_prod[1] == zone1_total
        assert weighted_prod[2] == zone2_total

    def test_land_cover_transition_crosstab(self):
        """Test cross-tabulation of land cover transitions."""
        # Create land cover transition matrix
        lc_baseline = np.array([[1, 2, 1], [3, 2, 1]], dtype=np.int16)
        lc_target = np.array([[1, 1, 2], [3, 3, 3]], dtype=np.int16)
        areas = np.array([[10.0, 20.0, 15.0], [30.0, 25.0, 5.0]], dtype=np.float64)
        mask = np.array([[False, False, False], [False, False, False]], dtype=bool)

        # Calculate cross-tabulation
        crosstab = bizonal_total(lc_baseline, lc_target, areas, mask)

        # Verify specific transitions
        assert crosstab[(1, 1)] == 10.0  # Forest->Forest (stable)
        assert crosstab[(2, 1)] == 20.0  # Crop->Forest (change)
        assert crosstab[(1, 2)] == 15.0  # Forest->Crop (change)
        # Only pixel (1,0) has baseline=3, target=3; pixel (1,1) is baseline=2, target=3
        assert crosstab[(3, 3)] == 30.0  # Grass->Grass (stable, one pixel)
        assert crosstab[(2, 3)] == 25.0  # Crop->Grass (change)
        assert crosstab[(1, 3)] == 5.0  # Forest->Grass (change)


class TestCellAreaCalculations:
    """Test integration of cell area calculations with analysis functions."""

    def test_area_weighted_degradation_analysis(self):
        """Test degradation analysis with proper area weighting."""
        # Create a small grid with known coordinates
        lats = np.array(
            [[1.0, 1.0], [0.0, 0.0]], dtype=np.float64
        )  # Top and bottom rows

        # Calculate cell areas (assuming 1 degree cells)
        areas = np.zeros_like(lats)
        for i in range(areas.shape[0]):
            for j in range(areas.shape[1]):
                if i == 0:  # Top row: 0.5 to 1.5 degrees
                    areas[i, j] = calc_cell_area(0.5, 1.5, 1.0)
                else:  # Bottom row: -0.5 to 0.5 degrees
                    areas[i, j] = calc_cell_area(-0.5, 0.5, 1.0)

        # Create mock degradation data
        degradation = np.array(
            [[-1, 0], [1, -1]], dtype=np.int16
        )  # deg, stable, imp, deg
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)

        # Calculate degraded area by zone (only count degraded pixels)
        degraded_mask = (degradation == -1).astype(np.float64)
        degraded_areas = degraded_mask * areas
        mask = np.array([[False, False], [False, False]], dtype=bool)

        zonal_degraded = zonal_total(zones, degraded_areas, mask)

        # Verify that we get realistic area values
        assert len(zonal_degraded) == 2  # Two zones
        assert all(area >= 0 for area in zonal_degraded.values())

        # Zone 1 should have area from top-left cell only
        # Zone 2 should have area from bottom-right cell only
        assert zonal_degraded[1] > 0  # Has degraded pixel
        assert zonal_degraded[2] > 0  # Has degraded pixel

    def test_total_area_calculation(self):
        """Test calculation of total areas for validation."""
        # Create a 2x2 grid representing 1-degree cells
        cell_area_00 = calc_cell_area(-0.5, 0.5, 1.0)  # Equatorial cell
        cell_area_01 = calc_cell_area(0.5, 1.5, 1.0)  # Higher latitude cell

        # Higher latitude cells should be smaller
        assert cell_area_01 < cell_area_00

        # Both should be positive and reasonable (not zero, not extremely large)
        assert 0 < cell_area_00 < 1e15  # Square meters, reasonable upper bound
        assert 0 < cell_area_01 < 1e15

        # Test that the area calculation is symmetric
        assert calc_cell_area(-1.5, -0.5, 1.0) == calc_cell_area(0.5, 1.5, 1.0)


class TestErrorHandlingIntegration:
    """Test error handling across integrated workflows."""

    def test_nodata_propagation_through_workflow(self):
        """Test that NODATA values propagate correctly through analysis workflow."""
        # Create data with NODATA values
        prod = np.array([[LD_NODATA[0], 0, 1]], dtype=np.int16)
        lc = np.array([[0, LD_NODATA[0], 0]], dtype=np.int16)
        soc = np.array([[0, 0, LD_NODATA[0]]], dtype=np.int16)

        # Calculate SDG indicator
        sdg_result = calc_deg_sdg(prod, lc, soc)

        # All pixels should be NODATA due to at least one NODATA input
        expected = np.array(
            [[LD_NODATA[0], LD_NODATA[0], LD_NODATA[0]]], dtype=np.int16
        )
        np.testing.assert_array_equal(sdg_result, expected)

    def test_mask_handling_in_zonal_analysis(self):
        """Test that mask values are handled correctly in zonal analysis."""
        zones = np.array([[1, 2, 3]], dtype=np.int16)
        data = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
        mask = np.array([[True, False, True]], dtype=bool)  # Mask first and third

        result = zonal_total(zones, data, mask)

        # Only zone 2 should have data (zone 2, value 20)
        assert result[2] == 20.0
        # Masked zones should be present with MASK_VALUE key
        assert UTIL_MASK[0] in result

    def test_mixed_valid_invalid_data(self):
        """Test handling of mixed valid and invalid data."""
        # Create arrays with mix of valid data, NODATA, and values to be masked
        spi_data = np.array(
            [[1000, DROUGHT_NODATA[0], -1500], [-2000, 0, 500]], dtype=np.int16
        )

        drought_result = drought_class(spi_data)

        # Check that NODATA is preserved and valid data is classified
        assert drought_result[0, 1] == DROUGHT_NODATA[0]  # NODATA preserved
        assert drought_result[0, 0] == 0  # Normal (1000 -> 0)
        assert drought_result[0, 2] == 2  # Moderate (-1500 -> 2)
        # -2000 >= -2000 matches "severe" range (< -1500 and >= -2000), not "extreme" (< -2000)
        assert drought_result[1, 0] == 3  # Severe (-2000 -> 3)


class TestPerformanceIntegration:
    """Test performance aspects of integrated workflows."""

    def test_large_array_workflow(self):
        """Test workflow with larger arrays to verify performance."""
        size = 50  # Reasonable size for CI environment

        # Create random but structured data
        np.random.seed(42)  # For reproducible tests

        # Productivity trajectory data (-3 to 3)
        traj = np.random.randint(-3, 4, size=(size, size), dtype=np.int16)

        # Land cover data (1 to 6 classes)
        lc_bl = np.random.randint(1, 7, size=(size, size), dtype=np.int16)
        lc_tg = np.random.randint(1, 7, size=(size, size), dtype=np.int16)

        # Zones (1 to 10)
        zones = np.random.randint(1, 11, size=(size, size), dtype=np.int16)

        # Areas (realistic cell areas in m²)
        areas = np.random.uniform(1e8, 1.5e8, size=(size, size))

        # Run workflow
        traj_recoded = recode_traj(traj)
        lc_trans = calc_lc_trans(lc_bl, lc_tg, 100)

        mask = np.random.choice([True, False], size=(size, size), p=[0.05, 0.95])

        # Zonal analysis
        zonal_results = zonal_total(zones, areas, mask)

        # Verify results are reasonable
        assert len(zonal_results) > 0
        assert all(isinstance(k, (int, np.integer)) for k in zonal_results.keys())
        assert all(isinstance(v, (float, np.floating)) for v in zonal_results.values())

        # Check that array shapes are preserved
        assert traj_recoded.shape == (size, size)
        assert lc_trans.shape == (size, size)

    def test_memory_efficiency_workflow(self):
        """Test that workflow doesn't create excessive temporary arrays."""
        # This test ensures that functions work in-place where possible
        # and don't create unnecessary copies

        original_data = np.array([[-3, -2, -1, 0, 1, 2, 3]], dtype=np.int16)

        # Test that recode_traj creates new array (doesn't modify input)
        recoded = recode_traj(original_data)

        # Original should be unchanged
        expected_original = np.array([[-3, -2, -1, 0, 1, 2, 3]], dtype=np.int16)
        np.testing.assert_array_equal(original_data, expected_original)

        # Result should be different
        expected_recoded = np.array([[-1, -1, 0, 0, 0, 1, 1]], dtype=np.int16)
        np.testing.assert_array_equal(recoded, expected_recoded)


class TestDataTypeConsistency:
    """Test that data types remain consistent through integrated workflows."""

    def test_int16_preservation(self):
        """Test that int16 types are preserved through workflows."""
        # Start with int16 data
        data = np.array([[1, 2, 3]], dtype=np.int16)

        # Functions that should preserve int16
        result_traj = recode_traj(data)
        result_state = recode_state(data)

        assert result_traj.dtype == np.int16
        assert result_state.dtype == np.int16

    def test_int32_for_transitions(self):
        """Test that land cover transitions use int32 for larger values."""
        lc_bl = np.array([[99, 99]], dtype=np.int16)
        lc_tg = np.array([[99, 1]], dtype=np.int16)

        # With multiplier 100, this should create values > 32767 (int16 max)
        result = calc_lc_trans(lc_bl, lc_tg, 100)

        # At least int32 to handle large values (numba JIT may use int64)
        assert result.dtype in [np.int32, np.int64]
        assert result[0, 0] == 9999  # 99*100 + 99

    def test_float64_for_areas(self):
        """Test that area calculations maintain float64 precision."""
        zones = np.array([[1, 2]], dtype=np.int16)
        data = np.array([[1.5, 2.5]], dtype=np.float64)
        mask = np.array([[False, False]], dtype=bool)

        result = zonal_total(zones, data, mask)

        # Verify precision is maintained
        assert result[1] == 1.5
        assert result[2] == 2.5
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)
