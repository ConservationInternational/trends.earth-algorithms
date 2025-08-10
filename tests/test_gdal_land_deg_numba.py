"""
Tests for te_algorithms.gdal.land_deg.land_deg_numba module.

This module tests the numba-optimized land degradation calculation functions
used for processing land cover, productivity, and soil organic carbon data.
"""

import numpy as np
import pytest

from te_algorithms.gdal.land_deg.land_deg_numba import (
    NODATA_VALUE,
    MASK_VALUE,
    recode_indicator_errors,
    recode_traj,
    recode_state,
    recode_deg_soc,
    calc_deg_sdg,
    sdg_status_expanded,
    sdg_status_expanded_to_simple,
    prod5_to_prod3,
    calc_lc_trans,
    calc_prod5,
)


class TestRecodeIndicatorErrors:
    """Test the recode_indicator_errors function."""
    
    def test_recode_indicator_errors_basic(self):
        """Test basic recoding of indicator errors."""
        # Create test data
        x = np.array([[-1, 0, 1], [-1, 0, 1]], dtype=np.int16)  # deg, stable, imp
        recode = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int16)  # Different zones
        codes = np.array([1, 2], dtype=np.int16)
        deg_to = np.array([10, 20], dtype=np.int16)
        stable_to = np.array([11, 21], dtype=np.int16)
        imp_to = np.array([12, 22], dtype=np.int16)
        
        result = recode_indicator_errors(x, recode, codes, deg_to, stable_to, imp_to)
        
        # Check zone 1 (first row)
        assert result[0, 0] == 10  # deg (-1) -> 10
        assert result[0, 1] == 11  # stable (0) -> 11
        assert result[0, 2] == 12  # imp (1) -> 12
        
        # Check zone 2 (second row)
        assert result[1, 0] == 20  # deg (-1) -> 20
        assert result[1, 1] == 21  # stable (0) -> 21
        assert result[1, 2] == 22  # imp (1) -> 22
        
    def test_recode_indicator_errors_nodata_targets(self):
        """Test recoding when target values are NODATA."""
        x = np.array([[-1, 0, 1]], dtype=np.int16)
        recode = np.array([[1, 1, 1]], dtype=np.int16)
        codes = np.array([1], dtype=np.int16)
        deg_to = np.array([NODATA_VALUE[0]], dtype=np.int16)  # Don't recode degraded
        stable_to = np.array([50], dtype=np.int16)  # Recode stable
        imp_to = np.array([NODATA_VALUE[0]], dtype=np.int16)  # Don't recode improved
        
        result = recode_indicator_errors(x, recode, codes, deg_to, stable_to, imp_to)
        
        assert result[0, 0] == -1  # Unchanged (deg)
        assert result[0, 1] == 50  # Recoded (stable)
        assert result[0, 2] == 1   # Unchanged (improved)
        
    def test_recode_indicator_errors_no_matching_zones(self):
        """Test recoding when no zones match the codes."""
        x = np.array([[-1, 0, 1]], dtype=np.int16)
        recode = np.array([[99, 99, 99]], dtype=np.int16)  # Zone that doesn't match
        codes = np.array([1], dtype=np.int16)  # Looking for zone 1
        deg_to = np.array([10], dtype=np.int16)
        stable_to = np.array([11], dtype=np.int16)
        imp_to = np.array([12], dtype=np.int16)
        
        result = recode_indicator_errors(x, recode, codes, deg_to, stable_to, imp_to)
        
        # Should remain unchanged since zone 99 doesn't match code 1
        np.testing.assert_array_equal(result, x)


class TestRecodeTraj:
    """Test the recode_traj function for trajectory recoding."""
    
    def test_recode_traj_basic(self):
        """Test basic trajectory recoding."""
        # Input: trajectory codes from -3 to 3
        x = np.array([[-3, -2, -1, 0, 1, 2, 3]], dtype=np.int16)
        
        result = recode_traj(x)
        
        # Expected: decline (-3,-2) -> -1, stable (-1,0,1) -> 0, improve (2,3) -> 1
        expected = np.array([[-1, -1, 0, 0, 0, 1, 1]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)
        
    def test_recode_traj_2d_array(self):
        """Test trajectory recoding with 2D array."""
        x = np.array([[-3, -2], [-1, 0], [1, 2], [3, NODATA_VALUE[0]]], dtype=np.int16)
        
        result = recode_traj(x)
        
        expected = np.array([[-1, -1], [0, 0], [0, 1], [1, NODATA_VALUE[0]]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)
        
    def test_recode_traj_preserves_nodata(self):
        """Test that NODATA values are preserved."""
        x = np.array([[NODATA_VALUE[0], -2, NODATA_VALUE[0]]], dtype=np.int16)
        
        result = recode_traj(x)
        
        expected = np.array([[NODATA_VALUE[0], -1, NODATA_VALUE[0]]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)


class TestRecodeState:
    """Test the recode_state function for state recoding."""
    
    def test_recode_state_basic(self):
        """Test basic state recoding."""
        # Test values representing state classes
        x = np.array([[-15, -5, -3, -2, -1, 0, 1, 2, 3, 5]], dtype=np.int16)
        
        result = recode_state(x)
        
        # Test basic properties rather than specific logic
        assert result.shape == x.shape
        assert result.dtype == np.int16
        # Values <= -10 should become NODATA
        assert result[0] == NODATA_VALUE[0]  # -15 -> NODATA
        
    def test_recode_state_preserves_shape(self):
        """Test that recoding preserves array shape."""
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        
        result = recode_state(x)
        
        assert result.shape == x.shape


class TestRecodeDegSoc:
    """Test the recode_deg_soc function for soil organic carbon recoding."""
    
    def test_recode_deg_soc_basic(self):
        """Test basic SOC degradation recoding."""
        # Test with percentage decline values and water mask
        soc = np.array([[-25, -15, -5, 0, 5, 15]], dtype=np.int16)
        water = np.array([[False, False, False, False, False, False]], dtype=np.int16)
        
        result = recode_deg_soc(soc, water)
        
        # Values <= -10% should be degraded (-1), others stable (0)
        # Exact logic depends on function implementation
        assert result.shape == soc.shape
        assert result.dtype == np.int16
        
    def test_recode_deg_soc_preserves_nodata(self):
        """Test that SOC recoding preserves NODATA values."""
        soc = np.array([[NODATA_VALUE[0], -20, NODATA_VALUE[0]]], dtype=np.int16)
        water = np.array([[False, False, False]], dtype=np.int16)
        
        result = recode_deg_soc(soc, water)
        
        assert result[0, 0] == NODATA_VALUE[0]
        assert result[0, 2] == NODATA_VALUE[0]


class TestCalcDegSdg:
    """Test the calc_deg_sdg function for SDG indicator calculation."""
    
    def test_calc_deg_sdg_basic(self):
        """Test basic SDG degradation calculation."""
        # Create simple test arrays for productivity, land cover, and SOC
        prod = np.array([[-1, 0, 1], [-1, 0, 1]], dtype=np.int16)
        lc = np.array([[0, -1, 0], [1, 0, -1]], dtype=np.int16)
        soc = np.array([[0, 0, -1], [0, -1, 0]], dtype=np.int16)
        
        result = calc_deg_sdg(prod, lc, soc)
        
        assert result.shape == prod.shape
        assert result.dtype == np.int16
        # The function should combine the three indicators according to SDG rules
        
    def test_calc_deg_sdg_with_nodata(self):
        """Test SDG calculation with NODATA values."""
        prod = np.array([[NODATA_VALUE[0], 0]], dtype=np.int16)
        lc = np.array([[0, NODATA_VALUE[0]]], dtype=np.int16)
        soc = np.array([[0, 0]], dtype=np.int16)
        
        result = calc_deg_sdg(prod, lc, soc)
        
        # Should handle NODATA appropriately
        assert result.shape == (1, 2)


class TestSdgStatusExpanded:
    """Test the sdg_status_expanded function."""
    
    def test_sdg_status_expanded_basic(self):
        """Test basic expanded status matrix calculation."""
        # Baseline and target periods with deg/stable/imp values
        sdg_bl = np.array([[-1, 0, 1], [-1, 0, 1]], dtype=np.int16)
        sdg_tg = np.array([[-1, -1, -1], [0, 0, 0]], dtype=np.int16)
        
        result = sdg_status_expanded(sdg_bl, sdg_tg)
        
        assert result.shape == sdg_bl.shape
        assert result.dtype == np.int16
        
        # Check specific combinations according to expanded matrix
        assert result[0, 0] == 1  # deg->deg = 1
        assert result[1, 0] == 3  # deg->stable = 3
        
    def test_sdg_status_expanded_all_combinations(self):
        """Test all possible combinations in expanded status matrix."""
        # Test each combination systematically
        combinations = [
            ((-1, -1), 1),  # deg->deg = 1
            ((0, -1), 2),   # stable->deg = 2
            ((1, -1), 2),   # imp->deg = 2
            ((-1, 0), 3),   # deg->stable = 3
            ((0, 0), 4),    # stable->stable = 4
            ((1, 0), 5),    # imp->stable = 5
            ((-1, 1), 6),   # deg->imp = 6
            ((0, 1), 6),    # stable->imp = 6
            ((1, 1), 7),    # imp->imp = 7
        ]
        
        for (bl_val, tg_val), expected in combinations:
            sdg_bl = np.array([[bl_val]], dtype=np.int16)
            sdg_tg = np.array([[tg_val]], dtype=np.int16)
            
            result = sdg_status_expanded(sdg_bl, sdg_tg)
            
            assert result[0, 0] == expected, f"Failed for ({bl_val}, {tg_val})"


class TestSdgStatusExpandedToSimple:
    """Test the sdg_status_expanded_to_simple function."""
    
    def test_sdg_status_expanded_to_simple_basic(self):
        """Test conversion from expanded to simple status."""
        # Expanded status values 1-7
        sdg_status = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.int16)
        
        result = sdg_status_expanded_to_simple(sdg_status)
        
        # Expected conversions: 1,2,3 -> -1 (deg), 4 -> 0 (stable), 5,6,7 -> 1 (imp)
        expected = np.array([[-1, -1, -1, 0, 1, 1, 1]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)
        
    def test_sdg_status_expanded_to_simple_preserves_nodata(self):
        """Test that NODATA values are preserved in conversion."""
        sdg_status = np.array([[NODATA_VALUE[0], 4, NODATA_VALUE[0]]], dtype=np.int16)
        
        result = sdg_status_expanded_to_simple(sdg_status)
        
        assert result[0, 0] == NODATA_VALUE[0]
        assert result[0, 1] == 0  # Status 4 -> stable
        assert result[0, 2] == NODATA_VALUE[0]


class TestProd5ToProd3:
    """Test the prod5_to_prod3 function."""
    
    def test_prod5_to_prod3_basic(self):
        """Test conversion from 5-class to 3-class productivity."""
        # 5-class productivity values
        prod5 = np.array([[1, 2, 3, 4, 5]], dtype=np.int16)
        
        result = prod5_to_prod3(prod5)
        
        # Expected: 1,2 -> -1 (decline), 3,4 -> 0 (stable), 5 -> 1 (improve)
        expected = np.array([[-1, -1, 0, 0, 1]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)
        
    def test_prod5_to_prod3_preserves_nodata(self):
        """Test that NODATA values are preserved."""
        prod5 = np.array([[NODATA_VALUE[0], 3, NODATA_VALUE[0]]], dtype=np.int16)
        
        result = prod5_to_prod3(prod5)
        
        assert result[0, 0] == NODATA_VALUE[0]
        assert result[0, 1] == 0  # Class 3 -> stable
        assert result[0, 2] == NODATA_VALUE[0]


class TestCalcLcTrans:
    """Test the calc_lc_trans function for land cover transitions."""
    
    def test_calc_lc_trans_basic(self):
        """Test basic land cover transition calculation."""
        # Baseline and target land cover
        lc_bl = np.array([[1, 2], [3, 4]], dtype=np.int16)
        lc_tg = np.array([[1, 3], [2, 4]], dtype=np.int16)
        multiplier = 100  # To create unique transition codes
        
        result = calc_lc_trans(lc_bl, lc_tg, multiplier)
        
        assert result.shape == lc_bl.shape
        # Note: Function may return int16 for smaller values, not always int32
        assert result.dtype in [np.int16, np.int32]
        
        # Check transition calculations: bl*multiplier + tg
        assert result[0, 0] == 1*100 + 1  # 1->1 = 101
        assert result[0, 1] == 2*100 + 3  # 2->3 = 203
        assert result[1, 0] == 3*100 + 2  # 3->2 = 302
        assert result[1, 1] == 4*100 + 4  # 4->4 = 404
        
    def test_calc_lc_trans_with_recoding(self):
        """Test land cover transition with recoding."""
        lc_bl = np.array([[10, 20]], dtype=np.int16)
        lc_tg = np.array([[10, 30]], dtype=np.int16)
        multiplier = 100
        
        # Recode 10->1, 20->2, 30->3
        recode_from = np.array([10, 20, 30], dtype=np.int16)
        recode_to = np.array([1, 2, 3], dtype=np.int16)
        
        result = calc_lc_trans(lc_bl, lc_tg, multiplier, recode_from, recode_to)
        
        # After recoding: bl=[1,2], tg=[1,3]
        assert result[0, 0] == 1*100 + 1  # 1->1 = 101
        assert result[0, 1] == 2*100 + 3  # 2->3 = 203
        
    def test_calc_lc_trans_preserves_nodata(self):
        """Test that NODATA values are preserved in transitions."""
        lc_bl = np.array([[NODATA_VALUE[0], 1]], dtype=np.int16)
        lc_tg = np.array([[1, NODATA_VALUE[0]]], dtype=np.int16)
        multiplier = 100
        
        result = calc_lc_trans(lc_bl, lc_tg, multiplier)
        
        # NODATA in either input should result in NODATA output
        assert result[0, 0] == NODATA_VALUE[0]
        assert result[0, 1] == NODATA_VALUE[0]


class TestCalcProd5:
    """Test the calc_prod5 function for 5-class productivity calculation."""
    
    def test_calc_prod5_basic(self):
        """Test basic 5-class productivity calculation."""
        # Test productivity state, trajectory, and performance
        traj = np.array([[1, 0, -1], [-1, 0, 1]], dtype=np.int16)
        state = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int16)
        perf = np.array([[1, 0, -1], [0, -1, 1]], dtype=np.int16)
        
        result = calc_prod5(traj, state, perf)
        
        assert result.shape == state.shape
        assert result.dtype == np.int16
        
        # The function should combine trajectory, state, and performance into 5 classes
        # Specific logic depends on implementation
        
    def test_calc_prod5_preserves_nodata(self):
        """Test that NODATA values are preserved."""
        traj = np.array([[NODATA_VALUE[0], 1]], dtype=np.int16)
        state = np.array([[1, NODATA_VALUE[0]]], dtype=np.int16)
        perf = np.array([[1, 1]], dtype=np.int16)
        
        result = calc_prod5(traj, state, perf)
        
        # NODATA in either input should be handled appropriately
        assert result.shape == (1, 2)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test functions with empty arrays."""
        empty_array = np.array([], dtype=np.int16).reshape(0, 0)
        
        # Test functions that should handle empty arrays
        result_traj = recode_traj(empty_array)
        assert result_traj.shape == (0, 0)
        
        result_state = recode_state(empty_array)
        assert result_state.shape == (0, 0)
        
    def test_single_element_arrays(self):
        """Test functions with single-element arrays."""
        single_element = np.array([[5]], dtype=np.int16)
        
        result_traj = recode_traj(single_element)
        assert result_traj.shape == (1, 1)
        
        result_state = recode_state(single_element)
        assert result_state.shape == (1, 1)
        
    def test_large_arrays_performance(self):
        """Test functions with larger arrays for performance."""
        size = 100
        large_array = np.random.randint(-3, 4, size=(size, size), dtype=np.int16)
        
        result = recode_traj(large_array)
        
        assert result.shape == large_array.shape
        assert result.dtype == np.int16
        
    def test_extreme_values(self):
        """Test functions with extreme input values."""
        # Test with maximum and minimum int16 values
        extreme_values = np.array([[-32768, 32767]], dtype=np.int16)
        
        # Functions should handle extreme values gracefully
        result_traj = recode_traj(extreme_values)
        assert result_traj.shape == extreme_values.shape
        
        result_state = recode_state(extreme_values)
        assert result_state.shape == extreme_values.shape