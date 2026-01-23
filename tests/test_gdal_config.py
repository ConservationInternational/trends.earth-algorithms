"""
Tests for te_algorithms.gdal.land_deg.config module.

This module tests the configuration constants and settings used for
land degradation analysis.
"""

import pytest

# Skip all tests in this module if numpy or te_algorithms.gdal modules are not available
np = pytest.importorskip("numpy")

try:
    from te_algorithms.gdal.land_deg.config import (
        ERROR_RECODE_INPUT_POLYS_BAND_NAME,
        FAO_WOCAT_LPD_BAND_NAME,
        JRC_LPD_BAND_NAME,
        LC_BAND_NAME,
        LC_DEG_BAND_NAME,
        LC_DEG_COMPARISON_BAND_NAME,
        LC_STATUS_BAND_NAME,
        LC_TRANS_BAND_NAME,
        MASK_VALUE,
        NODATA_VALUE,
        PERF_BAND_NAME,
        POP_AFFECTED_BAND_NAME,
        POPULATION_BAND_NAME,
        PROD_DEG_COMPARISON_BAND_NAME,
        PROD_STATUS_BAND_NAME,
        PRODUCTIVITY_CLASS_KEY,
        SDG_BAND_NAME,
        SDG_STATUS_BAND_NAME,
        SOC_BAND_NAME,
        SOC_DEG_BAND_NAME,
        SOC_STATUS_BAND_NAME,
        STATE_BAND_NAME,
        TE_LPD_BAND_NAME,
        TRAJ_BAND_NAME,
    )
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy and GDAL dependencies",
        allow_module_level=True,
    )


class TestConstants:
    """Test configuration constants."""

    def test_nodata_value_type_and_range(self):
        """Test NODATA_VALUE is correct type and in valid range."""
        assert isinstance(NODATA_VALUE, np.int16)
        assert NODATA_VALUE == -32768
        assert NODATA_VALUE == np.iinfo(np.int16).min  # Minimum value for int16

    def test_mask_value_type_and_range(self):
        """Test MASK_VALUE is correct type and in valid range."""
        assert isinstance(MASK_VALUE, np.int16)
        assert MASK_VALUE == -32767
        assert MASK_VALUE == np.iinfo(np.int16).min + 1  # One more than minimum

    def test_nodata_mask_values_distinct(self):
        """Test that NODATA and MASK values are distinct."""
        assert NODATA_VALUE != MASK_VALUE
        assert abs(NODATA_VALUE - MASK_VALUE) == 1

    def test_values_within_int16_range(self):
        """Test that all constant values are within int16 range."""
        int16_min = np.iinfo(np.int16).min
        int16_max = np.iinfo(np.int16).max

        assert int16_min <= NODATA_VALUE <= int16_max
        assert int16_min <= MASK_VALUE <= int16_max


class TestBandNames:
    """Test band name constants."""

    def test_sdg_band_names_are_strings(self):
        """Test that SDG band names are strings."""
        assert isinstance(SDG_BAND_NAME, str)
        assert isinstance(SDG_STATUS_BAND_NAME, str)
        assert len(SDG_BAND_NAME) > 0
        assert len(SDG_STATUS_BAND_NAME) > 0

    def test_productivity_band_names_are_strings(self):
        """Test that productivity band names are strings."""
        prod_bands = [
            PROD_STATUS_BAND_NAME,
            PROD_DEG_COMPARISON_BAND_NAME,
            JRC_LPD_BAND_NAME,
            FAO_WOCAT_LPD_BAND_NAME,
            TE_LPD_BAND_NAME,
            TRAJ_BAND_NAME,
            PERF_BAND_NAME,
            STATE_BAND_NAME,
        ]

        for band_name in prod_bands:
            assert isinstance(band_name, str)
            assert len(band_name) > 0

    def test_land_cover_band_names(self):
        """Test land cover band name configurations."""
        assert isinstance(LC_STATUS_BAND_NAME, str)
        assert isinstance(LC_DEG_BAND_NAME, str)
        assert isinstance(LC_DEG_COMPARISON_BAND_NAME, str)
        assert isinstance(LC_TRANS_BAND_NAME, str)

        # LC_BAND_NAME should be a list
        assert isinstance(LC_BAND_NAME, list)
        assert len(LC_BAND_NAME) > 0
        assert all(isinstance(name, str) for name in LC_BAND_NAME)

    def test_soil_carbon_band_names(self):
        """Test soil organic carbon band names."""
        assert isinstance(SOC_STATUS_BAND_NAME, str)
        assert isinstance(SOC_DEG_BAND_NAME, str)
        assert isinstance(SOC_BAND_NAME, str)
        assert len(SOC_STATUS_BAND_NAME) > 0
        assert len(SOC_DEG_BAND_NAME) > 0
        assert len(SOC_BAND_NAME) > 0

    def test_population_band_names(self):
        """Test population-related band names."""
        assert isinstance(POPULATION_BAND_NAME, str)
        assert isinstance(POP_AFFECTED_BAND_NAME, str)
        assert len(POPULATION_BAND_NAME) > 0
        assert len(POP_AFFECTED_BAND_NAME) > 0

    def test_error_recode_band_name(self):
        """Test error recode band name."""
        assert isinstance(ERROR_RECODE_INPUT_POLYS_BAND_NAME, str)
        assert len(ERROR_RECODE_INPUT_POLYS_BAND_NAME) > 0

    def test_band_names_are_unique(self):
        """Test that band names are unique where they should be."""
        individual_bands = [
            SDG_BAND_NAME,
            SDG_STATUS_BAND_NAME,
            PROD_STATUS_BAND_NAME,
            LC_STATUS_BAND_NAME,
            SOC_STATUS_BAND_NAME,
            ERROR_RECODE_INPUT_POLYS_BAND_NAME,
            TRAJ_BAND_NAME,
            PERF_BAND_NAME,
            STATE_BAND_NAME,
            LC_DEG_BAND_NAME,
            LC_TRANS_BAND_NAME,
            SOC_DEG_BAND_NAME,
            SOC_BAND_NAME,
            POPULATION_BAND_NAME,
            POP_AFFECTED_BAND_NAME,
        ]

        # Check that all individual band names are unique
        assert len(individual_bands) == len(set(individual_bands))


class TestProductivityClassKey:
    """Test the productivity classification mapping."""

    def test_productivity_class_key_structure(self):
        """Test that PRODUCTIVITY_CLASS_KEY has correct structure."""
        assert isinstance(PRODUCTIVITY_CLASS_KEY, dict)
        assert len(PRODUCTIVITY_CLASS_KEY) > 0

        # Check that all keys are strings
        assert all(isinstance(key, str) for key in PRODUCTIVITY_CLASS_KEY.keys())

        # Check that all values are integers or numpy integers
        for value in PRODUCTIVITY_CLASS_KEY.values():
            assert isinstance(value, (int, np.integer))

    def test_productivity_class_key_expected_keys(self):
        """Test that expected productivity classes are present."""
        expected_keys = {
            "Increasing",
            "Stable",
            "Stressed",
            "Moderate decline",
            "Declining",
            "No data",
        }

        actual_keys = set(PRODUCTIVITY_CLASS_KEY.keys())
        assert expected_keys.issubset(actual_keys)

    def test_productivity_class_values_range(self):
        """Test that productivity class values are in expected range."""
        values = list(PRODUCTIVITY_CLASS_KEY.values())

        # Should have values 1-5 plus NODATA_VALUE
        expected_numeric_values = {1, 2, 3, 4, 5}
        numeric_values = {v for v in values if v != NODATA_VALUE}

        assert expected_numeric_values.issubset(numeric_values)
        assert NODATA_VALUE in values

    def test_productivity_class_ordering(self):
        """Test that productivity classes have logical ordering."""
        # Higher numbers should indicate better productivity
        assert (
            PRODUCTIVITY_CLASS_KEY["Declining"]
            < PRODUCTIVITY_CLASS_KEY["Moderate decline"]
        )
        assert (
            PRODUCTIVITY_CLASS_KEY["Moderate decline"]
            < PRODUCTIVITY_CLASS_KEY["Stressed"]
        )
        assert PRODUCTIVITY_CLASS_KEY["Stressed"] < PRODUCTIVITY_CLASS_KEY["Stable"]
        assert PRODUCTIVITY_CLASS_KEY["Stable"] < PRODUCTIVITY_CLASS_KEY["Increasing"]

    def test_productivity_class_nodata_mapping(self):
        """Test that 'No data' maps to NODATA_VALUE."""
        assert PRODUCTIVITY_CLASS_KEY["No data"] == NODATA_VALUE

    def test_productivity_class_values_unique(self):
        """Test that productivity class values are unique."""
        values = list(PRODUCTIVITY_CLASS_KEY.values())
        assert len(values) == len(set(values))


class TestConfigurationIntegrity:
    """Test overall configuration integrity."""

    def test_no_conflicting_special_values(self):
        """Test that special values don't conflict with productivity classes."""
        prod_values = set(PRODUCTIVITY_CLASS_KEY.values())

        # NODATA_VALUE should be in productivity values (for "No data" class)
        assert NODATA_VALUE in prod_values

        # MASK_VALUE should not conflict with productivity values
        assert MASK_VALUE not in prod_values

    def test_band_names_contain_expected_keywords(self):
        """Test that band names contain expected keywords."""
        # SDG bands should mention SDG
        assert "SDG" in SDG_BAND_NAME
        assert "SDG" in SDG_STATUS_BAND_NAME

        # Productivity bands should mention productivity
        prod_bands = [
            PROD_STATUS_BAND_NAME,
            TRAJ_BAND_NAME,
            PERF_BAND_NAME,
            STATE_BAND_NAME,
        ]
        for band in prod_bands:
            assert "Productivity" in band or "productivity" in band

        # Land cover bands should mention land cover
        lc_bands = [LC_STATUS_BAND_NAME, LC_DEG_BAND_NAME, LC_TRANS_BAND_NAME]
        for band in lc_bands:
            assert "Land cover" in band or "land cover" in band

        # SOC bands should mention soil or carbon
        soc_bands = [SOC_STATUS_BAND_NAME, SOC_DEG_BAND_NAME, SOC_BAND_NAME]
        for band in soc_bands:
            assert (
                "soil" in band.lower() and "carbon" in band.lower()
            ) or "SOC" in band

        # Population bands should mention population
        pop_bands = [POPULATION_BAND_NAME, POP_AFFECTED_BAND_NAME]
        for band in pop_bands:
            assert "Population" in band or "population" in band

    def test_consistent_naming_patterns(self):
        """Test that similar band types follow consistent naming patterns."""
        # Status bands should contain "status"
        status_bands = [
            SDG_STATUS_BAND_NAME,
            PROD_STATUS_BAND_NAME,
            LC_STATUS_BAND_NAME,
            SOC_STATUS_BAND_NAME,
        ]
        for band in status_bands:
            assert "status" in band

        # Degradation bands should contain "degradation"
        deg_bands = [
            PROD_DEG_COMPARISON_BAND_NAME,
            LC_DEG_BAND_NAME,
            LC_DEG_COMPARISON_BAND_NAME,
            SOC_DEG_BAND_NAME,
        ]
        for band in deg_bands:
            assert "degradation" in band

    def test_configuration_completeness(self):
        """Test that configuration includes all necessary components."""
        # Test that we have configurations for all main SDG indicators
        required_components = {
            "productivity": [
                PROD_STATUS_BAND_NAME,
                TRAJ_BAND_NAME,
                PERF_BAND_NAME,
                STATE_BAND_NAME,
            ],
            "land_cover": [LC_STATUS_BAND_NAME, LC_DEG_BAND_NAME, LC_TRANS_BAND_NAME],
            "soil_carbon": [SOC_STATUS_BAND_NAME, SOC_DEG_BAND_NAME, SOC_BAND_NAME],
            "sdg": [SDG_BAND_NAME, SDG_STATUS_BAND_NAME],
            "population": [POPULATION_BAND_NAME, POP_AFFECTED_BAND_NAME],
        }

        for component, bands in required_components.items():
            assert len(bands) > 0, f"No bands defined for {component}"
            for band in bands:
                assert isinstance(band, str), (
                    f"Band name for {component} is not a string"
                )
                assert len(band) > 0, f"Empty band name for {component}"


class TestNumpyCompatibility:
    """Test numpy compatibility of configuration values."""

    def test_constants_work_with_numpy_arrays(self):
        """Test that constants work properly with numpy operations."""
        # Create arrays with the constants
        nodata_array = np.array([NODATA_VALUE, NODATA_VALUE], dtype=np.int16)
        mask_array = np.array([MASK_VALUE, MASK_VALUE], dtype=np.int16)

        # Test comparisons work
        assert np.all(nodata_array == NODATA_VALUE)
        assert np.all(mask_array == MASK_VALUE)
        assert np.all(nodata_array != MASK_VALUE)
        assert np.all(mask_array != NODATA_VALUE)

    def test_productivity_values_work_with_numpy(self):
        """Test that productivity class values work with numpy arrays."""
        # Get numeric values (excluding NODATA_VALUE which is a special case)
        numeric_values = [
            v for v in PRODUCTIVITY_CLASS_KEY.values() if v != NODATA_VALUE
        ]

        # Create array and test operations
        prod_array = np.array(numeric_values, dtype=np.int16)

        # Test that array operations work
        assert prod_array.min() >= 1
        assert prod_array.max() <= 5
        assert len(np.unique(prod_array)) == len(numeric_values)  # All values unique
