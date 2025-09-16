"""
Tests for te_algorithms.gdal.land_deg.land_deg_recode module.

This module tests the error recoding functions for land degradation analysis,
including the rasterization of error recode polygons.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Skip all tests in this module if required dependencies are not available
try:
    from te_algorithms.gdal.land_deg.land_deg_recode import rasterize_error_recode
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy, GDAL, and te_schemas dependencies",
        allow_module_level=True,
    )


class TestRasterizeErrorRecode:
    """Test the rasterize_error_recode function."""

    def create_mock_error_recode_polygons(self, features_data=None):
        """Create a mock ErrorRecodePolygons object for testing."""
        if features_data is None:
            features_data = [
                {
                    "properties": {
                        "recode_deg_to": 1,
                        "recode_stable_to": 2,
                        "recode_imp_to": 3,
                        "periods_affected": ["baseline", "reporting_1"],
                    }
                },
                {
                    "properties": {
                        "recode_deg_to": 2,
                        "recode_stable_to": 1,
                        "recode_imp_to": 1,
                        "periods_affected": ["reporting_2"],
                    }
                },
            ]

        mock_polygons = Mock()
        mock_polygons.recode_to_trans_code_dict = {
            (1, 2, 3): 100,
            (2, 1, 1): 200,
        }

        return mock_polygons, {"features": features_data}

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_basic(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test basic functionality of rasterize_error_recode."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        error_polygons, geojson_data = self.create_mock_error_recode_polygons()

        # Mock the Schema().dump() method to return our test data
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify
        mock_rasterize_class.assert_called_once()
        mock_worker.work.assert_called_once()

        # Check that Rasterize was called with correct parameters
        call_args = mock_rasterize_class.call_args
        assert call_args[0][0] == out_file  # out_file
        assert call_args[0][1] == model_file  # model_file
        assert call_args[0][3] == ["error_recode", "periods_mask"]  # properties

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_period_bitmask(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test that periods_affected are correctly converted to bitmasks."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        # Test different period combinations
        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": ["baseline"],  # Should result in bitmask 1
                }
            },
            {
                "properties": {
                    "recode_deg_to": 2,
                    "recode_stable_to": 1,
                    "recode_imp_to": 1,
                    "periods_affected": ["reporting_1"],  # Should result in bitmask 2
                }
            },
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 1,
                    "recode_imp_to": 2,
                    "periods_affected": ["reporting_2"],  # Should result in bitmask 4
                }
            },
            {
                "properties": {
                    "recode_deg_to": 3,
                    "recode_stable_to": 2,
                    "recode_imp_to": 1,
                    "periods_affected": [
                        "baseline",
                        "reporting_1",
                        "reporting_2",
                    ],  # Should result in bitmask 7 (1+2+4)
                }
            },
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )
        # Add the additional recode mappings
        error_polygons.recode_to_trans_code_dict.update(
            {
                (1, 1, 2): 300,
                (3, 2, 1): 400,
            }
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify the geojson data passed to Rasterize contains correct periods_mask
        call_args = mock_rasterize_class.call_args
        geojson_data_passed = call_args[0][2]  # The geojson dict

        # Check that periods_mask was correctly calculated
        assert (
            geojson_data_passed["features"][0]["properties"]["periods_mask"] == 1
        )  # baseline only
        assert (
            geojson_data_passed["features"][1]["properties"]["periods_mask"] == 2
        )  # reporting_1 only
        assert (
            geojson_data_passed["features"][2]["properties"]["periods_mask"] == 4
        )  # reporting_2 only
        assert (
            geojson_data_passed["features"][3]["properties"]["periods_mask"] == 7
        )  # all periods (1+2+4)

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_default_periods(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test that missing periods_affected raises ValueError."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        # Feature without periods_affected property
        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    # No periods_affected - should raise ValueError
                }
            }
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute and verify it raises ValueError
        with pytest.raises(
            ValueError, match="periods_affected is required and cannot be empty"
        ):
            rasterize_error_recode(out_file, model_file, error_polygons)

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_error_code_mapping(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test that error recode codes are correctly mapped."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": ["baseline"],
                }
            }
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify
        call_args = mock_rasterize_class.call_args
        geojson_data_passed = call_args[0][2]

        # Check that error_recode code was correctly set from the mapping
        assert geojson_data_passed["features"][0]["properties"]["error_recode"] == 100

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_empty_periods(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test behavior with empty periods_affected list."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": [],  # Empty list - should raise ValueError
                }
            }
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute and verify it raises ValueError
        with pytest.raises(
            ValueError, match="periods_affected is required and cannot be empty"
        ):
            rasterize_error_recode(out_file, model_file, error_polygons)

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_multiple_features(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test processing multiple features with different configurations."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        # Multiple features with different period combinations
        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": ["baseline", "reporting_1"],
                }
            },
            {
                "properties": {
                    "recode_deg_to": 2,
                    "recode_stable_to": 1,
                    "recode_imp_to": 1,
                    "periods_affected": ["reporting_2"],
                }
            },
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify
        call_args = mock_rasterize_class.call_args
        geojson_data_passed = call_args[0][2]

        # Check both features were processed correctly
        assert len(geojson_data_passed["features"]) == 2
        assert geojson_data_passed["features"][0]["properties"]["error_recode"] == 100
        assert (
            geojson_data_passed["features"][0]["properties"]["periods_mask"] == 3
        )  # baseline + reporting_1 (1+2)
        assert geojson_data_passed["features"][1]["properties"]["error_recode"] == 200
        assert (
            geojson_data_passed["features"][1]["properties"]["periods_mask"] == 4
        )  # reporting_2 only

    @patch("te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize")
    @patch("te_algorithms.gdal.land_deg.land_deg_recode.ErrorRecodePolygons")
    def test_rasterize_error_recode_properties_preserved(
        self, mock_error_recode_class, mock_rasterize_class
    ):
        """Test that original properties are preserved while adding new ones."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": ["baseline"],
                    "custom_property": "test_value",
                    "numeric_property": 42,
                }
            }
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_error_recode_class.Schema.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify
        call_args = mock_rasterize_class.call_args
        geojson_data_passed = call_args[0][2]

        properties = geojson_data_passed["features"][0]["properties"]

        # Check that original properties are preserved
        assert properties["custom_property"] == "test_value"
        assert properties["numeric_property"] == 42
        assert properties["recode_deg_to"] == 1
        assert properties["recode_stable_to"] == 2
        assert properties["recode_imp_to"] == 3
        assert properties["periods_affected"] == ["baseline"]

        # Check that new properties were added
        assert properties["error_recode"] == 100
        assert properties["periods_mask"] == 1
