"""
Tests for te_algorithms.gdal.land_deg.land_deg_recode module.

This module tests the error recoding functions for land degradation analysis,
including the rasterization of error recode polygons.
"""

import copy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip all tests in this module if required dependencies are not available
try:
    from te_algorithms.gdal.land_deg.land_deg_recode import rasterize_error_recode
    from te_schemas.error_recode import ErrorRecodePolygons
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy, GDAL, and te_schemas dependencies",
        allow_module_level=True,
    )


# Common patches needed for temp file / VRT operations in rasterize_error_recode
_RASTERIZE_PATCHES = {
    "rasterize": "te_algorithms.gdal.land_deg.land_deg_recode.workers.Rasterize",
    "build_vrt": "te_algorithms.gdal.land_deg.land_deg_recode.gdal.BuildVRT",
    "tempfile": "te_algorithms.gdal.land_deg.land_deg_recode.tempfile.NamedTemporaryFile",
}


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
                        "periods_affected": ["baseline", "report_1"],
                    }
                },
                {
                    "properties": {
                        "recode_deg_to": 2,
                        "recode_stable_to": 1,
                        "recode_imp_to": 1,
                        "periods_affected": ["report_2"],
                    }
                },
            ]

        mock_polygons = Mock()
        mock_polygons.recode_to_trans_code_dict = {
            (1, 2, 3): 100,
            (2, 1, 1): 200,
        }

        geojson_data = {"features": copy.deepcopy(features_data)}
        return mock_polygons, geojson_data

    @patch(_RASTERIZE_PATCHES["tempfile"])
    @patch(_RASTERIZE_PATCHES["build_vrt"])
    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_basic(
        self, mock_schema_class, mock_rasterize_class, mock_build_vrt, mock_tempfile
    ):
        """Test basic functionality of rasterize_error_recode."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker

        # Setup temp file mocks
        mock_tempfile.return_value.name = "/tmp/test_temp.tif"

        error_polygons, geojson_data = self.create_mock_error_recode_polygons()

        # Mock the Schema().dump() method to return our test data
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify: function creates two separate Rasterize workers
        assert mock_rasterize_class.call_count == 2
        assert mock_worker.work.call_count == 2

        # Check that first Rasterize was called for "error_recode"
        first_call = mock_rasterize_class.call_args_list[0]
        assert first_call[0][1] == model_file
        assert first_call[0][3] == "error_recode"

        # Check that second Rasterize was called for "periods_mask"
        second_call = mock_rasterize_class.call_args_list[1]
        assert second_call[0][1] == model_file
        assert second_call[0][3] == "periods_mask"

        # Verify BuildVRT was called to combine the two rasters
        mock_build_vrt.assert_called_once()

    @patch(_RASTERIZE_PATCHES["tempfile"])
    @patch(_RASTERIZE_PATCHES["build_vrt"])
    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_period_bitmask(
        self, mock_schema_class, mock_rasterize_class, mock_build_vrt, mock_tempfile
    ):
        """Test that periods_affected are correctly converted to bitmasks."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker
        mock_tempfile.return_value.name = "/tmp/test_temp.tif"

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
                    "periods_affected": ["report_1"],  # Should result in bitmask 2
                }
            },
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 1,
                    "recode_imp_to": 2,
                    "periods_affected": ["report_2"],  # Should result in bitmask 4
                }
            },
            {
                "properties": {
                    "recode_deg_to": 3,
                    "recode_stable_to": 2,
                    "recode_imp_to": 1,
                    "periods_affected": [
                        "baseline",
                        "report_1",
                        "report_2",
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
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify the geojson data was modified with correct periods_mask values.
        # Both Rasterize calls receive the same modified geojson dict.
        call_args = mock_rasterize_class.call_args_list[0]
        geojson_data_passed = call_args[0][2]

        # Check that periods_mask was correctly calculated
        assert (
            geojson_data_passed["features"][0]["properties"]["periods_mask"] == 1
        )  # baseline only
        assert (
            geojson_data_passed["features"][1]["properties"]["periods_mask"] == 2
        )  # report_1 only
        assert (
            geojson_data_passed["features"][2]["properties"]["periods_mask"] == 4
        )  # report_2 only
        assert (
            geojson_data_passed["features"][3]["properties"]["periods_mask"] == 7
        )  # all periods (1+2+4)

    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_default_periods(
        self, mock_schema_class, mock_rasterize_class
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
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute and verify it raises ValueError
        with pytest.raises(
            ValueError, match="periods_affected is required and cannot be empty"
        ):
            rasterize_error_recode(out_file, model_file, error_polygons)

    @patch(_RASTERIZE_PATCHES["tempfile"])
    @patch(_RASTERIZE_PATCHES["build_vrt"])
    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_error_code_mapping(
        self, mock_schema_class, mock_rasterize_class, mock_build_vrt, mock_tempfile
    ):
        """Test that error recode codes are correctly mapped."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker
        mock_tempfile.return_value.name = "/tmp/test_temp.tif"

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
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify error_recode code was set from the mapping dict
        call_args = mock_rasterize_class.call_args_list[0]
        geojson_data_passed = call_args[0][2]
        assert geojson_data_passed["features"][0]["properties"]["error_recode"] == 100

    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_empty_periods(
        self, mock_schema_class, mock_rasterize_class
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
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute and verify it raises ValueError
        with pytest.raises(
            ValueError, match="periods_affected is required and cannot be empty"
        ):
            rasterize_error_recode(out_file, model_file, error_polygons)

    @patch(_RASTERIZE_PATCHES["tempfile"])
    @patch(_RASTERIZE_PATCHES["build_vrt"])
    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_multiple_features(
        self, mock_schema_class, mock_rasterize_class, mock_build_vrt, mock_tempfile
    ):
        """Test processing multiple features with different configurations."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker
        mock_tempfile.return_value.name = "/tmp/test_temp.tif"

        # Multiple features with different period combinations
        features_data = [
            {
                "properties": {
                    "recode_deg_to": 1,
                    "recode_stable_to": 2,
                    "recode_imp_to": 3,
                    "periods_affected": ["baseline", "report_1"],
                }
            },
            {
                "properties": {
                    "recode_deg_to": 2,
                    "recode_stable_to": 1,
                    "recode_imp_to": 1,
                    "periods_affected": ["report_2"],
                }
            },
        ]

        error_polygons, geojson_data = self.create_mock_error_recode_polygons(
            features_data
        )

        # Mock the Schema().dump() method
        mock_schema = Mock()
        mock_schema.dump.return_value = geojson_data
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify data passed to Rasterize (both calls get the same dict)
        call_args = mock_rasterize_class.call_args_list[0]
        geojson_data_passed = call_args[0][2]

        # Check both features were processed correctly
        assert len(geojson_data_passed["features"]) == 2
        assert geojson_data_passed["features"][0]["properties"]["error_recode"] == 100
        assert (
            geojson_data_passed["features"][0]["properties"]["periods_mask"] == 3
        )  # baseline + report_1 (1+2)
        assert geojson_data_passed["features"][1]["properties"]["error_recode"] == 200
        assert (
            geojson_data_passed["features"][1]["properties"]["periods_mask"] == 4
        )  # report_2 only

    @patch(_RASTERIZE_PATCHES["tempfile"])
    @patch(_RASTERIZE_PATCHES["build_vrt"])
    @patch(_RASTERIZE_PATCHES["rasterize"])
    @patch.object(ErrorRecodePolygons, "Schema")
    def test_rasterize_error_recode_properties_preserved(
        self, mock_schema_class, mock_rasterize_class, mock_build_vrt, mock_tempfile
    ):
        """Test that original properties are preserved while adding new ones."""
        # Setup
        mock_worker = Mock()
        mock_rasterize_class.return_value = mock_worker
        mock_tempfile.return_value.name = "/tmp/test_temp.tif"

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
        mock_schema_class.return_value = mock_schema

        out_file = Path("test_output.tif")
        model_file = Path("test_model.tif")

        # Execute
        rasterize_error_recode(out_file, model_file, error_polygons)

        # Verify data passed to Rasterize
        call_args = mock_rasterize_class.call_args_list[0]
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
