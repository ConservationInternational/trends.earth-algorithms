"""
Tests for te_algorithms.gdal.land_deg.land_deg_stats functions.

This module tests the land degradation statistics calculation functions used for
analyzing degradation indicators across different spatial units and geometries.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

# Import te_schemas classes directly - tests will fail if not available
from te_schemas.results import JsonResults

# Skip all tests in this module if numpy or te_algorithms.gdal modules are not available
np = pytest.importorskip("numpy")

try:
    # Mock GDAL before importing the module under test
    with patch.dict(
        "sys.modules",
        {
            "osgeo": Mock(),
            "osgeo.gdal": Mock(),
            "osgeo.ogr": Mock(),
        },
    ):
        # Import the module under test
        from te_algorithms.gdal.land_deg import config, land_deg_stats
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require numpy and GDAL dependencies",
        allow_module_level=True,
    )


class TestLandDegStats(unittest.TestCase):
    """Test cases for land degradation statistics functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data arrays for different scenarios
        self.test_array_small = np.array(
            [[-1, 0, 1, -32768], [0, 1, -1, 0], [1, -1, 0, 1]], dtype=np.int16
        )

        self.test_array_large = np.random.randint(-1, 2, (50, 50), dtype=np.int16)
        # Add some NODATA values
        self.test_array_large[0:5, 0:5] = -32768

        # Cell areas for testing (in hectares)
        self.cell_areas_small = np.ones((3, 4)) * 0.25  # 0.25 hectares per cell
        self.cell_areas_large = np.ones((50, 50)) * 0.25

        # NODATA value
        self.nodata = -32768

        # Test productivity data (1-5 scale)
        self.prod_array = np.array(
            [
                [1, 2, 3, 0],  # declining, moderate decline, stressed, nodata
                [4, 5, 1, 2],  # stable, increasing, declining, moderate decline
                [3, 4, 5, 0],  # stressed, stable, increasing, nodata
            ],
            dtype=np.int16,
        )

        # Test SOC data (percentage change)
        self.soc_array = np.array(
            [
                [-50, -10, 0, 10],  # degraded, degraded, stable, improved
                [20, -32768, -5, 15],  # improved, nodata, stable, improved
                [-101, 5, -20, 0],  # nodata, stable, degraded, stable
            ],
            dtype=np.int16,
        )

    def test_get_stats_for_band_sdg_indicator(self):
        """Test _get_stats_for_band for SDG indicator data."""
        # Create masked array for SDG data (-1, 0, 1)
        masked = np.ma.MaskedArray(self.test_array_small, mask=False)

        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )

        # Check that all expected keys are present
        expected_keys = [
            "area_ha",
            "degraded_pct",
            "stable_pct",
            "improved_pct",
            "nodata_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)

        # Check total area calculation
        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats["area_ha"], total_area, places=2)

        # Check percentage calculations (should sum to 100%)
        total_pct = (
            stats["degraded_pct"]
            + stats["stable_pct"]
            + stats["improved_pct"]
            + stats["nodata_pct"]
        )
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_get_stats_for_band_productivity_5class(self):
        """Test _get_stats_for_band for 5-class productivity data."""
        masked = np.ma.MaskedArray(self.prod_array, mask=False)

        stats = land_deg_stats._get_stats_for_band(
            config.JRC_LPD_BAND_NAME, masked, self.cell_areas_small, 0
        )

        # Check that all expected keys are present
        expected_keys = [
            "area_ha",
            "degraded_pct",
            "stable_pct",
            "improved_pct",
            "nodata_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)

        # For 5-class productivity:
        # degraded = classes 1,2; stable = classes 3,4; improved = class 5
        # Count occurrences: 1 appears 3 times, 2 appears 2 times, 3 appears 2 times,
        # 4 appears 2 times, 5 appears 2 times, 0 appears 2 times

        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats["area_ha"], total_area, places=2)

    def test_get_stats_for_band_soc_degradation(self):
        """Test _get_stats_for_band for SOC degradation data."""
        masked = np.ma.MaskedArray(self.soc_array, mask=False)

        stats = land_deg_stats._get_stats_for_band(
            config.SOC_DEG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )

        # Check that all expected keys are present
        expected_keys = [
            "area_ha",
            "degraded_pct",
            "stable_pct",
            "improved_pct",
            "nodata_pct",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)

        # For SOC: degraded <= -10 AND >= -101, stable = 0, improved >= 10
        # Values: -50(deg), -10(deg), 0(stable), 10(imp), 20(imp), -32768(nodata),
        #         -5(stable), 15(imp), -101(nodata), 5(stable), -20(deg), 0(stable)

        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats["area_ha"], total_area, places=2)

    def test_get_stats_for_band_with_mask(self):
        """Test _get_stats_for_band with masked data."""
        # Create a mask that excludes some cells
        mask = np.array(
            [
                [True, False, False, True],
                [False, False, True, False],
                [False, True, False, False],
            ]
        )

        masked = np.ma.MaskedArray(self.test_array_small, mask=mask)

        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )

        # Area should only include unmasked cells
        unmasked_cells = np.sum(~mask)
        expected_area = unmasked_cells * 0.25
        self.assertAlmostEqual(stats["area_ha"], expected_area, places=2)

    def test_get_stats_for_band_edge_cases(self):
        """Test _get_stats_for_band with edge cases."""
        # Test with all NODATA
        nodata_array = np.full((3, 3), self.nodata, dtype=np.int16)
        masked = np.ma.MaskedArray(nodata_array, mask=False)
        cell_areas = np.ones((3, 3)) * 0.25

        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, cell_areas, self.nodata
        )

        self.assertAlmostEqual(stats["nodata_pct"], 100.0, places=1)
        self.assertAlmostEqual(stats["degraded_pct"], 0.0, places=1)
        self.assertAlmostEqual(stats["stable_pct"], 0.0, places=1)
        self.assertAlmostEqual(stats["improved_pct"], 0.0, places=1)

    def test_get_stats_for_band_all_degraded(self):
        """Test _get_stats_for_band with all degraded values."""
        degraded_array = np.full((3, 3), -1, dtype=np.int16)
        masked = np.ma.MaskedArray(degraded_array, mask=False)
        cell_areas = np.ones((3, 3)) * 0.25

        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, cell_areas, self.nodata
        )

        self.assertAlmostEqual(stats["degraded_pct"], 100.0, places=1)
        self.assertAlmostEqual(stats["stable_pct"], 0.0, places=1)
        self.assertAlmostEqual(stats["improved_pct"], 0.0, places=1)
        self.assertAlmostEqual(stats["nodata_pct"], 0.0, places=1)

    def test_get_stats_for_band_unknown_band(self):
        """Test _get_stats_for_band with unknown band name."""
        masked = np.ma.MaskedArray(self.test_array_small, mask=False)

        stats = land_deg_stats._get_stats_for_band(
            "Unknown Band", masked, self.cell_areas_small, self.nodata
        )

        # Unknown bands should have area calculation and all percentages should be 0.0
        self.assertIn("area_ha", stats)
        self.assertIn("degraded_pct", stats)
        self.assertIn("stable_pct", stats)
        self.assertIn("improved_pct", stats)
        self.assertEqual(stats["degraded_pct"], 0.0)
        self.assertEqual(stats["stable_pct"], 0.0)
        self.assertEqual(stats["improved_pct"], 0.0)

    def test_get_raster_bounds(self):
        """Test _get_raster_bounds function."""
        with patch.object(land_deg_stats, "ogr") as mock_ogr:
            # Mock raster dataset
            mock_rds = Mock()
            mock_rds.GetGeoTransform.return_value = (
                0,
                1,
                0,
                10,
                0,
                -1,
            )  # ul_x, x_res, _, ul_y, _, y_res
            mock_rds.RasterXSize = 10
            mock_rds.RasterYSize = 10

            # Mock geometry creation
            mock_geom = Mock()
            mock_ogr.CreateGeometryFromWkt.return_value = mock_geom

            result = land_deg_stats._get_raster_bounds(mock_rds)

            # Verify that CreateGeometryFromWkt was called with correct polygon
            mock_ogr.CreateGeometryFromWkt.assert_called_once()
            call_args = mock_ogr.CreateGeometryFromWkt.call_args[0][0]

            # Check that the WKT contains expected coordinates
            self.assertIn("POLYGON", call_args)
            self.assertIn("0", call_args)  # ul_x
            self.assertIn("10", call_args)  # ul_y and lr_x
            self.assertIn("0", call_args)  # lr_y
            self.assertEqual(result, mock_geom)

    def test_get_stats_for_geom_basic(self):
        """Test get_stats_for_geom with basic functionality."""
        with patch.object(land_deg_stats, "gdal") as mock_gdal, patch.object(
            land_deg_stats, "ogr"
        ) as mock_ogr, patch.object(land_deg_stats, "calc_cell_area") as mock_calc_area:
            # Mock GDAL objects
            mock_rds = Mock()
            mock_rds.GetGeoTransform.return_value = (0, 1, 0, 10, 0, -1)
            mock_rds.RasterXSize = 4
            mock_rds.RasterYSize = 3

            mock_rb = Mock()
            mock_rb.ReadAsArray.return_value = self.test_array_small
            mock_rb.GetNoDataValue.return_value = self.nodata
            mock_rds.GetRasterBand.return_value = mock_rb

            mock_gdal.Open.return_value = mock_rds
            mock_gdal.GA_ReadOnly = 0
            mock_gdal.GetDriverByName.return_value.Create.return_value = Mock()

            # Mock geometry
            mock_geom = Mock()
            mock_geom.GetArea.return_value = 1.0
            mock_geom.Intersection.return_value = mock_geom
            mock_geom.GetEnvelope.return_value = (
                0,
                4,
                0,
                3,
            )  # x_min, x_max, y_min, y_max
            mock_geom.ExportToWkt.return_value = "POLYGON((0 0, 4 0, 4 3, 0 3, 0 0))"

            # Mock OGR objects
            mock_ogr.CreateGeometryFromWkt.return_value = mock_geom
            mock_mem_drv = Mock()
            mock_mem_ds = Mock()
            mock_layer = Mock()
            mock_layer.GetLayerDefn.return_value = Mock()
            mock_mem_ds.CreateLayer.return_value = mock_layer
            mock_mem_drv.CreateDataSource.return_value = mock_mem_ds
            mock_ogr.GetDriverByName.return_value = mock_mem_drv
            mock_ogr.wkbPolygon = 3

            # Mock feature
            mock_feature = Mock()
            mock_ogr.Feature.return_value = mock_feature

            # Mock cell area calculation
            mock_calc_area.return_value = 10000  # 1 hectare in square meters

            # Mock rasterization
            mock_gdal.RasterizeLayer = Mock()

            result = land_deg_stats.get_stats_for_geom(
                "/fake/path.tif",
                [{"name": config.SDG_BAND_NAME, "index": 1}],
                mock_geom,
            )

            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn(config.SDG_BAND_NAME, result)
            self.assertIn("area_ha", result[config.SDG_BAND_NAME])
            self.assertIn("degraded_pct", result[config.SDG_BAND_NAME])
            self.assertIn("stable_pct", result[config.SDG_BAND_NAME])
            self.assertIn("improved_pct", result[config.SDG_BAND_NAME])

    def test_get_stats_for_geom_raster_open_failure(self):
        """Test get_stats_for_geom when raster fails to open."""
        with patch.object(land_deg_stats, "gdal") as mock_gdal:
            mock_gdal.Open.return_value = None
            mock_geom = Mock()

            with self.assertRaises(Exception) as context:
                land_deg_stats.get_stats_for_geom(
                    "/fake/path.tif",
                    [{"name": config.SDG_BAND_NAME, "index": 1}],
                    mock_geom,
                )

            self.assertIn("Failed to open raster", str(context.exception))

    def test_get_stats_for_geom_band_not_found(self):
        """Test get_stats_for_geom when band is not found."""
        with patch.object(land_deg_stats, "gdal") as mock_gdal, patch.object(
            land_deg_stats, "ogr"
        ) as mock_ogr:
            mock_rds = Mock()
            mock_rds.GetGeoTransform.return_value = (0, 1, 0, 10, 0, -1)
            mock_rds.RasterXSize = 4
            mock_rds.RasterYSize = 3
            mock_rds.GetRasterBand.return_value = None
            mock_gdal.Open.return_value = mock_rds

            # Mock geometry
            mock_geom = Mock()
            mock_geom.GetArea.return_value = 1.0
            mock_geom.Intersection.return_value = mock_geom
            mock_geom.GetEnvelope.return_value = (0, 4, 0, 3)

            # Mock OGR geometry creation
            mock_ogr.CreateGeometryFromWkt.return_value = mock_geom

            with self.assertRaises(Exception) as context:
                land_deg_stats.get_stats_for_geom(
                    "/fake/path.tif",
                    [{"name": config.SDG_BAND_NAME, "index": 999}],
                    mock_geom,
                )

            self.assertIn("Band", str(context.exception))

    def test_calc_features_stats_basic(self):
        """Test _calc_features_stats with basic functionality."""
        with patch.object(land_deg_stats, "ogr") as mock_ogr:
            # Mock GeoJSON data
            test_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"uuid": "test-uuid-1"},
                        "geometry": {"type": "Polygon", "coordinates": []},
                    }
                ],
            }

            # Mock OGR layer and feature
            mock_layer = Mock()
            mock_feature = Mock()
            mock_feature.GetField.return_value = "test-uuid-1"
            mock_geom = Mock()
            mock_feature.geometry.return_value = mock_geom

            mock_layer.__iter__ = Mock(return_value=iter([mock_feature]))
            mock_ogr.Open.return_value = [mock_layer]

            # Mock get_stats_for_geom
            with patch.object(land_deg_stats, "get_stats_for_geom") as mock_get_stats:
                mock_get_stats.return_value = {
                    config.SDG_BAND_NAME: {
                        "area_ha": 100.0,
                        "degraded_pct": 10.0,
                        "stable_pct": 80.0,
                        "improved_pct": 10.0,
                    }
                }

                result = land_deg_stats._calc_features_stats(
                    test_geojson,
                    "/fake/path.tif",
                    [{"name": config.SDG_BAND_NAME, "index": 1}],
                )

            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn(config.SDG_BAND_NAME, result)
            self.assertIn("test-uuid-1", result[config.SDG_BAND_NAME])

    def test_calculate_statistics_basic(self):
        """Test calculate_statistics with basic functionality."""
        # Mock input parameters
        test_params = {
            "error_polygons": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"uuid": "test-uuid-1"},
                        "geometry": {"type": "Polygon", "coordinates": []},
                    }
                ],
            },
            "path": "/fake/path.tif",
            "band_datas": [
                {"name": config.SDG_BAND_NAME, "index": 1},
                {"name": config.SOC_DEG_BAND_NAME, "index": 2},
            ],
        }

        # Mock _calc_features_stats
        with patch.object(land_deg_stats, "_calc_features_stats") as mock_calc_stats:
            mock_calc_stats.return_value = {
                config.SDG_BAND_NAME: {
                    "test-uuid-1": {
                        "area_ha": 100.0,
                        "degraded_pct": 10.0,
                        "stable_pct": 80.0,
                        "improved_pct": 10.0,
                    }
                },
                config.SOC_DEG_BAND_NAME: {
                    "test-uuid-1": {
                        "area_ha": 100.0,
                        "degraded_pct": 15.0,
                        "stable_pct": 75.0,
                        "improved_pct": 10.0,
                    }
                },
            }

            result = land_deg_stats.calculate_statistics(test_params)

        # Verify result structure
        self.assertIsInstance(result, JsonResults)
        self.assertEqual(result.name, "sdg-15-3-1-statistics")
        self.assertIn("stats", result.data)

        # Check that stats are reorganized by UUID
        stats_data = result.data["stats"]
        self.assertIn("test-uuid-1", stats_data)

        uuid_stats = stats_data["test-uuid-1"]
        self.assertIn(config.SDG_BAND_NAME, uuid_stats)
        self.assertIn(config.SOC_DEG_BAND_NAME, uuid_stats)

    def test_calculate_statistics_uuid_mismatch(self):
        """Test calculate_statistics with mismatched UUIDs across bands."""
        test_params = {
            "error_polygons": {},
            "path": "/fake/path.tif",
            "band_datas": [
                {"name": config.SDG_BAND_NAME, "index": 1},
                {"name": config.SOC_DEG_BAND_NAME, "index": 2},
            ],
        }

        # Mock _calc_features_stats with different UUIDs
        with patch.object(land_deg_stats, "_calc_features_stats") as mock_calc_stats:
            mock_calc_stats.return_value = {
                config.SDG_BAND_NAME: {"uuid-1": {}, "uuid-2": {}},
                config.SOC_DEG_BAND_NAME: {
                    "uuid-1": {},
                    "uuid-3": {},
                },  # Different UUID
            }

            with self.assertRaises(AssertionError):
                land_deg_stats.calculate_statistics(test_params)

    def test_calculate_statistics_single_band(self):
        """Test calculate_statistics with single band."""
        test_params = {
            "error_polygons": {},
            "path": "/fake/path.tif",
            "band_datas": [{"name": config.SDG_BAND_NAME, "index": 1}],
        }

        with patch.object(land_deg_stats, "_calc_features_stats") as mock_calc_stats:
            mock_calc_stats.return_value = {
                config.SDG_BAND_NAME: {
                    "test-uuid": {
                        "area_ha": 100.0,
                        "degraded_pct": 10.0,
                        "stable_pct": 80.0,
                        "improved_pct": 10.0,
                    }
                },
            }

            result = land_deg_stats.calculate_statistics(test_params)

            # Should not raise assertion error for single band
            self.assertIsInstance(result, JsonResults)

    def test_stats_performance_large_dataset(self):
        """Test _get_stats_for_band performance with larger dataset."""
        # Create larger test dataset
        large_array = np.random.choice(
            [-1, 0, 1, -32768], size=(100, 100), p=[0.3, 0.4, 0.2, 0.1]
        )
        large_areas = np.ones((100, 100)) * 0.01  # 0.01 hectares per cell

        masked = np.ma.MaskedArray(large_array, mask=False)

        # This should complete quickly
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, large_areas, -32768
        )

        # Basic validation
        self.assertIn("area_ha", stats)
        self.assertGreater(stats["area_ha"], 0)

        # Percentages should sum to 100%
        total_pct = (
            stats["degraded_pct"]
            + stats["stable_pct"]
            + stats["improved_pct"]
            + stats["nodata_pct"]
        )
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_band_name_constants_coverage(self):
        """Test that key band name constants are accessible."""
        # Test that band name constants are available
        band_names = [
            config.SDG_BAND_NAME,
            config.SDG_STATUS_BAND_NAME,
            config.LC_DEG_BAND_NAME,
            config.LC_DEG_COMPARISON_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            config.FAO_WOCAT_LPD_BAND_NAME,
            config.TE_LPD_BAND_NAME,
            config.PROD_DEG_COMPARISON_BAND_NAME,
            config.SOC_DEG_BAND_NAME,
        ]

        for band_name in band_names:
            self.assertIsInstance(band_name, str)
            self.assertGreater(len(band_name), 0)

    def test_all_degradation_band_types(self):
        """Test _get_stats_for_band with all supported degradation band types."""
        # Test SDG indicator bands
        sdg_bands = [
            config.SDG_BAND_NAME,
            config.SDG_STATUS_BAND_NAME,
            config.LC_DEG_BAND_NAME,
            config.LC_DEG_COMPARISON_BAND_NAME,
        ]

        for band_name in sdg_bands:
            masked = np.ma.MaskedArray(self.test_array_small, mask=False)
            stats = land_deg_stats._get_stats_for_band(
                band_name, masked, self.cell_areas_small, self.nodata
            )

            self.assertIn("degraded_pct", stats)
            self.assertIn("stable_pct", stats)
            self.assertIn("improved_pct", stats)
            self.assertIn("nodata_pct", stats)

        # Test productivity bands
        prod_bands = [
            config.JRC_LPD_BAND_NAME,
            config.FAO_WOCAT_LPD_BAND_NAME,
            config.TE_LPD_BAND_NAME,
            config.PROD_DEG_COMPARISON_BAND_NAME,
        ]

        for band_name in prod_bands:
            masked = np.ma.MaskedArray(self.prod_array, mask=False)
            stats = land_deg_stats._get_stats_for_band(
                band_name, masked, self.cell_areas_small, 0
            )

            self.assertIn("degraded_pct", stats)
            self.assertIn("stable_pct", stats)
            self.assertIn("improved_pct", stats)
            self.assertIn("nodata_pct", stats)

        # Test SOC band
        masked = np.ma.MaskedArray(self.soc_array, mask=False)
        stats = land_deg_stats._get_stats_for_band(
            config.SOC_DEG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )

        self.assertIn("degraded_pct", stats)
        self.assertIn("stable_pct", stats)
        self.assertIn("improved_pct", stats)
        self.assertIn("nodata_pct", stats)


class TestHelperFunctions(unittest.TestCase):
    """Test cases for the helper functions used in land degradation statistics."""

    def setUp(self):
        """Set up test fixtures for helper function tests."""
        # SDG-style data: -1=degraded, 0=stable, 1=improved
        self.sdg_array = np.array([[-1, 0, 1, -32768], [0, 1, -1, 0]], dtype=np.int16)

        # Status layer data: 1,2,3=degraded, 4=stable, 5,6,7=improved
        self.status_array = np.array([[1, 2, 3, 4], [5, 6, 7, -32768]], dtype=np.int16)

        # 5-class productivity data: 1,2=degraded, 3,4=stable, 5=improved, 0=nodata
        self.prod_array = np.array([[1, 2, 3, 4], [5, 0, 1, 2]], dtype=np.int16)

        # SOC data: <=-10=degraded, 0=stable, >=10=improved
        self.soc_array = np.array(
            [[-50, -10, 0, 10], [20, -32768, -5, 15]], dtype=np.int16
        )

        self.nodata = -32768

    def test_get_degraded_mask_sdg_bands(self):
        """Test _get_degraded_mask for SDG-style bands."""
        result = land_deg_stats._get_degraded_mask(config.SDG_BAND_NAME, self.sdg_array)
        expected = np.array([[True, False, False, False], [False, False, True, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_degraded_mask_status_bands(self):
        """Test _get_degraded_mask for status layer bands."""
        result = land_deg_stats._get_degraded_mask(
            config.SDG_STATUS_BAND_NAME, self.status_array
        )
        expected = np.array([[True, True, True, False], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_degraded_mask_productivity_bands(self):
        """Test _get_degraded_mask for 5-class productivity bands."""
        result = land_deg_stats._get_degraded_mask(
            config.JRC_LPD_BAND_NAME, self.prod_array
        )
        expected = np.array([[True, True, False, False], [False, False, True, True]])
        np.testing.assert_array_equal(result, expected)

    def test_get_degraded_mask_soc_band(self):
        """Test _get_degraded_mask for SOC degradation band."""
        result = land_deg_stats._get_degraded_mask(
            config.SOC_DEG_BAND_NAME, self.soc_array
        )
        expected = np.array([[True, True, False, False], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_stable_mask_sdg_bands(self):
        """Test _get_stable_mask for SDG-style bands."""
        result = land_deg_stats._get_stable_mask(config.SDG_BAND_NAME, self.sdg_array)
        expected = np.array([[False, True, False, False], [True, False, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_get_stable_mask_status_bands(self):
        """Test _get_stable_mask for status layer bands."""
        result = land_deg_stats._get_stable_mask(
            config.SDG_STATUS_BAND_NAME, self.status_array
        )
        expected = np.array([[False, False, False, True], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_stable_mask_productivity_bands(self):
        """Test _get_stable_mask for 5-class productivity bands."""
        result = land_deg_stats._get_stable_mask(
            config.JRC_LPD_BAND_NAME, self.prod_array
        )
        expected = np.array([[False, False, True, True], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_stable_mask_soc_band(self):
        """Test _get_stable_mask for SOC degradation band."""
        result = land_deg_stats._get_stable_mask(
            config.SOC_DEG_BAND_NAME, self.soc_array
        )
        expected = np.array([[False, False, True, False], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_improved_mask_sdg_bands(self):
        """Test _get_improved_mask for SDG-style bands."""
        result = land_deg_stats._get_improved_mask(config.SDG_BAND_NAME, self.sdg_array)
        expected = np.array([[False, False, True, False], [False, True, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_improved_mask_status_bands(self):
        """Test _get_improved_mask for status layer bands."""
        result = land_deg_stats._get_improved_mask(
            config.SDG_STATUS_BAND_NAME, self.status_array
        )
        expected = np.array([[False, False, False, False], [True, True, True, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_improved_mask_productivity_bands(self):
        """Test _get_improved_mask for 5-class productivity bands."""
        result = land_deg_stats._get_improved_mask(
            config.JRC_LPD_BAND_NAME, self.prod_array
        )
        expected = np.array([[False, False, False, False], [True, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_improved_mask_soc_band(self):
        """Test _get_improved_mask for SOC degradation band."""
        result = land_deg_stats._get_improved_mask(
            config.SOC_DEG_BAND_NAME, self.soc_array
        )
        expected = np.array([[False, False, False, True], [True, False, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_get_nodata_mask_regular_bands(self):
        """Test _get_nodata_mask for regular bands (only nodata value)."""
        result = land_deg_stats._get_nodata_mask(
            config.SDG_BAND_NAME, self.sdg_array, self.nodata
        )
        expected = np.array([[False, False, False, True], [False, False, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_get_nodata_mask_productivity_bands(self):
        """Test _get_nodata_mask for productivity bands (nodata and 0)."""
        result = land_deg_stats._get_nodata_mask(
            config.JRC_LPD_BAND_NAME, self.prod_array, self.nodata
        )
        expected = np.array([[False, False, False, False], [False, True, False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_recode_to_common_classes_sdg_band(self):
        """Test _recode_to_common_classes for SDG bands."""
        result = land_deg_stats._recode_to_common_classes(
            config.SDG_BAND_NAME, self.sdg_array, self.nodata
        )
        expected = np.array([[-1, 0, 1, -32768], [0, 1, -1, 0]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_recode_to_common_classes_status_band(self):
        """Test _recode_to_common_classes for status bands."""
        result = land_deg_stats._recode_to_common_classes(
            config.SDG_STATUS_BAND_NAME, self.status_array, self.nodata
        )
        expected = np.array([[-1, -1, -1, 0], [1, 1, 1, -32768]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_recode_to_common_classes_productivity_band(self):
        """Test _recode_to_common_classes for productivity bands."""
        result = land_deg_stats._recode_to_common_classes(
            config.JRC_LPD_BAND_NAME, self.prod_array, self.nodata
        )
        expected = np.array([[-1, -1, 0, 0], [1, -32768, -1, -1]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_recode_to_common_classes_soc_band(self):
        """Test _recode_to_common_classes for SOC bands."""
        result = land_deg_stats._recode_to_common_classes(
            config.SOC_DEG_BAND_NAME, self.soc_array, self.nodata
        )
        expected = np.array([[-1, -1, 0, 1], [1, -32768, -32768, 1]], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_recode_preserves_nodata(self):
        """Test that _recode_to_common_classes preserves nodata values."""
        # Create array with nodata in various positions
        test_array = np.array(
            [[1, self.nodata, 0], [self.nodata, 2, 3]], dtype=np.int16
        )
        result = land_deg_stats._recode_to_common_classes(
            config.JRC_LPD_BAND_NAME, test_array, self.nodata
        )

        # Check that nodata values are preserved
        self.assertEqual(result[0, 1], self.nodata)
        self.assertEqual(result[1, 0], self.nodata)

    def test_helper_functions_unknown_band(self):
        """Test helper functions with unknown band names return appropriate defaults."""
        unknown_band = "Unknown Band"
        test_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)

        # Should return all False for unknown bands
        degraded_mask = land_deg_stats._get_degraded_mask(unknown_band, test_array)
        stable_mask = land_deg_stats._get_stable_mask(unknown_band, test_array)
        improved_mask = land_deg_stats._get_improved_mask(unknown_band, test_array)

        expected_false = np.zeros_like(test_array, dtype=bool)
        np.testing.assert_array_equal(degraded_mask, expected_false)
        np.testing.assert_array_equal(stable_mask, expected_false)
        np.testing.assert_array_equal(improved_mask, expected_false)


class TestCrosstabFunction(unittest.TestCase):
    """Test cases for the crosstab statistics function."""

    def setUp(self):
        """Set up test fixtures for crosstab function tests."""
        # Create simple test data
        self.band_1 = np.array([[-1, 0, 1], [0, 1, -1]], dtype=np.int16)  # SDG-style
        self.band_2 = np.array(
            [[1, 2, 5], [3, 4, 1]], dtype=np.int16
        )  # Productivity-style
        self.cell_areas = np.ones((2, 3)) * 0.25  # 0.25 hectares per cell
        self.nodata = -32768

    def test_get_stats_crosstab_basic(self):
        """Test basic crosstab functionality."""
        result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            self.band_1,
            self.band_2,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Check structure
        self.assertIn("total_area_ha", result)
        self.assertIn("crosstab", result)
        self.assertIn("marginals_1", result)
        self.assertIn("marginals_2", result)
        self.assertIn("band_1_name", result)
        self.assertIn("band_2_name", result)

        # Check total area
        expected_total = 2 * 3 * 0.25  # 1.5 hectares
        self.assertAlmostEqual(result["total_area_ha"], expected_total, places=2)

        # Check crosstab structure
        for class_name in ["degraded", "stable", "improved"]:
            self.assertIn(class_name, result["crosstab"])
            for class_name_2 in ["degraded", "stable", "improved"]:
                self.assertIn(class_name_2, result["crosstab"][class_name])
                self.assertIn("area_ha", result["crosstab"][class_name][class_name_2])
                self.assertIn("area_pct", result["crosstab"][class_name][class_name_2])

    def test_get_stats_crosstab_with_nodata(self):
        """Test crosstab with nodata values."""
        # Add nodata to test arrays
        band_1_with_nodata = self.band_1.copy()
        band_2_with_nodata = self.band_2.copy()
        band_1_with_nodata[0, 0] = self.nodata
        band_2_with_nodata[1, 2] = 0  # 0 is nodata for productivity bands

        result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            band_1_with_nodata,
            band_2_with_nodata,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Total area should exclude nodata cells
        expected_total = 4 * 0.25  # 4 valid cells * 0.25 hectares = 1.0 hectare
        self.assertAlmostEqual(result["total_area_ha"], expected_total, places=2)

    def test_get_stats_crosstab_all_nodata(self):
        """Test crosstab when all data is nodata."""
        nodata_array = np.full((2, 3), self.nodata, dtype=np.int16)

        result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            nodata_array,
            nodata_array,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Should return zero area and empty crosstab
        self.assertEqual(result["total_area_ha"], 0.0)
        self.assertEqual(result["crosstab"], {})

    def test_crosstab_marginals_sum_correctly(self):
        """Test that marginal totals sum to 100%."""
        result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            self.band_1,
            self.band_2,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Check marginals for band 1
        total_pct_1 = sum(
            result["marginals_1"][class_name]["area_pct"]
            for class_name in ["degraded", "stable", "improved"]
        )
        self.assertAlmostEqual(total_pct_1, 100.0, places=1)

        # Check marginals for band 2
        total_pct_2 = sum(
            result["marginals_2"][class_name]["area_pct"]
            for class_name in ["degraded", "stable", "improved"]
        )
        self.assertAlmostEqual(total_pct_2, 100.0, places=1)

        # Check that crosstab percentages sum to 100%
        total_crosstab_pct = sum(
            result["crosstab"][class_1][class_2]["area_pct"]
            for class_1 in ["degraded", "stable", "improved"]
            for class_2 in ["degraded", "stable", "improved"]
        )
        self.assertAlmostEqual(total_crosstab_pct, 100.0, places=1)


class TestConsistencyBetweenFunctions(unittest.TestCase):
    """Test cases to verify consistency between different statistics functions."""

    def setUp(self):
        """Set up test fixtures for consistency tests."""
        # Create test data with known distribution
        self.band_1 = np.array(
            [[-1, 0, 1, -32768], [0, 1, -1, 0]], dtype=np.int16
        )  # SDG-style
        self.band_2 = np.array(
            [[1, 2, 5, -32768], [3, 4, 1, 2]], dtype=np.int16
        )  # Productivity-style
        self.cell_areas = np.ones((2, 4)) * 0.25  # 0.25 hectares per cell
        self.nodata = -32768

    def test_crosstab_marginals_match_individual_stats(self):
        """Test that crosstab marginal totals match individual band statistics."""
        # Get crosstab stats
        crosstab_result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            self.band_1,
            self.band_2,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Calculate expected percentages based on valid overlap area
        valid_mask_1 = np.logical_and(
            self.band_1 != self.nodata, self.band_2 != self.nodata
        )
        valid_mask_2 = np.logical_and(
            self.band_2 != self.nodata, self.band_1 != self.nodata
        )

        # For band 1 marginals vs individual stats
        degraded_area_1 = np.sum(
            np.logical_and(self.band_1 == -1, valid_mask_1) * self.cell_areas
        )
        stable_area_1 = np.sum(
            np.logical_and(self.band_1 == 0, valid_mask_1) * self.cell_areas
        )
        improved_area_1 = np.sum(
            np.logical_and(self.band_1 == 1, valid_mask_1) * self.cell_areas
        )

        # Check that marginals_1 percentages are correct
        total_valid_area = crosstab_result["total_area_ha"]
        expected_degraded_pct_1 = (
            (degraded_area_1 / total_valid_area * 100) if total_valid_area > 0 else 0
        )
        expected_stable_pct_1 = (
            (stable_area_1 / total_valid_area * 100) if total_valid_area > 0 else 0
        )
        expected_improved_pct_1 = (
            (improved_area_1 / total_valid_area * 100) if total_valid_area > 0 else 0
        )

        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["degraded"]["area_pct"],
            expected_degraded_pct_1,
            places=2,
            msg="Band 1 degraded marginal percentage should match expected value",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["stable"]["area_pct"],
            expected_stable_pct_1,
            places=2,
            msg="Band 1 stable marginal percentage should match expected value",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["improved"]["area_pct"],
            expected_improved_pct_1,
            places=2,
            msg="Band 1 improved marginal percentage should match expected value",
        )

        # For band 2 marginals - check using recode function for consistency
        recoded_2 = land_deg_stats._recode_to_common_classes(
            config.JRC_LPD_BAND_NAME, self.band_2, self.nodata
        )

        degraded_area_2 = np.sum(
            np.logical_and(recoded_2 == -1, valid_mask_2) * self.cell_areas
        )
        stable_area_2 = np.sum(
            np.logical_and(recoded_2 == 0, valid_mask_2) * self.cell_areas
        )
        improved_area_2 = np.sum(
            np.logical_and(recoded_2 == 1, valid_mask_2) * self.cell_areas
        )

        expected_degraded_pct_2 = (
            (degraded_area_2 / total_valid_area * 100) if total_valid_area > 0 else 0
        )
        expected_stable_pct_2 = (
            (stable_area_2 / total_valid_area * 100) if total_valid_area > 0 else 0
        )
        expected_improved_pct_2 = (
            (improved_area_2 / total_valid_area * 100) if total_valid_area > 0 else 0
        )

        self.assertAlmostEqual(
            crosstab_result["marginals_2"]["degraded"]["area_pct"],
            expected_degraded_pct_2,
            places=2,
            msg="Band 2 degraded marginal percentage should match expected value",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_2"]["stable"]["area_pct"],
            expected_stable_pct_2,
            places=2,
            msg="Band 2 stable marginal percentage should match expected value",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_2"]["improved"]["area_pct"],
            expected_improved_pct_2,
            places=2,
            msg="Band 2 improved marginal percentage should match expected value",
        )

    def test_individual_stats_vs_crosstab_with_masked_arrays(self):
        """Test direct comparison between individual stats and crosstab marginals using masked arrays."""
        # Create test data without nodata to ensure clean comparison
        band_simple = np.array([[-1, 0, 1], [1, 0, -1]], dtype=np.int16)
        cell_areas_simple = np.ones((2, 3)) * 0.25

        # Create masked arrays (no masking applied)
        masked_array = np.ma.MaskedArray(band_simple, mask=False)

        # Get individual stats
        individual_stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked_array, cell_areas_simple, self.nodata
        )

        # Get crosstab stats using the same band against itself
        crosstab_result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.SDG_BAND_NAME,
            band_simple,
            band_simple,
            cell_areas_simple,
            self.nodata,
            self.nodata,
        )

        # The marginals should match the individual stats (both bands are identical)
        # Since there's no nodata and no masking, areas should be identical

        # Check area calculations are consistent
        self.assertAlmostEqual(
            individual_stats["area_ha"],
            crosstab_result["total_area_ha"],
            places=2,
            msg="Total areas should match when no masking is applied",
        )

        # Check that marginal percentages match individual stats percentages
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["degraded"]["area_pct"],
            individual_stats["degraded_pct"],
            places=2,
            msg="Degraded percentages should match between individual stats and crosstab marginals",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["stable"]["area_pct"],
            individual_stats["stable_pct"],
            places=2,
            msg="Stable percentages should match between individual stats and crosstab marginals",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["improved"]["area_pct"],
            individual_stats["improved_pct"],
            places=2,
            msg="Improved percentages should match between individual stats and crosstab marginals",
        )

        # Since it's the same band against itself, marginals_2 should equal marginals_1
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["degraded"]["area_pct"],
            crosstab_result["marginals_2"]["degraded"]["area_pct"],
            places=2,
            msg="Marginals for same band should be identical",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["stable"]["area_pct"],
            crosstab_result["marginals_2"]["stable"]["area_pct"],
            places=2,
            msg="Marginals for same band should be identical",
        )
        self.assertAlmostEqual(
            crosstab_result["marginals_1"]["improved"]["area_pct"],
            crosstab_result["marginals_2"]["improved"]["area_pct"],
            places=2,
            msg="Marginals for same band should be identical",
        )

    def test_crosstab_marginals_sum_to_100_percent(self):
        """Test that each band's marginal totals sum to 100%."""
        crosstab_result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            self.band_1,
            self.band_2,
            self.cell_areas,
            self.nodata,
            self.nodata,
        )

        # Check marginals_1 sum to 100%
        total_pct_1 = (
            crosstab_result["marginals_1"]["degraded"]["area_pct"]
            + crosstab_result["marginals_1"]["stable"]["area_pct"]
            + crosstab_result["marginals_1"]["improved"]["area_pct"]
        )
        self.assertAlmostEqual(
            total_pct_1,
            100.0,
            places=1,
            msg="Band 1 marginal percentages should sum to 100%",
        )

        # Check marginals_2 sum to 100%
        total_pct_2 = (
            crosstab_result["marginals_2"]["degraded"]["area_pct"]
            + crosstab_result["marginals_2"]["stable"]["area_pct"]
            + crosstab_result["marginals_2"]["improved"]["area_pct"]
        )
        self.assertAlmostEqual(
            total_pct_2,
            100.0,
            places=1,
            msg="Band 2 marginal percentages should sum to 100%",
        )

    def test_crosstab_consistency_with_same_band(self):
        """Test crosstab with the same band against itself - should have no off-diagonal values."""
        # Create a simple band with clear classes
        same_band = np.array([[-1, 0, 1], [1, 0, -1]], dtype=np.int16)
        cell_areas_small = np.ones((2, 3)) * 0.25

        crosstab_result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.SDG_BAND_NAME,  # Same band
            same_band,
            same_band,
            cell_areas_small,
            self.nodata,
            self.nodata,
        )

        # All off-diagonal values should be 0
        self.assertEqual(
            crosstab_result["crosstab"]["degraded"]["stable"]["area_pct"], 0.0
        )
        self.assertEqual(
            crosstab_result["crosstab"]["degraded"]["improved"]["area_pct"], 0.0
        )
        self.assertEqual(
            crosstab_result["crosstab"]["stable"]["degraded"]["area_pct"], 0.0
        )
        self.assertEqual(
            crosstab_result["crosstab"]["stable"]["improved"]["area_pct"], 0.0
        )
        self.assertEqual(
            crosstab_result["crosstab"]["improved"]["degraded"]["area_pct"], 0.0
        )
        self.assertEqual(
            crosstab_result["crosstab"]["improved"]["stable"]["area_pct"], 0.0
        )

        # Diagonal values should match marginal totals
        self.assertAlmostEqual(
            crosstab_result["crosstab"]["degraded"]["degraded"]["area_pct"],
            crosstab_result["marginals_1"]["degraded"]["area_pct"],
            places=2,
        )
        self.assertAlmostEqual(
            crosstab_result["crosstab"]["stable"]["stable"]["area_pct"],
            crosstab_result["marginals_1"]["stable"]["area_pct"],
            places=2,
        )
        self.assertAlmostEqual(
            crosstab_result["crosstab"]["improved"]["improved"]["area_pct"],
            crosstab_result["marginals_1"]["improved"]["area_pct"],
            places=2,
        )

    def test_crosstab_with_different_nodata_values(self):
        """Test crosstab with different nodata values for each band."""
        # Create bands with different nodata values at different positions
        # SDG band: -1=degraded, 0=stable, 1=improved
        # JRC_LPD band: 1,2=degraded, 3,4=stable, 5=improved, 0=nodata
        band_1 = np.array(
            [[-1, 0, 1], [1, 0, -32768]], dtype=np.int16
        )  # nodata=-32768 at [1,2]
        band_2 = np.array(
            [[1, 3, -9999], [2, 4, 5]], dtype=np.int16
        )  # nodata=-9999 at [0,2]
        cell_areas_small = np.ones((2, 3)) * 0.25

        crosstab_result = land_deg_stats._get_stats_crosstab(
            config.SDG_BAND_NAME,
            config.JRC_LPD_BAND_NAME,
            band_1,
            band_2,
            cell_areas_small,
            -32768,  # band_1_nodata
            -9999,  # band_2_nodata
        )

        # Should exclude cells where either band has nodata
        # band_1 has nodata at [1,2], band_2 has nodata at [0,2]
        # So we exclude 2 cells, leaving 4 valid cells
        expected_total = 4 * 0.25  # 1.0 hectare
        self.assertAlmostEqual(
            crosstab_result["total_area_ha"], expected_total, places=2
        )


if __name__ == "__main__":
    unittest.main()
