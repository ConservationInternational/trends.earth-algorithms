"""
Tests for te_algorithms.gdal.land_deg.land_deg_stats functions.

This module tests the land degradation statistics calculation functions used for
analyzing degradation indicators across different spatial units and geometries.
"""

import unittest
from unittest.mock import Mock, patch
import pytest

# Skip all tests in this module if numpy or te_algorithms.gdal modules are not available
np = pytest.importorskip("numpy")

# Import te_schemas classes directly (no mocking)
try:
    from te_schemas.results import JsonResults
    TE_SCHEMAS_AVAILABLE = True
except ImportError:
    # Fallback to mock objects if te_schemas not available
    from unittest.mock import Mock
    TE_SCHEMAS_AVAILABLE = False

try:
    # Mock GDAL and te_schemas before importing the module under test
    with patch.dict('sys.modules', {
        'osgeo': Mock(),
        'osgeo.gdal': Mock(),
        'osgeo.ogr': Mock(),
        'te_schemas': Mock(),
        'te_schemas.jobs': Mock(),
        'te_schemas.results': Mock()
    }):
        # Import the module under test
        from te_algorithms.gdal.land_deg import land_deg_stats
        from te_algorithms.gdal.land_deg import config
except ImportError:
    pytest.skip("te_algorithms.gdal modules require numpy and GDAL dependencies", allow_module_level=True)


class TestLandDegStats(unittest.TestCase):
    """Test cases for land degradation statistics functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data arrays for different scenarios
        self.test_array_small = np.array([
            [-1, 0, 1, -32768],
            [0, 1, -1, 0],
            [1, -1, 0, 1]
        ], dtype=np.int16)
        
        self.test_array_large = np.random.randint(-1, 2, (50, 50), dtype=np.int16)
        # Add some NODATA values
        self.test_array_large[0:5, 0:5] = -32768
        
        # Cell areas for testing (in hectares)
        self.cell_areas_small = np.ones((3, 4)) * 0.25  # 0.25 hectares per cell
        self.cell_areas_large = np.ones((50, 50)) * 0.25
        
        # NODATA value
        self.nodata = -32768
        
        # Test productivity data (1-5 scale)
        self.prod_array = np.array([
            [1, 2, 3, 0],   # declining, moderate decline, stressed, nodata
            [4, 5, 1, 2],   # stable, increasing, declining, moderate decline
            [3, 4, 5, 0]    # stressed, stable, increasing, nodata
        ], dtype=np.int16)
        
        # Test SOC data (percentage change)
        self.soc_array = np.array([
            [-50, -10, 0, 10],    # degraded, degraded, stable, improved
            [20, -32768, -5, 15], # improved, nodata, stable, improved
            [-101, 5, -20, 0]     # nodata, stable, degraded, stable
        ], dtype=np.int16)

    def test_get_stats_for_band_sdg_indicator(self):
        """Test _get_stats_for_band for SDG indicator data."""
        # Create masked array for SDG data (-1, 0, 1)
        masked = np.ma.MaskedArray(self.test_array_small, mask=False)
        
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )
        
        # Check that all expected keys are present
        expected_keys = ['area_ha', 'degraded_pct', 'stable_pct', 'improved_pct', 'nodata_pct']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)
        
        # Check total area calculation
        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats['area_ha'], total_area, places=2)
        
        # Check percentage calculations (should sum to 100%)
        total_pct = (stats['degraded_pct'] + stats['stable_pct'] + 
                    stats['improved_pct'] + stats['nodata_pct'])
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_get_stats_for_band_productivity_5class(self):
        """Test _get_stats_for_band for 5-class productivity data."""
        masked = np.ma.MaskedArray(self.prod_array, mask=False)
        
        stats = land_deg_stats._get_stats_for_band(
            config.JRC_LPD_BAND_NAME, masked, self.cell_areas_small, 0
        )
        
        # Check that all expected keys are present
        expected_keys = ['area_ha', 'degraded_pct', 'stable_pct', 'improved_pct', 'nodata_pct']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)
        
        # For 5-class productivity:
        # degraded = classes 1,2; stable = classes 3,4; improved = class 5
        # Count occurrences: 1 appears 3 times, 2 appears 2 times, 3 appears 2 times, 
        # 4 appears 2 times, 5 appears 2 times, 0 appears 2 times
        
        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats['area_ha'], total_area, places=2)

    def test_get_stats_for_band_soc_degradation(self):
        """Test _get_stats_for_band for SOC degradation data."""
        masked = np.ma.MaskedArray(self.soc_array, mask=False)
        
        stats = land_deg_stats._get_stats_for_band(
            config.SOC_DEG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )
        
        # Check that all expected keys are present
        expected_keys = ['area_ha', 'degraded_pct', 'stable_pct', 'improved_pct', 'nodata_pct']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)
        
        # For SOC: degraded <= -10 AND >= -101, stable = 0, improved >= 10
        # Values: -50(deg), -10(deg), 0(stable), 10(imp), 20(imp), -32768(nodata), 
        #         -5(stable), 15(imp), -101(nodata), 5(stable), -20(deg), 0(stable)
        
        total_area = 3 * 4 * 0.25  # 3 hectares
        self.assertAlmostEqual(stats['area_ha'], total_area, places=2)

    def test_get_stats_for_band_with_mask(self):
        """Test _get_stats_for_band with masked data."""
        # Create a mask that excludes some cells
        mask = np.array([
            [True, False, False, True],
            [False, False, True, False],
            [False, True, False, False]
        ])
        
        masked = np.ma.MaskedArray(self.test_array_small, mask=mask)
        
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )
        
        # Area should only include unmasked cells
        unmasked_cells = np.sum(~mask)
        expected_area = unmasked_cells * 0.25
        self.assertAlmostEqual(stats['area_ha'], expected_area, places=2)

    def test_get_stats_for_band_edge_cases(self):
        """Test _get_stats_for_band with edge cases."""
        # Test with all NODATA
        nodata_array = np.full((3, 3), self.nodata, dtype=np.int16)
        masked = np.ma.MaskedArray(nodata_array, mask=False)
        cell_areas = np.ones((3, 3)) * 0.25
        
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, cell_areas, self.nodata
        )
        
        self.assertAlmostEqual(stats['nodata_pct'], 100.0, places=1)
        self.assertAlmostEqual(stats['degraded_pct'], 0.0, places=1)
        self.assertAlmostEqual(stats['stable_pct'], 0.0, places=1)
        self.assertAlmostEqual(stats['improved_pct'], 0.0, places=1)

    def test_get_stats_for_band_all_degraded(self):
        """Test _get_stats_for_band with all degraded values."""
        degraded_array = np.full((3, 3), -1, dtype=np.int16)
        masked = np.ma.MaskedArray(degraded_array, mask=False)
        cell_areas = np.ones((3, 3)) * 0.25
        
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, cell_areas, self.nodata
        )
        
        self.assertAlmostEqual(stats['degraded_pct'], 100.0, places=1)
        self.assertAlmostEqual(stats['stable_pct'], 0.0, places=1)
        self.assertAlmostEqual(stats['improved_pct'], 0.0, places=1)
        self.assertAlmostEqual(stats['nodata_pct'], 0.0, places=1)

    def test_get_stats_for_band_unknown_band(self):
        """Test _get_stats_for_band with unknown band name."""
        masked = np.ma.MaskedArray(self.test_array_small, mask=False)
        
        stats = land_deg_stats._get_stats_for_band(
            "Unknown Band", masked, self.cell_areas_small, self.nodata
        )
        
        # Should only have area calculation, no percentage breakdowns
        self.assertIn('area_ha', stats)
        self.assertNotIn('degraded_pct', stats)
        self.assertNotIn('stable_pct', stats)
        self.assertNotIn('improved_pct', stats)

    def test_get_raster_bounds(self):
        """Test _get_raster_bounds function."""
        with patch.object(land_deg_stats, 'ogr') as mock_ogr:
            # Mock raster dataset
            mock_rds = Mock()
            mock_rds.GetGeoTransform.return_value = (0, 1, 0, 10, 0, -1)  # ul_x, x_res, _, ul_y, _, y_res
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
        with patch.object(land_deg_stats, 'gdal') as mock_gdal, \
             patch.object(land_deg_stats, 'ogr') as mock_ogr, \
             patch.object(land_deg_stats, 'calc_cell_area') as mock_calc_area:
            
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
            mock_geom.GetEnvelope.return_value = (0, 4, 0, 3)  # x_min, x_max, y_min, y_max
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
                "/fake/path.tif", config.SDG_BAND_NAME, 1, mock_geom
            )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('area_ha', result)
            self.assertIn('degraded_pct', result)
            self.assertIn('stable_pct', result)
            self.assertIn('improved_pct', result)

    def test_get_stats_for_geom_raster_open_failure(self):
        """Test get_stats_for_geom when raster fails to open."""
        with patch.object(land_deg_stats, 'gdal') as mock_gdal:
            mock_gdal.Open.return_value = None
            mock_geom = Mock()
            
            with self.assertRaises(Exception) as context:
                land_deg_stats.get_stats_for_geom(
                    "/fake/path.tif", config.SDG_BAND_NAME, 1, mock_geom
                )
            
            self.assertIn("Failed to open raster", str(context.exception))

    def test_get_stats_for_geom_band_not_found(self):
        """Test get_stats_for_geom when band is not found."""
        with patch.object(land_deg_stats, 'gdal') as mock_gdal:
            mock_rds = Mock()
            mock_rds.GetRasterBand.return_value = None
            mock_gdal.Open.return_value = mock_rds
            mock_geom = Mock()
            
            with self.assertRaises(Exception) as context:
                land_deg_stats.get_stats_for_geom(
                    "/fake/path.tif", config.SDG_BAND_NAME, 999, mock_geom
                )
            
            self.assertIn("Band", str(context.exception))

    def test_calc_features_stats_basic(self):
        """Test _calc_features_stats with basic functionality."""
        with patch.object(land_deg_stats, 'ogr') as mock_ogr:
            # Mock GeoJSON data
            test_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"uuid": "test-uuid-1"},
                        "geometry": {"type": "Polygon", "coordinates": []}
                    }
                ]
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
            with patch.object(land_deg_stats, 'get_stats_for_geom') as mock_get_stats:
                mock_get_stats.return_value = {
                    'area_ha': 100.0,
                    'degraded_pct': 10.0,
                    'stable_pct': 80.0,
                    'improved_pct': 10.0
                }
                
                result = land_deg_stats._calc_features_stats(
                    test_geojson, "/fake/path.tif", config.SDG_BAND_NAME, 1
                )
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertEqual(result['band_name'], config.SDG_BAND_NAME)
            self.assertIn('stats', result)
            self.assertIn('test-uuid-1', result['stats'])

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
                        "geometry": {"type": "Polygon", "coordinates": []}
                    }
                ]
            },
            "path": "/fake/path.tif",
            "band_datas": [
                {"name": config.SDG_BAND_NAME, "index": 1},
                {"name": config.SOC_DEG_BAND_NAME, "index": 2}
            ]
        }
        
        # Mock _calc_features_stats
        with patch.object(land_deg_stats, '_calc_features_stats') as mock_calc_stats:
            mock_calc_stats.side_effect = [
                {
                    "band_name": config.SDG_BAND_NAME,
                    "stats": {
                        "test-uuid-1": {
                            'area_ha': 100.0,
                            'degraded_pct': 10.0,
                            'stable_pct': 80.0,
                            'improved_pct': 10.0
                        }
                    }
                },
                {
                    "band_name": config.SOC_DEG_BAND_NAME,
                    "stats": {
                        "test-uuid-1": {
                            'area_ha': 100.0,
                            'degraded_pct': 15.0,
                            'stable_pct': 75.0,
                            'improved_pct': 10.0
                        }
                    }
                }
            ]
            
            result = land_deg_stats.calculate_statistics(test_params)
        
        # Verify result structure
        if TE_SCHEMAS_AVAILABLE:
            self.assertIsInstance(result, JsonResults)
            self.assertEqual(result.name, "sdg-15-3-1-statistics")
            self.assertIn("stats", result.data)
            
            # Check that stats are reorganized by UUID
            stats_data = result.data["stats"]
            self.assertIn("test-uuid-1", stats_data)
            
            uuid_stats = stats_data["test-uuid-1"]
            self.assertIn(config.SDG_BAND_NAME, uuid_stats)
            self.assertIn(config.SOC_DEG_BAND_NAME, uuid_stats)
        else:
            # When te_schemas not available, function should still work
            self.assertIsNotNone(result)

    def test_calculate_statistics_uuid_mismatch(self):
        """Test calculate_statistics with mismatched UUIDs across bands."""
        test_params = {
            "error_polygons": {},
            "path": "/fake/path.tif",
            "band_datas": [
                {"name": config.SDG_BAND_NAME, "index": 1},
                {"name": config.SOC_DEG_BAND_NAME, "index": 2}
            ]
        }
        
        # Mock _calc_features_stats with different UUIDs
        with patch.object(land_deg_stats, '_calc_features_stats') as mock_calc_stats:
            mock_calc_stats.side_effect = [
                {
                    "band_name": config.SDG_BAND_NAME,
                    "stats": {"uuid-1": {}, "uuid-2": {}}
                },
                {
                    "band_name": config.SOC_DEG_BAND_NAME,
                    "stats": {"uuid-1": {}, "uuid-3": {}}  # Different UUID
                }
            ]
            
            with self.assertRaises(AssertionError):
                land_deg_stats.calculate_statistics(test_params)

    def test_calculate_statistics_single_band(self):
        """Test calculate_statistics with single band."""
        test_params = {
            "error_polygons": {},
            "path": "/fake/path.tif",
            "band_datas": [
                {"name": config.SDG_BAND_NAME, "index": 1}
            ]
        }
        
        with patch.object(land_deg_stats, '_calc_features_stats') as mock_calc_stats:
            mock_calc_stats.return_value = {
                "band_name": config.SDG_BAND_NAME,
                "stats": {
                    "test-uuid": {
                        'area_ha': 100.0,
                        'degraded_pct': 10.0,
                        'stable_pct': 80.0,
                        'improved_pct': 10.0
                    }
                }
            }
            
            result = land_deg_stats.calculate_statistics(test_params)
            
            # Should not raise assertion error for single band
            if TE_SCHEMAS_AVAILABLE:
                self.assertIsInstance(result, JsonResults)
            else:
                self.assertIsNotNone(result)

    def test_stats_performance_large_dataset(self):
        """Test _get_stats_for_band performance with larger dataset."""
        # Create larger test dataset
        large_array = np.random.choice([-1, 0, 1, -32768], size=(100, 100), p=[0.3, 0.4, 0.2, 0.1])
        large_areas = np.ones((100, 100)) * 0.01  # 0.01 hectares per cell
        
        masked = np.ma.MaskedArray(large_array, mask=False)
        
        # This should complete quickly
        stats = land_deg_stats._get_stats_for_band(
            config.SDG_BAND_NAME, masked, large_areas, -32768
        )
        
        # Basic validation
        self.assertIn('area_ha', stats)
        self.assertGreater(stats['area_ha'], 0)
        
        # Percentages should sum to 100%
        total_pct = (stats['degraded_pct'] + stats['stable_pct'] + 
                    stats['improved_pct'] + stats['nodata_pct'])
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
            config.SOC_DEG_BAND_NAME
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
            config.LC_DEG_COMPARISON_BAND_NAME
        ]
        
        for band_name in sdg_bands:
            masked = np.ma.MaskedArray(self.test_array_small, mask=False)
            stats = land_deg_stats._get_stats_for_band(
                band_name, masked, self.cell_areas_small, self.nodata
            )
            
            self.assertIn('degraded_pct', stats)
            self.assertIn('stable_pct', stats)
            self.assertIn('improved_pct', stats)
            self.assertIn('nodata_pct', stats)
        
        # Test productivity bands
        prod_bands = [
            config.JRC_LPD_BAND_NAME,
            config.FAO_WOCAT_LPD_BAND_NAME,
            config.TE_LPD_BAND_NAME,
            config.PROD_DEG_COMPARISON_BAND_NAME
        ]
        
        for band_name in prod_bands:
            masked = np.ma.MaskedArray(self.prod_array, mask=False)
            stats = land_deg_stats._get_stats_for_band(
                band_name, masked, self.cell_areas_small, 0
            )
            
            self.assertIn('degraded_pct', stats)
            self.assertIn('stable_pct', stats)
            self.assertIn('improved_pct', stats)
            self.assertIn('nodata_pct', stats)
        
        # Test SOC band
        masked = np.ma.MaskedArray(self.soc_array, mask=False)
        stats = land_deg_stats._get_stats_for_band(
            config.SOC_DEG_BAND_NAME, masked, self.cell_areas_small, self.nodata
        )
        
        self.assertIn('degraded_pct', stats)
        self.assertIn('stable_pct', stats)
        self.assertIn('improved_pct', stats)
        self.assertIn('nodata_pct', stats)


if __name__ == '__main__':
    unittest.main()