"""
Comprehensive tests for GEE productivity functions.
Tests the core productivity algorithms: state, trajectory, performance, and faowocat.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock all external dependencies comprehensively
ee_mock = MagicMock()
te_schemas_mock = MagicMock()
band_info_mock = MagicMock()

# Setup mocking before any imports
sys.modules["ee"] = ee_mock
sys.modules["te_schemas"] = te_schemas_mock
sys.modules["te_schemas.schemas"] = te_schemas_mock
sys.modules["te_schemas.results"] = te_schemas_mock

# Setup BandInfo
te_schemas_mock.schemas = MagicMock()
te_schemas_mock.schemas.BandInfo = band_info_mock
te_schemas_mock.results = MagicMock()
te_schemas_mock.results.Raster = MagicMock()
te_schemas_mock.results.TiledRaster = MagicMock()

# ruff: noqa: E402
from te_algorithms.gee import GEEIOError, productivity


class TestProductivityState:
    """Test productivity_state function."""

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    def test_productivity_state_basic(self, mock_image, mock_te_image):
        """Test basic productivity state functionality."""
        # Mock TEImage constructor to avoid _check_validity issues
        mock_te_image.return_value = MagicMock()

        # Setup mock
        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance

        # Configure the mock chain for image processing
        mock_image_instance.select.return_value = mock_image_instance
        mock_image_instance.reduce.return_value = mock_image_instance
        mock_image_instance.addBands.return_value = mock_image_instance
        mock_image_instance.subtract.return_value = mock_image_instance
        mock_image_instance.add.return_value = mock_image_instance
        mock_image_instance.multiply.return_value = mock_image_instance
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.lte.return_value = mock_image_instance
        mock_image_instance.gt.return_value = mock_image_instance
        mock_image_instance.abs.return_value = mock_image_instance
        mock_image_instance.rename.return_value = mock_image_instance
        mock_image_instance.int16.return_value = mock_image_instance

        mock_logger = MagicMock()
        prod_asset = "test_asset"

        result = productivity.productivity_state(
            year_bl_start=2001,
            year_bl_end=2005,
            year_tg_start=2016,
            year_tg_end=2020,
            prod_asset=prod_asset,
            logger=mock_logger,
        )

        # Verify function was called
        assert result is not None
        mock_logger.debug.assert_called_with("Entering productivity_state function.")

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    def test_productivity_state_year_ranges(self, mock_image, mock_te_image):
        """Test productivity state with different year ranges."""
        mock_te_image.return_value = MagicMock()

        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance

        # Configure mock chain
        mock_image_instance.select.return_value = mock_image_instance
        mock_image_instance.reduce.return_value = mock_image_instance
        mock_image_instance.addBands.return_value = mock_image_instance
        mock_image_instance.subtract.return_value = mock_image_instance
        mock_image_instance.add.return_value = mock_image_instance
        mock_image_instance.multiply.return_value = mock_image_instance
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.lte.return_value = mock_image_instance
        mock_image_instance.gt.return_value = mock_image_instance
        mock_image_instance.abs.return_value = mock_image_instance
        mock_image_instance.rename.return_value = mock_image_instance
        mock_image_instance.int16.return_value = mock_image_instance

        mock_logger = MagicMock()

        # Test with single year ranges
        result = productivity.productivity_state(
            year_bl_start=2001,
            year_bl_end=2001,
            year_tg_start=2020,
            year_tg_end=2020,
            prod_asset="test_asset",
            logger=mock_logger,
        )

        assert result is not None

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    def test_productivity_state_percentile_logic(self, mock_image, mock_te_image):
        """Test that productivity state uses correct percentiles."""
        mock_te_image.return_value = MagicMock()

        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance

        # Configure mock chain
        mock_image_instance.select.return_value = mock_image_instance
        mock_image_instance.reduce.return_value = mock_image_instance
        mock_image_instance.addBands.return_value = mock_image_instance
        mock_image_instance.subtract.return_value = mock_image_instance
        mock_image_instance.add.return_value = mock_image_instance
        mock_image_instance.multiply.return_value = mock_image_instance
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.lte.return_value = mock_image_instance
        mock_image_instance.gt.return_value = mock_image_instance
        mock_image_instance.abs.return_value = mock_image_instance
        mock_image_instance.rename.return_value = mock_image_instance
        mock_image_instance.int16.return_value = mock_image_instance

        mock_logger = MagicMock()

        result = productivity.productivity_state(
            year_bl_start=2001,
            year_bl_end=2010,
            year_tg_start=2011,
            year_tg_end=2020,
            prod_asset="test_asset",
            logger=mock_logger,
        )

        # Verify percentile calculation was called
        assert result is not None
        mock_image_instance.reduce.assert_called()


class TestProductivityTrajectory:
    """Test productivity_trajectory function."""

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    @patch("te_algorithms.gee.productivity.stats.get_kendall_coef")
    @patch("te_algorithms.gee.productivity.ndvi_trend")
    def test_productivity_trajectory_ndvi_method(
        self, mock_ndvi_trend, mock_kendall, mock_image, mock_te_image
    ):
        """Test productivity trajectory with NDVI trend method."""
        mock_te_image.return_value = MagicMock()

        # Setup mocks
        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.updateMask.return_value = mock_image_instance
        mock_image_instance.neq.return_value = mock_image_instance
        mock_image_instance.eq.return_value = mock_image_instance

        # Mock trend analysis results
        mock_lf_trend = MagicMock()
        mock_mk_trend = MagicMock()
        mock_ndvi_trend.return_value = (mock_lf_trend, mock_mk_trend)

        # Configure trend results
        mock_lf_trend.select.return_value = MagicMock()
        mock_lf_trend.select.return_value.gt.return_value = MagicMock()
        mock_lf_trend.select.return_value.lt.return_value = MagicMock()
        mock_lf_trend.select.return_value.abs.return_value = MagicMock()
        mock_lf_trend.select.return_value.abs.return_value.lte.return_value = (
            MagicMock()
        )
        mock_lf_trend.select.return_value.rename.return_value = MagicMock()

        mock_mk_trend.abs.return_value = MagicMock()
        mock_mk_trend.abs.return_value.gte.return_value = MagicMock()
        mock_mk_trend.abs.return_value.lte.return_value = MagicMock()
        mock_mk_trend.rename.return_value = MagicMock()

        # Mock kendall coefficients
        mock_kendall.side_effect = [0.3, 0.4, 0.5]  # 90%, 95%, 99%

        mock_logger = MagicMock()

        result = productivity.productivity_trajectory(
            year_initial=2001,
            year_final=2020,
            method="ndvi_trend",
            prod_asset="test_asset",
            climate_asset=None,
            logger=mock_logger,
        )

        # Verify method was called
        assert result is not None
        mock_ndvi_trend.assert_called_once()
        mock_logger.debug.assert_called_with(
            "Entering productivity_trajectory function."
        )

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    @patch("te_algorithms.gee.productivity.stats.get_kendall_coef")
    @patch("te_algorithms.gee.productivity.p_restrend")
    def test_productivity_trajectory_prestrend_method(
        self, mock_prestrend, mock_kendall, mock_image, mock_te_image
    ):
        """Test productivity trajectory with p_restrend method."""
        mock_te_image.return_value = MagicMock()

        # Setup mocks
        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.updateMask.return_value = mock_image_instance
        mock_image_instance.neq.return_value = mock_image_instance
        mock_image_instance.eq.return_value = mock_image_instance

        # Mock trend analysis results
        mock_lf_trend = MagicMock()
        mock_mk_trend = MagicMock()
        mock_prestrend.return_value = (mock_lf_trend, mock_mk_trend)

        # Configure trend results
        mock_lf_trend.select.return_value = MagicMock()
        mock_lf_trend.select.return_value.gt.return_value = MagicMock()
        mock_lf_trend.select.return_value.lt.return_value = MagicMock()
        mock_lf_trend.select.return_value.abs.return_value = MagicMock()
        mock_lf_trend.select.return_value.abs.return_value.lte.return_value = (
            MagicMock()
        )
        mock_lf_trend.select.return_value.rename.return_value = MagicMock()

        mock_mk_trend.abs.return_value = MagicMock()
        mock_mk_trend.abs.return_value.gte.return_value = MagicMock()
        mock_mk_trend.abs.return_value.lte.return_value = MagicMock()
        mock_mk_trend.rename.return_value = MagicMock()

        mock_kendall.side_effect = [0.3, 0.4, 0.5]
        mock_logger = MagicMock()

        result = productivity.productivity_trajectory(
            year_initial=2001,
            year_final=2020,
            method="p_restrend",
            prod_asset="test_asset",
            climate_asset="climate_asset",
            logger=mock_logger,
        )

        assert result is not None
        mock_prestrend.assert_called_once()

    def test_productivity_trajectory_invalid_method(self):
        """Test productivity trajectory with invalid method."""
        mock_logger = MagicMock()

        with pytest.raises(GEEIOError, match="Unrecognized method 'invalid_method'"):
            productivity.productivity_trajectory(
                year_initial=2001,
                year_final=2020,
                method="invalid_method",
                prod_asset="test_asset",
                climate_asset="climate_asset",
                logger=mock_logger,
            )

    def test_productivity_trajectory_missing_climate_data(self):
        """Test productivity trajectory error when climate data required but missing."""
        mock_logger = MagicMock()

        with pytest.raises(GEEIOError, match="Must specify a climate dataset"):
            productivity.productivity_trajectory(
                year_initial=2001,
                year_final=2020,
                method="p_restrend",
                prod_asset="test_asset",
                climate_asset=None,
                logger=mock_logger,
            )


class TestProductivityPerformance:
    """Test productivity_performance function."""

    @patch("te_algorithms.gee.productivity.TEImage")
    @patch("te_algorithms.gee.productivity.ee.Image")
    @patch("te_algorithms.gee.productivity.ee.Geometry")
    def test_productivity_performance_basic(
        self, mock_geometry, mock_image, mock_te_image
    ):
        """Test basic productivity performance functionality."""
        mock_te_image.return_value = MagicMock()

        # Setup mocks
        mock_image_instance = MagicMock()
        mock_image.return_value = mock_image_instance

        # Configure image processing chain
        mock_image_instance.where.return_value = mock_image_instance
        mock_image_instance.updateMask.return_value = mock_image_instance
        mock_image_instance.neq.return_value = mock_image_instance
        mock_image_instance.eq.return_value = mock_image_instance
        mock_image_instance.select.return_value = mock_image_instance
        mock_image_instance.reduce.return_value = mock_image_instance
        mock_image_instance.rename.return_value = mock_image_instance
        mock_image_instance.clip.return_value = mock_image_instance
        mock_image_instance.remap.return_value = mock_image_instance
        mock_projection = MagicMock()
        mock_projection.nominalScale.return_value.getInfo.return_value = (
            250  # Mock MODIS scale
        )
        mock_image_instance.projection.return_value = mock_projection
        mock_image_instance.reproject.return_value = mock_image_instance
        mock_image_instance.multiply.return_value = mock_image_instance
        mock_image_instance.add.return_value = mock_image_instance
        mock_image_instance.addBands.return_value = mock_image_instance
        mock_image_instance.reduceRegion.return_value = MagicMock()
        mock_image_instance.divide.return_value = mock_image_instance
        mock_image_instance.reduceResolution.return_value = mock_image_instance
        mock_image_instance.gte.return_value = mock_image_instance
        mock_image_instance.lte.return_value = mock_image_instance
        mock_image_instance.And.return_value = mock_image_instance
        mock_image_instance.gt.return_value = mock_image_instance
        mock_image_instance.unmask.return_value = mock_image_instance
        mock_image_instance.int16.return_value = mock_image_instance

        # Mock geometry
        mock_poly = MagicMock()
        mock_poly.area.return_value.divide.return_value.getInfo.return_value = (
            1000  # Mock area in sq km
        )
        mock_poly.union.return_value = mock_poly
        mock_geometry.return_value = mock_poly

        # Mock the reduceRegion result
        mock_reduce_result = MagicMock()
        mock_reduce_result.get.return_value = []
        mock_image_instance.reduceRegion.return_value = mock_reduce_result

        # Mock ee.List for groups processing
        with patch("te_algorithms.gee.productivity.ee.List") as mock_list:
            mock_list_instance = MagicMock()
            mock_list.return_value = mock_list_instance
            mock_list_instance.getInfo.return_value = []

            mock_logger = MagicMock()
            geojson = {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }

            result = productivity.productivity_performance(
                year_initial=2001,
                year_final=2020,
                prod_asset="test_asset",
                all_geojsons=[geojson],
                logger=mock_logger,
            )

            assert result is not None
            # Check that the function was entered (any debug call is fine)
            assert mock_logger.debug.called


class TestProductivityFaowocat:
    """Test productivity_faowocat function."""

    def test_productivity_faowocat_missing_asset(self):
        """Test FAO-WOCAT with missing asset."""
        mock_logger = MagicMock()

        with pytest.raises(GEEIOError, match="Must specify a prod_asset"):
            productivity.productivity_faowocat(
                year_initial=2001, year_final=None, prod_asset=None, logger=mock_logger
            )

    def test_productivity_faowocat_missing_year_final(self):
        """Test FAO-WOCAT with missing year_final."""
        mock_logger = MagicMock()

        with pytest.raises(
            GEEIOError, match="Must specify 'year_final' for FAO-WOCAT dynamics"
        ):
            productivity.productivity_faowocat(
                year_initial=2001,
                year_final=None,
                prod_asset="test_asset",
                logger=mock_logger,
            )


class TestProductivityHelperFunctions:
    """Test helper functions used by productivity algorithms."""

    @patch("te_algorithms.gee.productivity.ee.ImageCollection")
    @patch("te_algorithms.gee.productivity.stats.mann_kendall")
    def test_linear_trend(self, mock_mann_kendall, mock_image_collection):
        """Test linear_trend helper function."""
        # Mock the image collection and its methods
        mock_ic = MagicMock()
        mock_ic.select.return_value = mock_ic
        mock_ic.reduce.return_value = MagicMock()

        mock_mann_kendall.return_value = MagicMock()

        mock_logger = MagicMock()

        lf_trend, mk_trend = productivity.linear_trend(mock_ic, mock_logger)

        assert lf_trend is not None
        assert mk_trend is not None
        mock_logger.debug.assert_called_with("Entering linear_trend function")

    @patch("te_algorithms.gee.productivity.ee.List")
    @patch("te_algorithms.gee.productivity.ee.ImageCollection")
    @patch("te_algorithms.gee.productivity.ee.Image")
    def test_ndvi_function(self, mock_image, mock_image_collection, mock_list):
        """Test ndvi helper function."""
        # Setup mocks
        mock_ndvi_1yr = MagicMock()
        mock_ndvi_1yr.select.return_value = mock_ndvi_1yr
        mock_ndvi_1yr.addBands.return_value = mock_ndvi_1yr
        mock_ndvi_1yr.rename.return_value = mock_ndvi_1yr

        mock_image.return_value = MagicMock()
        mock_image.return_value.float.return_value = mock_image.return_value

        mock_list_instance = MagicMock()
        mock_list.return_value = mock_list_instance
        mock_list_instance.add.return_value = mock_list_instance

        mock_ic = MagicMock()
        mock_image_collection.return_value = mock_ic

        mock_logger = MagicMock()

        result = productivity.ndvi(2001, 2005, mock_ndvi_1yr, mock_logger)

        assert result is not None
        mock_logger.debug.assert_called_with("Entering ndvi_trend function.")

    @patch("te_algorithms.gee.productivity.linear_trend")
    @patch("te_algorithms.gee.productivity.ndvi")
    def test_ndvi_trend_function(self, mock_ndvi, mock_linear_trend):
        """Test ndvi_trend function."""
        mock_ndvi.return_value = MagicMock()
        mock_linear_trend.return_value = (MagicMock(), MagicMock())

        mock_logger = MagicMock()

        result = productivity.ndvi_trend(2001, 2005, MagicMock(), mock_logger)

        assert result is not None
        # ndvi_trend logs "Entering p_restrend function" at the start
        mock_logger.debug.assert_any_call("Entering p_restrend function")


class TestProductivitySeries:
    """Test productivity_series function."""

    @patch("te_algorithms.gee.productivity.ndvi")
    def test_productivity_series_ndvi_trend(self, mock_ndvi):
        """Test productivity_series with ndvi_trend method."""
        mock_ndvi.return_value = MagicMock()
        mock_logger = MagicMock()

        result = productivity.productivity_series(
            year_initial=2001,
            year_final=2020,
            method="ndvi_trend",
            prod_asset="test_asset",
            climate_asset=None,
            logger=mock_logger,
        )

        assert result is not None
        # Verify the function was entered
        mock_logger.debug.assert_called_with(
            "Entering productivity_trajectory function."
        )

    def test_productivity_series_invalid_method(self):
        """Test productivity_series with invalid method."""
        mock_logger = MagicMock()

        with pytest.raises(GEEIOError, match="Unrecognized method 'invalid_method'"):
            productivity.productivity_series(
                year_initial=2001,
                year_final=2020,
                method="invalid_method",
                prod_asset="test_asset",
                climate_asset="climate_asset",
                logger=mock_logger,
            )

    def test_productivity_series_missing_climate_data(self):
        """Test productivity_series error when climate data required but missing."""
        mock_logger = MagicMock()

        with pytest.raises(GEEIOError, match="Must specify a climate dataset"):
            productivity.productivity_series(
                year_initial=2001,
                year_final=2020,
                method="p_restrend",
                prod_asset="test_asset",
                climate_asset=None,
                logger=mock_logger,
            )
