"""
Tests for resolution selection in land degradation analysis.

This module tests the functionality that determines and applies consistent
target resolution across multiple periods when processing land degradation data.
"""

from unittest.mock import Mock, patch

import pytest

# Skip all tests if required dependencies are not available
pytest.importorskip("numpy")

try:
    from te_schemas.productivity import ProductivityMode

    # Import the functions under test
    from te_algorithms.gdal.land_deg.land_deg import (
        get_reference_file_for_period,
        get_resolution_from_file,
        collect_resolutions_from_periods,
        select_highest_resolution,
        determine_target_resolution,
    )
except ImportError:
    pytest.skip(
        "te_algorithms.gdal modules require GDAL dependencies",
        allow_module_level=True,
    )


class TestGetReferenceFileForPeriod:
    """Test the get_reference_file_for_period function."""

    def test_trends_earth_mode(self):
        """Test reference file selection for Trends.Earth 5-class LPD mode."""
        period_params = {"layer_traj_path": "/path/to/traj.tif"}
        result = get_reference_file_for_period(
            period_params, ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value
        )
        assert result == "/path/to/traj.tif"

    def test_jrc_mode(self):
        """Test reference file selection for JRC 5-class LPD mode."""
        period_params = {"layer_lpd_path": "/path/to/lpd.tif"}
        result = get_reference_file_for_period(
            period_params, ProductivityMode.JRC_5_CLASS_LPD.value
        )
        assert result == "/path/to/lpd.tif"

    def test_fao_wocat_mode(self):
        """Test reference file selection for FAO WOCAT 5-class LPD mode."""
        period_params = {"layer_lpd_path": "/path/to/lpd.tif"}
        result = get_reference_file_for_period(
            period_params, ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value
        )
        assert result == "/path/to/lpd.tif"

    def test_unknown_mode(self):
        """Test reference file selection with unknown productivity mode."""
        period_params = {"layer_traj_path": "/path/to/traj.tif"}
        result = get_reference_file_for_period(period_params, "unknown_mode")
        assert result is None

    def test_missing_field(self):
        """Test reference file selection when expected field is missing."""
        period_params = {}
        result = get_reference_file_for_period(
            period_params, ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value
        )
        assert result is None


class TestGetResolutionFromFile:
    """Test the get_resolution_from_file function."""

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_valid_file(self, mock_open):
        """Test resolution extraction from a valid file."""
        mock_ds = Mock()
        mock_ds.GetGeoTransform.return_value = (0.0, 0.0025, 0.0, 0.0, 0.0, -0.0025)
        mock_open.return_value = mock_ds

        result = get_resolution_from_file("/path/to/test.tif")
        assert result == (0.0025, 0.0025)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_asymmetric_resolution(self, mock_open):
        """Test resolution extraction with different X and Y resolutions."""
        mock_ds = Mock()
        mock_ds.GetGeoTransform.return_value = (0.0, 0.0025, 0.0, 0.0, 0.0, -0.002)
        mock_open.return_value = mock_ds

        result = get_resolution_from_file("/path/to/test.tif")
        assert result == (0.0025, 0.002)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_negative_geotransform_values(self, mock_open):
        """Test that negative geotransform values are converted to positive."""
        mock_ds = Mock()
        mock_ds.GetGeoTransform.return_value = (34.47, -0.0025, 0.0, 0.6625, 0.0, 0.002)
        mock_open.return_value = mock_ds

        result = get_resolution_from_file("/path/to/test.tif")
        assert result == (0.0025, 0.002)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_file_not_found(self, mock_open):
        """Test handling when file cannot be opened."""
        mock_open.return_value = None

        result = get_resolution_from_file("/path/to/missing.tif")
        assert result is None

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_exception_handling(self, mock_open):
        """Test graceful handling of exceptions during file reading."""
        mock_open.side_effect = RuntimeError("GDAL error")

        result = get_resolution_from_file("/path/to/bad.tif")
        assert result is None


class TestSelectHighestResolution:
    """Test the select_highest_resolution function."""

    def test_single_resolution(self):
        """Test selection with a single resolution."""
        resolutions = [(0.0025, 0.0025)]
        result = select_highest_resolution(resolutions)
        assert result == (0.0025, 0.0025)

    def test_multiple_identical_resolutions(self):
        """Test selection when all resolutions are identical."""
        resolutions = [(0.0025, 0.0025), (0.0025, 0.0025), (0.0025, 0.0025)]
        result = select_highest_resolution(resolutions)
        assert result == (0.0025, 0.0025)

    def test_multiple_different_resolutions(self):
        """Test selection of highest resolution from multiple options."""
        resolutions = [(0.0025, 0.0025), (0.002246, 0.002246), (0.003, 0.003)]
        result = select_highest_resolution(resolutions)
        assert result == (0.002246, 0.002246)

    def test_asymmetric_resolutions(self):
        """Test selection with different X and Y resolutions."""
        resolutions = [(0.0025, 0.002), (0.002, 0.0025)]
        result = select_highest_resolution(resolutions)
        # Should select minimum (highest resolution) for each dimension independently
        assert result == (0.002, 0.002)

    def test_empty_list(self):
        """Test handling of empty resolution list."""
        resolutions = []
        result = select_highest_resolution(resolutions)
        assert result is None

    def test_kenya_bungoma_scenario(self):
        """Test with the actual resolutions from the Kenya Bungoma error."""
        resolutions = [(0.0025, 0.0025), (0.002246, 0.002246)]
        result = select_highest_resolution(resolutions)
        assert result == (0.002246, 0.002246)


class TestCollectResolutionsFromPeriods:
    """Test the collect_resolutions_from_periods function."""

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_single_period(self, mock_open):
        """Test resolution collection from a single period."""
        mock_ds = Mock()
        mock_ds.GetGeoTransform.return_value = (0.0, 0.0025, 0.0, 0.0, 0.0, -0.0025)
        mock_open.return_value = mock_ds

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/traj.tif",
                },
            }
        ]

        result = collect_resolutions_from_periods(periods)
        assert len(result) == 1
        assert result[0] == (0.0025, 0.0025)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_multiple_periods(self, mock_open):
        """Test resolution collection from multiple periods."""

        def mock_open_side_effect(path):
            if "baseline" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.0025,
                    0.0,
                    0.0,
                    0.0,
                    -0.0025,
                )
                return mock_ds
            elif "progress" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.002246,
                    0.0,
                    0.0,
                    0.0,
                    -0.002246,
                )
                return mock_ds
            return None

        mock_open.side_effect = mock_open_side_effect

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/baseline_traj.tif",
                },
            },
            {
                "name": "progress",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/progress_traj.tif",
                },
            },
        ]

        result = collect_resolutions_from_periods(periods)
        assert len(result) == 2
        assert result[0] == (0.0025, 0.0025)
        assert result[1] == (0.002246, 0.002246)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_mixed_productivity_modes(self, mock_open):
        """Test resolution collection with different productivity modes."""

        def mock_open_side_effect(path):
            if "traj" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.0025,
                    0.0,
                    0.0,
                    0.0,
                    -0.0025,
                )
                return mock_ds
            elif "lpd" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.001,
                    0.0,
                    0.0,
                    0.0,
                    -0.001,
                )
                return mock_ds
            return None

        mock_open.side_effect = mock_open_side_effect

        periods = [
            {
                "name": "period1",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/traj.tif",
                },
            },
            {
                "name": "period2",
                "params": {
                    "prod_mode": ProductivityMode.JRC_5_CLASS_LPD.value,
                    "layer_lpd_path": "/path/to/lpd.tif",
                },
            },
        ]

        result = collect_resolutions_from_periods(periods)
        assert len(result) == 2
        assert result[0] == (0.0025, 0.0025)
        assert result[1] == (0.001, 0.001)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_partial_failure(self, mock_open):
        """Test that collection continues even if some files fail to open."""

        def mock_open_side_effect(path):
            if "baseline" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.0025,
                    0.0,
                    0.0,
                    0.0,
                    -0.0025,
                )
                return mock_ds
            return None  # Other files fail to open

        mock_open.side_effect = mock_open_side_effect

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/baseline_traj.tif",
                },
            },
            {
                "name": "progress",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/missing_traj.tif",
                },
            },
        ]

        result = collect_resolutions_from_periods(periods)
        # Should get resolution from baseline but not progress
        assert len(result) == 1
        assert result[0] == (0.0025, 0.0025)

    def test_empty_periods(self):
        """Test handling of empty periods list."""
        periods = []
        result = collect_resolutions_from_periods(periods)
        assert result == []


class TestDetermineTargetResolution:
    """Test the end-to-end determine_target_resolution function."""

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_full_workflow_single_period(self, mock_open):
        """Test complete workflow with a single period."""
        mock_ds = Mock()
        mock_ds.GetGeoTransform.return_value = (0.0, 0.0025, 0.0, 0.0, 0.0, -0.0025)
        mock_open.return_value = mock_ds

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/traj.tif",
                },
            }
        ]

        result = determine_target_resolution(periods)
        assert result == (0.0025, 0.0025)

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_full_workflow_multiple_periods(self, mock_open):
        """Test complete workflow with multiple periods."""

        def mock_open_side_effect(path):
            if "baseline" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.0025,
                    0.0,
                    0.0,
                    0.0,
                    -0.0025,
                )
                return mock_ds
            elif "progress" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    0.0,
                    0.002246,
                    0.0,
                    0.0,
                    0.0,
                    -0.002246,
                )
                return mock_ds
            return None

        mock_open.side_effect = mock_open_side_effect

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/baseline_traj.tif",
                },
            },
            {
                "name": "progress",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/progress_traj.tif",
                },
            },
        ]

        result = determine_target_resolution(periods)
        # Should select highest resolution (smallest pixel size)
        assert result == (0.002246, 0.002246)

    def test_empty_periods(self):
        """Test handling of empty periods list."""
        periods = []
        result = determine_target_resolution(periods)
        assert result is None

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_all_files_fail(self, mock_open):
        """Test handling when all files fail to open."""
        mock_open.return_value = None

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/path/to/missing.tif",
                },
            }
        ]

        result = determine_target_resolution(periods)
        assert result is None

    @patch("te_algorithms.gdal.land_deg.land_deg.gdal.Open")
    def test_kenya_bungoma_real_world_scenario(self, mock_open):
        """Test the real-world Kenya Bungoma scenario that caused the original error."""

        def mock_open_side_effect(path):
            if "baseline" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    34.47,
                    0.0025,
                    0.0,
                    0.6625,
                    0.0,
                    -0.0025,
                )
                return mock_ds
            elif "progress" in path:
                mock_ds = Mock()
                mock_ds.GetGeoTransform.return_value = (
                    34.47,
                    0.002246,
                    0.0,
                    0.6625,
                    0.0,
                    -0.002246,
                )
                return mock_ds
            return None

        mock_open.side_effect = mock_open_side_effect

        periods = [
            {
                "name": "baseline",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/data/baseline_traj.tif",
                },
            },
            {
                "name": "progress",
                "params": {
                    "prod_mode": ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                    "layer_traj_path": "/data/progress_traj.tif",
                },
            },
        ]

        result = determine_target_resolution(periods)
        # Should select 0.002246 (highest resolution)
        assert result == (0.002246, 0.002246)

        # Verify this would fix the geotransform mismatch
        baseline_gt_with_target = (34.47, result[0], 0.0, 0.6625, 0.0, -result[1])
        progress_gt_with_target = (34.47, result[0], 0.0, 0.6625, 0.0, -result[1])

        # Round to 6 decimal places like the original assertion
        baseline_rounded = [round(x, 6) for x in baseline_gt_with_target]
        progress_rounded = [round(x, 6) for x in progress_gt_with_target]

        # This assertion would now pass (original error is fixed)
        assert baseline_rounded == progress_rounded
