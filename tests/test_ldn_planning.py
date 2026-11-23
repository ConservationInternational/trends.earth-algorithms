"""Unit tests for te_algorithms.gdal.land_deg.ldn_planning.

Uses small synthetic rasters created in memory so no GDAL file I/O is needed
for the algorithmic logic (BAU), and temp files for the raster-writing tests.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal", reason="GDAL not available")
pytest.importorskip("osgeo.ogr", reason="OGR not available")
pytest.importorskip("osgeo.osr", reason="OSR not available")

from osgeo import gdal, ogr, osr

from te_algorithms.gdal.land_deg import ldn_planning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raster(arr: np.ndarray, path: str, nodata: int = -32768) -> None:
    """Write a single-band Int16 raster to *path*."""
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = arr.shape
    ds = driver.Create(path, xsize, ysize, 1, gdal.GDT_Int16)
    # Small WGS84 extent centred on 0°N 0°E
    gt = (0.0, 0.1, 0.0, 5.0, 0.0, -0.1)  # 10 cols × 10 rows → 1°×1°
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds.FlushCache()
    del ds


def _write_grid(
    arr: np.ndarray,
    path: str,
    geotransform: tuple,
    epsg: int,
    nodata: int = -32768,
) -> None:
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = arr.shape
    ds = driver.Create(path, xsize, ysize, 1, gdal.GDT_Int16)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds.FlushCache()
    del ds


def _write_float_grid(
    arr: np.ndarray,
    path: str,
    geotransform: tuple,
    epsg: int,
    nodata: float = -9999.0,
) -> None:
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    ysize, xsize = arr.shape
    ds = driver.Create(path, xsize, ysize, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr.astype(np.float32))
    ds.FlushCache()
    del ds


def test_hotspots_uses_declared_encoding_for_ambiguous_values(tmp_path):
    status_path = str(tmp_path / "status.tif")
    _write_raster(np.ones((2, 2), dtype=np.int16), status_path)
    for status_type, expected in (("3class", 0.0), ("7class", 1.0)):
        summary, _, _ = ldn_planning.compute_hotspots(
            status_path,
            status_type=status_type,
            output_vector_path=str(tmp_path / f"{status_type}.gpkg"),
            output_raster_path=str(tmp_path / f"{status_type}.tif"),
        )
        assert summary["mean_degraded_fraction"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ARR classification tests
# ---------------------------------------------------------------------------


class TestARRClassification(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _status_path(self, arr: np.ndarray) -> str:
        path = os.path.join(self.tmpdir, "status.tif")
        _write_raster(arr, path)
        return path

    def test_3class_all_degraded(self):
        arr = np.full((5, 5), -1, dtype=np.int16)
        out = os.path.join(self.tmpdir, "arr_deg.tif")
        summary, path = ldn_planning.compute_arr_classification(
            self._status_path(arr), output_path=out
        )
        self.assertEqual(path, out)
        self.assertGreater(summary["reverse_km2"], 0)
        self.assertAlmostEqual(summary["avoid_km2"], 0.0, places=3)
        self.assertAlmostEqual(summary["reduce_km2"], 0.0, places=3)

    def test_3class_all_stable(self):
        arr = np.full((5, 5), 0, dtype=np.int16)
        out = os.path.join(self.tmpdir, "arr_stab.tif")
        summary, path = ldn_planning.compute_arr_classification(
            self._status_path(arr), output_path=out
        )
        # No trajectory → all non-degraded = Avoid
        self.assertGreater(summary["avoid_km2"], 0)
        self.assertAlmostEqual(summary["reduce_km2"], 0.0, places=3)
        self.assertAlmostEqual(summary["reverse_km2"], 0.0, places=3)

    def test_3class_mixed(self):
        arr = np.array(
            [
                [-1, 0, 1, 0, -1],
                [0, -1, 0, 1, 0],
                [1, 0, -1, 0, 1],
                [0, 1, 0, -1, 0],
                [-1, 0, 1, 0, -1],
            ],
            dtype=np.int16,
        )
        out = os.path.join(self.tmpdir, "arr_mix.tif")
        summary, path = ldn_planning.compute_arr_classification(
            self._status_path(arr), output_path=out
        )
        self.assertGreater(summary["reverse_km2"], 0)
        self.assertGreater(summary["avoid_km2"], 0)
        self.assertAlmostEqual(
            summary["avoid_km2"] + summary["reverse_km2"],
            summary["total_km2"],
            places=2,
        )

    def test_7class_degraded_values(self):
        # 7-class: 1=persistent deg, 2=recent deg, 3=baseline deg → all Reverse
        arr = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 4, 1, 2],
                [3, 4, 5, 6, 7],
                [1, 3, 5, 7, 4],
                [2, 4, 6, 1, 3],
            ],
            dtype=np.int16,
        )
        out = os.path.join(self.tmpdir, "arr_7cls.tif")
        summary, path = ldn_planning.compute_arr_classification(
            self._status_path(arr), output_path=out
        )
        self.assertGreater(summary["reverse_km2"], 0)
        self.assertEqual(summary["status_type"], "7class")

    def test_with_trajectory_adds_reduce(self):
        status = np.full((5, 5), 0, dtype=np.int16)  # all stable → Avoid without LPD
        # LPD class 3 = "stable but stressed": not degraded, but at risk → Reduce.
        # (Classes 1/2 are already degraded and would be Reverse, so they cannot
        # supply the at-risk signal.)
        traj = np.full((5, 5), 3, dtype=np.int16)

        s_path = os.path.join(self.tmpdir, "status_t.tif")
        t_path = os.path.join(self.tmpdir, "traj.tif")
        out = os.path.join(self.tmpdir, "arr_traj.tif")
        _write_raster(status, s_path)
        _write_raster(traj, t_path)
        summary, _ = ldn_planning.compute_arr_classification(
            s_path, output_path=out, trajectory_path=t_path
        )
        # With stressed LPD, all non-degraded stable → Reduce
        self.assertGreater(summary["reduce_km2"], 0)
        self.assertAlmostEqual(summary["avoid_km2"], 0.0, places=3)

    def test_lpd_declining_is_not_at_risk_signal(self):
        # LPD class 1 (Declining) is already degraded; when overall status is
        # non-degraded, class 1 must NOT create Reduce pixels (it is not a valid
        # non-degraded early-warning class). All should remain Avoid.
        status = np.full((5, 5), 0, dtype=np.int16)  # not degraded overall
        traj = np.full((5, 5), 1, dtype=np.int16)  # Declining (already-degraded class)

        s_path = os.path.join(self.tmpdir, "status_d.tif")
        t_path = os.path.join(self.tmpdir, "traj_d.tif")
        out = os.path.join(self.tmpdir, "arr_traj_d.tif")
        _write_raster(status, s_path)
        _write_raster(traj, t_path)
        summary, _ = ldn_planning.compute_arr_classification(
            s_path, output_path=out, trajectory_path=t_path
        )
        self.assertAlmostEqual(summary["reduce_km2"], 0.0, places=3)
        self.assertGreater(summary["avoid_km2"], 0)

    def test_nodata_excluded_from_summary(self):
        arr = np.full((5, 5), -32768, dtype=np.int16)
        out = os.path.join(self.tmpdir, "arr_nd.tif")
        summary, _ = ldn_planning.compute_arr_classification(
            self._status_path(arr), output_path=out
        )
        self.assertAlmostEqual(summary["total_km2"], 0.0, places=3)
        self.assertGreater(summary["nodata_km2"], 0)

    def test_auto_rejects_all_one_status_but_explicit_encoding_works(self):
        arr = np.ones((5, 5), dtype=np.int16)
        path = self._status_path(arr)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ldn_planning.compute_arr_classification(path)

        output = os.path.join(self.tmpdir, "arr_one_3class.tif")
        summary, _ = ldn_planning.compute_arr_classification(
            path, output_path=output, status_type="3class"
        )
        self.assertGreater(summary["avoid_km2"], 0)

        output_7class = os.path.join(self.tmpdir, "arr_one_7class.tif")
        summary, _ = ldn_planning.compute_arr_classification(
            path, output_path=output_7class, status_type="7class"
        )
        self.assertGreater(summary["reverse_km2"], 0)

    def test_shifted_trajectory_is_aligned_and_risk_nodata_is_unknown(self):
        status_path = os.path.join(self.tmpdir, "status_align.tif")
        risk_path = os.path.join(self.tmpdir, "risk_align.tif")
        output_path = os.path.join(self.tmpdir, "arr_align.tif")
        _write_grid(
            np.zeros((4, 4), dtype=np.int16),
            status_path,
            (0, 0.1, 0, 5, 0, -0.1),
            4326,
        )
        risk = np.full((4, 4), 1, dtype=np.int16)
        risk[1, 1] = -32768
        _write_grid(
            risk,
            risk_path,
            (0.05, 0.1, 0, 5, 0, -0.1),
            4326,
        )
        summary, _ = ldn_planning.compute_arr_classification(
            status_path,
            risk_layer_path=risk_path,
            output_path=output_path,
            status_type="3class",
        )
        result_ds = gdal.Open(output_path)
        result = result_ds.GetRasterBand(1).ReadAsArray()
        result_ds = None
        self.assertEqual(result[1, 1], ldn_planning.ARR_NODATA)
        reason_ds = gdal.Open(output_path)
        self.assertEqual(
            reason_ds.GetRasterBand(2).ReadAsArray()[1, 1],
            ldn_planning.ARR_REASON_MISSING_REQUIRED_INPUT,
        )
        reason_ds = None
        self.assertGreater(summary["reduce_km2"], 0)
        self.assertGreater(summary["nodata_km2"], 0)

    def test_float_risk_uses_strict_configured_threshold_and_reason_codes(self):
        status_path = os.path.join(self.tmpdir, "status_threshold.tif")
        risk_path = os.path.join(self.tmpdir, "risk_threshold.tif")
        output_path = os.path.join(self.tmpdir, "arr_threshold.tif")
        geotransform = (0, 0.1, 0, 5, 0, -0.1)
        _write_grid(np.zeros((2, 2), dtype=np.int16), status_path, geotransform, 4326)
        _write_float_grid(
            np.array([[0.25, 0.75], [0.5, 0.6]]),
            risk_path,
            geotransform,
            4326,
        )
        summary, _ = ldn_planning.compute_arr_classification(
            status_path,
            risk_layer_path=risk_path,
            output_path=output_path,
            status_type="3class",
            risk_threshold=0.5,
        )
        result_ds = gdal.Open(output_path)
        class_arr = result_ds.GetRasterBand(1).ReadAsArray()
        reason_arr = result_ds.GetRasterBand(2).ReadAsArray()
        result_ds = None
        self.assertEqual(np.count_nonzero(class_arr == ldn_planning.ARR_REDUCE), 2)
        self.assertEqual(summary["reason_band_index"], 2)
        self.assertEqual(
            summary["reason_counts"][ldn_planning.ARR_REASON_RISK_POSITIVE], 2
        )
        self.assertEqual(reason_arr[0, 1], ldn_planning.ARR_REASON_RISK_POSITIVE)

    def test_arr_cancellation_removes_partial_output(self):
        status_path = self._status_path(np.zeros((5, 5), dtype=np.int16))
        output_path = os.path.join(self.tmpdir, "cancelled_arr.tif")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            ldn_planning.compute_arr_classification(
                status_path,
                output_path=output_path,
                killed_callback=lambda: True,
            )
        self.assertFalse(os.path.exists(output_path))

    def test_aoi_masks_arr_and_projected_cell_area_is_physical(self):
        status_path = os.path.join(self.tmpdir, "status_aoi.tif")
        aoi_path = os.path.join(self.tmpdir, "aoi.gpkg")
        output_path = os.path.join(self.tmpdir, "arr_aoi.tif")
        _write_grid(
            np.zeros((2, 2), dtype=np.int16),
            status_path,
            (0, 0.1, 0, 5, 0, -0.1),
            4326,
        )
        driver = ogr.GetDriverByName("GPKG")
        ds = driver.CreateDataSource(aoi_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer("aoi", srs=srs, geom_type=ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in [(0, 5), (0.1, 5), (0.1, 4.9), (0, 4.9), (0, 5)]:
            ring.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)
        ds = None

        summary, _ = ldn_planning.compute_arr_classification(
            status_path,
            output_path=output_path,
            status_type="3class",
            aoi_path=aoi_path,
        )
        result_ds = gdal.Open(output_path)
        result = result_ds.GetRasterBand(1).ReadAsArray()
        result_ds = None
        self.assertEqual(np.count_nonzero(result == ldn_planning.ARR_AVOID), 1)
        self.assertAlmostEqual(summary["total_km2"], summary["avoid_km2"], places=5)

        projected = osr.SpatialReference()
        projected.ImportFromEPSG(3857)
        areas = ldn_planning._compute_pixel_areas_km2(
            (0, 1000, 0, 1000, 0, -1000),
            0,
            1,
            1,
            projected.ExportToWkt(),
        )
        self.assertAlmostEqual(float(areas[0, 0]), 1.0, delta=0.01)


# ---------------------------------------------------------------------------
# BAU rate tests
# ---------------------------------------------------------------------------


class TestBAURate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _assert_trajectory(self, summary, degraded, period, target_year):
        year_initial, year_final = period
        annual = degraded / (year_final - year_initial)
        projection = degraded + annual * (target_year - year_final)
        self.assertAlmostEqual(summary["annual_change_km2"], annual, places=6)
        self.assertAlmostEqual(
            summary[f"bau_projection_{target_year}_km2"], projection, places=6
        )
        self.assertAlmostEqual(
            summary["shortfall_km2"],
            max(0.0, projection - summary["degraded_area_baseline_km2"]),
            places=6,
        )
        self.assertAlmostEqual(
            summary["ldntarget_km2"], summary["degraded_area_baseline_km2"], places=6
        )

    def test_reporting_band_rate_uses_its_full_period(self):
        # Status degraded classes are cumulative since the baseline start, so
        # the rate is the reporting degraded area over 2001→2019 (18 years).
        baseline = np.array([[1, 2, 3], [4, 4, 4], [6, 7, 4]], dtype=np.int16)
        reporting = np.array([[1, 2, 3], [1, 2, 4], [6, 7, 4]], dtype=np.int16)
        p_bl = os.path.join(self.tmpdir, "bau_bl.tif")
        p_rp = os.path.join(self.tmpdir, "bau_rp.tif")
        _write_raster(baseline, p_bl)
        _write_raster(reporting, p_rp)
        summary = ldn_planning.compute_bau_rate(
            p_bl,
            1,
            p_rp,
            1,
            year_initial=2001,
            year_final=2019,
            target_year=2030,
            status_baseline_type="7class",
            status_reporting_type="7class",
        )
        deg_rp = summary["degraded_area_reporting_km2"]
        self.assertGreater(deg_rp, summary["degraded_area_baseline_km2"])
        self._assert_trajectory(summary, deg_rp, (2001, 2019), 2030)
        self.assertGreater(summary["shortfall_km2"], 0)

    def test_baseline_band_supplies_rate_without_reporting(self):
        arr = np.array([[-1, 0], [0, -1]], dtype=np.int16)
        p = os.path.join(self.tmpdir, "bau_norepobj.tif")
        _write_raster(arr, p)
        summary = ldn_planning.compute_bau_rate(
            p, 1, year_initial=2000, year_final=2015, target_year=2030
        )
        self.assertIsNone(summary["degraded_area_reporting_km2"])
        self._assert_trajectory(
            summary, summary["degraded_area_baseline_km2"], (2000, 2015), 2030
        )

    def test_invalid_period_has_no_projection(self):
        arr = np.array([[-1, 0], [0, -1]], dtype=np.int16)
        p = os.path.join(self.tmpdir, "bau_empty_period.tif")
        _write_raster(arr, p)
        summary = ldn_planning.compute_bau_rate(
            p, 1, year_initial=2015, year_final=2015
        )
        self.assertIsNone(summary["annual_change_km2"])
        self.assertIsNone(summary["bau_projection_2030_km2"])
        self.assertIsNone(summary["shortfall_km2"])

    def test_zone_rates_use_full_period(self):
        baseline = np.array([[-1, 0], [0, 0]], dtype=np.int16)
        reporting = np.array([[-1, -1], [-1, 0]], dtype=np.int16)
        zones = np.array([[1, 1], [2, 2]], dtype=np.int16)
        paths = {}
        for name, arr in (
            ("baseline", baseline),
            ("reporting", reporting),
            ("zones", zones),
        ):
            paths[name] = os.path.join(self.tmpdir, f"bau_zone_{name}.tif")
            _write_raster(arr, paths[name])
        summary = ldn_planning.compute_bau_rate(
            paths["baseline"],
            1,
            paths["reporting"],
            1,
            year_initial=2001,
            year_final=2019,
            target_year=2030,
            zones_raster_path=paths["zones"],
            zones_raster_labels={1: "north", 2: "south"},
        )
        zones_by_name = {z["zone_name"]: z for z in summary["jurisdictions"]}
        self.assertEqual(set(zones_by_name), {"north", "south"})
        for zone in zones_by_name.values():
            self._assert_trajectory(
                zone, zone["degraded_area_reporting_km2"], (2001, 2019), 2030
            )

    def test_explicit_baseline_and_reporting_encoding_and_aoi(self):
        path = os.path.join(self.tmpdir, "bau_all_improved.tif")
        _write_grid(
            np.ones((2, 2), dtype=np.int16),
            path,
            (0, 0.1, 0, 5, 0, -0.1),
            4326,
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ldn_planning.compute_bau_rate(path, 1, path, 1)

        aoi = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 5], [0.1, 5], [0.1, 4.9], [0, 4.9], [0, 5]]
                        ],
                    },
                    "properties": {},
                }
            ],
        }

        class AoiStub:
            def get_geojson(self):
                return aoi

        summary = ldn_planning.compute_bau_rate(
            path,
            1,
            path,
            1,
            status_baseline_type="3class",
            status_reporting_type="3class",
            aoi=AoiStub(),
        )
        self.assertAlmostEqual(summary["degraded_area_baseline_km2"], 0.0)
        expected_area = ldn_planning._compute_pixel_areas_km2(
            (0, 0.1, 0, 5, 0, -0.1), 0, 1, 1
        )[0, 0]
        self.assertAlmostEqual(summary["total_area_km2"], expected_area, places=5)


# ---------------------------------------------------------------------------
# ARR scenario tests
# ---------------------------------------------------------------------------


class TestApplyScenario(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # 5×5 ARR raster: rows 0-1 = Reverse, 2-3 = Reduce, 4 = Avoid
        arr = np.array(
            [
                [3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.int16,
        )
        self.arr_path = os.path.join(self.tmpdir, "arr.tif")
        _write_raster(arr, self.arr_path, nodata=-32768)

    def _full_wkt(self):
        # WKT covering the full 5×5 raster extent (0°N 0°E to 0.5°N 0.5°E)
        return "POLYGON ((0 0, 0.5 0, 0.5 5.0, 0 5.0, 0 0))"

    def test_reverse_generates_gains_only(self):
        targets = [
            {
                "wkt_geometry": self._full_wkt(),
                "intervention": "reverse",
                "effectiveness": 1.0,
            }
        ]
        out = os.path.join(self.tmpdir, "scen_rev.tif")
        summary, path = ldn_planning.apply_scenario(
            self.arr_path, targets=targets, output_path=out
        )
        self.assertGreater(summary["gains_km2_reverse"], 0)
        self.assertAlmostEqual(summary["avoided_losses_km2_reduce"], 0.0, places=3)
        self.assertAlmostEqual(summary["avoided_losses_km2_avoid"], 0.0, places=3)

    def test_reduce_generates_avoided_losses_not_gains(self):
        targets = [
            {
                "wkt_geometry": self._full_wkt(),
                "intervention": "reduce",
                "effectiveness": 1.0,
            }
        ]
        out = os.path.join(self.tmpdir, "scen_red.tif")
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path, targets=targets, output_path=out
        )
        self.assertAlmostEqual(summary["gains_km2_reverse"], 0.0, places=3)
        self.assertGreater(summary["avoided_losses_km2_reduce"], 0)
        self.assertAlmostEqual(summary["total_gains_km2"], 0.0, places=3)

    def test_avoid_generates_avoided_losses_not_gains(self):
        targets = [
            {
                "wkt_geometry": self._full_wkt(),
                "intervention": "avoid",
                "effectiveness": 1.0,
            }
        ]
        out = os.path.join(self.tmpdir, "scen_av.tif")
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path, targets=targets, output_path=out
        )
        self.assertAlmostEqual(summary["gains_km2_reverse"], 0.0, places=3)
        self.assertGreater(summary["avoided_losses_km2_avoid"], 0)

    def test_effectiveness_zero_yields_no_effect(self):
        targets = [
            {
                "wkt_geometry": self._full_wkt(),
                "intervention": "reverse",
                "effectiveness": 0.0,
            }
        ]
        out = os.path.join(self.tmpdir, "scen_eff0.tif")
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path, targets=targets, output_path=out
        )
        self.assertAlmostEqual(summary["total_gains_km2"], 0.0, places=3)

    def test_empty_targets(self):
        out = os.path.join(self.tmpdir, "scen_empty.tif")
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path, targets=[], output_path=out
        )
        self.assertAlmostEqual(summary["total_gains_km2"], 0.0, places=3)
        self.assertAlmostEqual(summary["total_avoided_losses_km2"], 0.0, places=3)

    def test_raster_mask_target_and_effectiveness_validation(self):
        selected_mask = os.path.join(self.tmpdir, "selected.tif")
        _write_raster(
            np.array(
                [
                    [3, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            selected_mask,
            nodata=-32768,
        )
        target = {
            "selected_mask_path": selected_mask,
            "band": 1,
            "intervention": "reverse",
            "effectiveness": 0.0,
        }
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path,
            targets=[target],
            output_path=os.path.join(self.tmpdir, "mask_scenario.tif"),
        )
        self.assertAlmostEqual(summary["total_gains_km2"], 0.0)
        self.assertGreater(summary["per_target"][0]["area_treated_km2"], 0)
        target["effectiveness"] = 1.5
        with self.assertRaisesRegex(ValueError, "effectiveness"):
            ldn_planning.apply_scenario(
                self.arr_path,
                targets=[target],
                output_path=os.path.join(self.tmpdir, "invalid_eff.tif"),
            )
        target["effectiveness"] = 1.0
        target["intervention"] = "reduce"
        with self.assertRaisesRegex(ValueError, "does not match"):
            ldn_planning.apply_scenario(
                self.arr_path,
                targets=[target],
                output_path=os.path.join(self.tmpdir, "mismatched_mask.tif"),
            )

    def test_multi_class_mask_targets_use_only_their_class(self):
        selected_mask = os.path.join(self.tmpdir, "selected_multi.tif")
        values = np.zeros((5, 5), dtype=np.int16)
        values[0, 0] = values[1, 1] = 3
        values[4, 4] = 1
        _write_raster(values, selected_mask, nodata=-32768)
        targets = [
            {
                "selected_mask_path": selected_mask,
                "intervention": intervention,
                "effectiveness": 1.0,
            }
            for intervention in ("reverse", "avoid")
        ]
        summary, _ = ldn_planning.apply_scenario(
            self.arr_path,
            targets=targets,
            output_path=os.path.join(self.tmpdir, "multi_mask.tif"),
        )
        reverse, avoid = summary["per_target"]
        self.assertGreater(reverse["area_treated_km2"], 0)
        self.assertGreater(avoid["area_treated_km2"], 0)
        # Two Reverse pixels and one Avoid pixel; geographic areas vary slightly.
        self.assertAlmostEqual(
            reverse["area_treated_km2"] / avoid["area_treated_km2"], 2, delta=0.01
        )

    def test_polygon_crs_and_aoi_are_transformed_to_arr_grid(self):
        polygon = ogr.CreateGeometryFromWkt(self._full_wkt())
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(3857)
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        polygon.Transform(osr.CoordinateTransformation(source_srs, target_srs))
        aoi = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 5], [0.1, 5], [0.1, 4.9], [0, 4.9], [0, 5]]
                        ],
                    },
                    "properties": {},
                }
            ],
        }

        class AoiStub:
            def get_geojson(self):
                return aoi

        summary, _ = ldn_planning.apply_scenario(
            self.arr_path,
            targets=[
                {
                    "wkt_geometry": polygon.ExportToWkt(),
                    "geometry_crs": target_srs.ExportToWkt(),
                    "intervention": "reverse",
                    "effectiveness": 1.0,
                }
            ],
            output_path=os.path.join(self.tmpdir, "scenario_crs_aoi.tif"),
            aoi=AoiStub(),
        )
        self.assertGreater(summary["gains_km2_reverse"], 0)
        self.assertAlmostEqual(
            summary["gains_km2_reverse"],
            summary["per_target"][0]["gains_km2"],
            places=5,
        )


# ---------------------------------------------------------------------------
# Report writer tests
# ---------------------------------------------------------------------------


class TestReportWriter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_excel_written(self):
        from pathlib import Path

        from te_algorithms.gdal.land_deg.ldn_planning_report import (
            save_ldn_planning_excel,
        )

        arr_summary = {
            "avoid_km2": 10.0,
            "reduce_km2": 5.0,
            "reverse_km2": 3.0,
            "total_km2": 18.0,
        }
        bau_summary = {
            "year_initial": 2000,
            "year_final": 2015,
            "target_year": 2030,
            "total_area_km2": 18.0,
            "degraded_area_baseline_km2": 3.0,
            "pct_degraded_baseline": 16.7,
            "degraded_area_reporting_km2": 4.0,
            "pct_degraded_reporting": 22.2,
            "annual_change_km2": 0.067,
            "bau_projection_2030_km2": 5.0,
            "ldntarget_km2": 3.0,
            "shortfall_km2": 2.0,
        }
        out = Path(self.tmpdir) / "test_report.xlsx"
        save_ldn_planning_excel(out, arr_summary=arr_summary, bau_summary=bau_summary)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_json_written(self):
        from pathlib import Path

        from te_algorithms.gdal.land_deg.ldn_planning_report import (
            save_ldn_planning_json,
        )

        arr = {"avoid_km2": 10.0, "reverse_km2": 3.0}
        out = Path(self.tmpdir) / "test_report.json"
        save_ldn_planning_json(out, task_name="Test", arr_summary=arr)
        self.assertTrue(out.exists())
        data = json.loads(out.read_text())
        self.assertEqual(data["task_name"], "Test")
        self.assertIn("arr", data)

    def test_scenario_breakdown_sheets_written(self):
        from pathlib import Path

        from te_algorithms.gdal.land_deg.ldn_planning_report import (
            save_ldn_planning_excel,
        )

        scenario_summary = {
            "gains_km2_reverse": 3.0,
            "avoided_losses_km2_reduce": 2.0,
            "avoided_losses_km2_avoid": 1.0,
            "total_gains_km2": 3.0,
            "total_avoided_losses_km2": 3.0,
            "net_balance_note": "test",
            "per_target": [],
            "by_land_type": [
                {
                    "name": "Cropland",
                    "code": 3,
                    "gains_km2": 2.0,
                    "avoided_losses_reduce_km2": 1.0,
                    "avoided_losses_avoid_km2": 0.5,
                    "total_gains_km2": 2.0,
                    "total_avoided_losses_km2": 1.5,
                },
            ],
            "by_jurisdiction": [
                {
                    "name": "Region A",
                    "code": 1,
                    "gains_km2": 1.0,
                    "avoided_losses_reduce_km2": 1.0,
                    "avoided_losses_avoid_km2": 0.5,
                    "total_gains_km2": 1.0,
                    "total_avoided_losses_km2": 1.5,
                },
            ],
        }
        out = Path(self.tmpdir) / "test_scenario_report.xlsx"
        save_ldn_planning_excel(out, scenario_summary=scenario_summary)
        self.assertTrue(out.exists())

        import openpyxl

        wb = openpyxl.load_workbook(out)
        self.assertIn("Scenario by Land Type", wb.sheetnames)
        self.assertIn("Scenario by Jurisdiction", wb.sheetnames)
        land_type_sheet = wb["Scenario by Land Type"]
        jurisdiction_sheet = wb["Scenario by Jurisdiction"]
        self.assertEqual(land_type_sheet.cell(4, 1).value, "Land Type")
        self.assertEqual(jurisdiction_sheet.cell(4, 1).value, "Jurisdiction")
        self.assertEqual(
            [cell.value for cell in land_type_sheet[5][:6]],
            ["Cropland", 2.0, 1.0, 0.5, 2.0, 1.5],
        )
        self.assertEqual(
            [cell.value for cell in jurisdiction_sheet[5][:6]],
            ["Region A", 1.0, 1.0, 0.5, 1.0, 1.5],
        )
        wb.close()

    def test_land_types_summary_and_report(self):
        from pathlib import Path

        import openpyxl

        from te_algorithms.gdal.land_deg.ldn_planning_report import (
            save_ldn_planning_excel,
            save_ldn_planning_json,
        )

        values = np.zeros((10, 10), dtype=np.int16)
        values[:4] = 1
        values[4:9] = 2
        values[9, :5] = 7
        path = os.path.join(self.tmpdir, "land_types.tif")
        _write_raster(values, path, nodata=-32768)
        summary = ldn_planning.summarize_land_types(
            path, {1: "1_1", 2: "1_2", 3: "2_2"}, ["cover", "soil"], block_rows=3
        )
        rows = {row["land_type_id"]: row for row in summary["land_types"]}
        self.assertEqual(summary["layer_names"], ["cover", "soil"])
        self.assertEqual(summary["n_land_types"], 4)
        self.assertEqual([rows[key]["pixels"] for key in (1, 2, 3, 7)], [40, 50, 0, 5])
        self.assertEqual(rows[2]["values"], ["1", "2"])
        self.assertEqual(rows[7]["label"], "7")
        self.assertAlmostEqual(
            sum(row["area_km2"] for row in rows.values()), summary["total_area_km2"]
        )
        self.assertAlmostEqual(sum(row["fraction"] for row in rows.values()), 1.0)
        self.assertAlmostEqual(
            summary["total_area_km2"],
            ldn_planning._compute_pixel_areas_km2(
                (0.0, 0.1, 0.0, 5.0, 0.0, -0.1), 0, 10, 10
            )[values != 0].sum(),
        )

        json_path = Path(self.tmpdir) / "land_types.json"
        report = save_ldn_planning_json(json_path, land_types_summary=summary)
        self.assertEqual(json.loads(json_path.read_text())["land_types"], summary)
        self.assertEqual(report["land_types"], summary)
        excel_path = Path(self.tmpdir) / "land_types.xlsx"
        save_ldn_planning_excel(excel_path, land_types_summary=summary)
        workbook = openpyxl.load_workbook(excel_path)
        sheet = workbook["Land Types"]
        self.assertEqual([cell.value for cell in sheet[6][:3]], [1, "1_1", 40])
        workbook.close()


class TestBauScenarioComparison(unittest.TestCase):
    """Pure-Python tests for project_scenario_against_bau (no GDAL needed)."""

    def _bau(self):
        return {
            "year_initial": 2000,
            "year_final": 2015,
            "target_year": 2030,
            "degraded_area_baseline_km2": 100.0,
            "degraded_area_reporting_km2": 120.0,
            "bau_projection_2030_km2": 150.0,
            "ldntarget_km2": 100.0,
            "shortfall_km2": 50.0,
            "zones": [],
        }

    def test_no_reporting_returns_none(self):
        from te_algorithms.gdal.land_deg.ldn_planning import (
            project_scenario_against_bau,
        )

        bau = self._bau()
        bau["bau_projection_2030_km2"] = None
        result = project_scenario_against_bau(bau, {}, 2030)
        self.assertIsNone(result)

    def test_partial_gap_close(self):
        from te_algorithms.gdal.land_deg.ldn_planning import (
            project_scenario_against_bau,
        )

        scenario = {"total_gains_km2": 10.0, "total_avoided_losses_km2": 20.0}
        p = project_scenario_against_bau(self._bau(), scenario, 2030)
        # BAU projection 150, contribution 30 -> adjusted 120, target 100
        self.assertAlmostEqual(p["scenario_degraded_km2"], 120.0)
        self.assertAlmostEqual(p["bau_shortfall_km2"], 50.0)
        self.assertAlmostEqual(p["remaining_shortfall_km2"], 20.0)
        self.assertAlmostEqual(p["gap_closed_pct"], 60.0)
        self.assertFalse(p["neutral"])

    def test_neutrality_achieved(self):
        from te_algorithms.gdal.land_deg.ldn_planning import (
            project_scenario_against_bau,
        )

        scenario = {"total_gains_km2": 30.0, "total_avoided_losses_km2": 30.0}
        p = project_scenario_against_bau(self._bau(), scenario, 2030)
        # contribution 60 >= shortfall 50 -> adjusted floored at target
        self.assertTrue(p["neutral"])
        self.assertAlmostEqual(p["gap_closed_pct"], 100.0)


class TestZoneCombination(unittest.TestCase):
    """Tests for the multi-raster zone combination logic."""

    def test_combine_arrays_unique_combinations(self):
        from te_algorithms.gdal.land_deg import zones

        a = np.array([[1, 1], [2, 2]], dtype=np.int32)
        b = np.array([[1, 2], [1, 2]], dtype=np.int32)
        zone_ids, labels = zones.combine_arrays_to_zones([a, b], [-32768, -32768])
        # 4 unique combinations -> 4 sequential zones
        self.assertEqual(len(labels), 4)
        self.assertEqual(sorted(set(zone_ids.ravel())), [1, 2, 3, 4])
        # labels decode to the source combination "a_b"
        self.assertIn("1_1", labels.values())
        self.assertIn("2_2", labels.values())

    def test_combine_arrays_nodata_becomes_zero(self):
        from te_algorithms.gdal.land_deg import zones

        a = np.array([[1, -32768]], dtype=np.int32)
        b = np.array([[5, 5]], dtype=np.int32)
        zone_ids, labels = zones.combine_arrays_to_zones([a, b], [-32768, -32768])
        # The nodata pixel -> zone id 0 (outside)
        self.assertEqual(zone_ids[0, 1], 0)
        self.assertEqual(zone_ids[0, 0], 1)
        self.assertEqual(len(labels), 1)


if __name__ == "__main__":
    unittest.main()
