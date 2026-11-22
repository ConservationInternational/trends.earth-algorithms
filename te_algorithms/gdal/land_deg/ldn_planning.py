"""LDN Planning Tool — core algorithm functions.

Implements the four MVP analysis steps:
  1. ``compute_arr_classification`` — Avoid / Reduce / Reverse (ARR) classification
  2. ``compute_hotspots``           — degradation hotspot prioritization
  3. ``compute_bau_rate``           — Business-As-Usual loss rate and projection
  4. ``apply_scenario``            — rule-based scenario / target-setting balance sheet

Conceptual basis (all rules referenced in the function docstrings come from):
  Cowie et al. (2018) "Land in balance: The scientific conceptual framework for
  Land Degradation Neutrality." Env. Sci. & Policy 79: 25–35.
  UNCCD (2025) Good Practice Guidance Addendum for SDG Indicator 15.3.1
  (Advanced Unedited Version).

Key methodological rules encoded here:
  - Overall degradation status uses one-out, all-out (1OAO) — a pixel is
    degraded if *any* of the three sub-indicators is degraded. This is already
    encoded in the SDG 15.3.1 status layer consumed as input.
  - LDN baseline (2015) is the frame of reference; target = baseline (no net loss).
  - Response-hierarchy priority: Avoid > Reduce > Reverse.
  - Only *Reverse* actions generate counterbalancing GAINS in natural capital;
    Avoid and Reduce prevent/reduce LOSSES but are not counterbalancing gains.
  - ARR *Reduce* class = at-risk (productivity trajectory declining or
    user-supplied risk layer), NOT "degraded in one sub-indicator" (that state
    cannot exist under 1OAO).
"""

import json
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

from ..progress import ProgressReporter
from ..util_numba import calc_cell_area
from . import config, zones

logger = logging.getLogger(__name__)

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# ARR class pixel values written to the output raster
# ---------------------------------------------------------------------------
ARR_AVOID: int = 1  # Healthy land to protect
ARR_REDUCE: int = 2  # At-risk land to manage with SLM
ARR_REVERSE: int = 3  # Already-degraded land to restore/rehabilitate
ARR_NODATA: int = int(config.NODATA_VALUE)  # -32768
ARR_REASON_UNKNOWN_STATUS: int = 0
ARR_REASON_DEGRADED: int = 1
ARR_REASON_LPD_STRESSED: int = 2
ARR_REASON_RISK_POSITIVE: int = 4
ARR_REASON_NO_RISK_SOURCE: int = 8
ARR_REASON_MISSING_REQUIRED_INPUT: int = 16
ARR_REASON_NO_RISK_SIGNAL: int = 32
ARR_REASON_OUTSIDE_AOI: int = 255

# Status values that indicate "overall degraded" in each status type
# 3-class: -1 = degraded (0 = stable, 1 = improved)
# 7-class (expanded status matrix):
#   1 = persistent degradation (degraded at baseline AND in reporting period)
#   2 = recent degradation     (stable/improved at baseline, degraded in reporting)
#   3 = baseline degradation   (degraded at baseline, stable in reporting — still degraded)
#   4 = stability, 5-7 = improvement classes
_STATUS_3CLASS_DEGRADED: set = {-1}
_STATUS_7CLASS_DEGRADED: set = {1, 2, 3}

# Land Productivity Dynamics (5-class LPD / PRODUCTIVITY_CLASS_KEY) value that
# signals at-risk-but-not-yet-degraded land.
#
# The 5-class LPD is recoded to 3-class degradation (prod5_to_prod3) as:
#   1 Declining           -> -1 degraded
#   2 Moderate decline    -> -1 degraded
#   3 Stable but stressed ->  0 not degraded   <-- early-warning "at-risk" class
#   4 Stable              ->  0 not degraded
#   5 Increasing          -> +1 improved
#
# Classes 1 and 2 are therefore ALREADY degraded (and, under one-out-all-out,
# already classified Reverse), so they cannot supply the *Reduce* (at-risk but
# not-yet-degraded) signal.  Only class 3 ("stable but stressed") is a
# non-degraded early-warning class, making it the correct at-risk proxy for the
# MVP.  A richer Reduce definition is a post-MVP refinement.
_LPD_STRESSED: set = {3}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_pixel_areas_km2(
    gt: Tuple,
    y_off: int,
    y_size: int,
    x_size: int,
    projection: Optional[str] = None,
) -> np.ndarray:
    """Compute per-pixel physical area in km² for a raster block."""
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError("Rotated raster grids are not supported for area calculation.")

    srs = osr.SpatialReference()
    if projection:
        if srs.ImportFromWkt(projection) != 0:
            raise ValueError("Raster has an invalid coordinate reference system.")
    else:
        srs.ImportFromEPSG(4326)
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    pixel_w_deg = abs(gt[1])
    areas = np.empty((y_size, x_size), dtype=np.float64)
    if srs.IsSame(wgs84):
        for row in range(y_size):
            lat_max = gt[3] + (y_off + row) * gt[5]
            lat_min = lat_max + gt[5]
            if lat_min > lat_max:
                lat_min, lat_max = lat_max, lat_min
            areas[row, :] = calc_cell_area(lat_min, lat_max, pixel_w_deg) / 1e6
        return areas

    equal_area = osr.SpatialReference()
    if (
        equal_area.ImportFromProj4(
            "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
        )
        != 0
    ):
        raise RuntimeError("Could not initialize the global equal-area projection.")
    equal_area.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(srs, equal_area)
    world_width = 4 * math.sqrt(2) * 6371007.1809

    for row in range(y_size):
        y_top = gt[3] + (y_off + row) * gt[5]
        y_bottom = y_top + gt[5]
        corners = []
        for col in range(x_size):
            x_left = gt[0] + col * gt[1]
            x_right = x_left + gt[1]
            corners.extend(
                [
                    (x_left, y_top, 0),
                    (x_right, y_top, 0),
                    (x_right, y_bottom, 0),
                    (x_left, y_bottom, 0),
                ]
            )
        transformed = transform.TransformPoints(corners)
        xy = np.asarray(transformed, dtype=np.float64)[:, :2].reshape(x_size, 4, 2)
        for col in range(x_size):
            polygon = xy[col].copy()
            # Keep cells crossing the antimeridian on the same Mollweide branch.
            for i in range(1, 4):
                while polygon[i, 0] - polygon[0, 0] > world_width / 2:
                    polygon[i, 0] -= world_width
                while polygon[i, 0] - polygon[0, 0] < -world_width / 2:
                    polygon[i, 0] += world_width
            areas[row, col] = (
                abs(
                    np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                    - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
                )
                / 2e6
            )
    return areas


def _validate_status_values(values: np.ndarray, status_type: str) -> None:
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("Status raster values must be finite integers.")
    allowed = {-1, 0, 1} if status_type == "3class" else set(range(1, 8))
    invalid = set(int(value) for value in np.unique(values)) - allowed
    if invalid:
        raise ValueError(
            f"Values {sorted(invalid)} are invalid for {status_type} status."
        )


def _prepare_aoi_geometries(
    aoi_path: Optional[str],
    target_projection: str,
    aoi_srs_wkt: Optional[str] = None,
    aoi=None,
) -> Optional[List[ogr.Geometry]]:
    if aoi_path is not None and aoi is not None:
        raise ValueError("Provide either aoi_path or aoi, not both.")
    if aoi_path is None and aoi is None:
        return None
    datasource = None
    if aoi_path is not None:
        datasource = ogr.Open(aoi_path, 0)
        if datasource is None:
            raise RuntimeError(f"Cannot open AOI: {aoi_path}")
        layer = datasource.GetLayer(0)
        source_srs = layer.GetSpatialRef()
        source_geometries = [
            feature.GetGeometryRef().Clone()
            for feature in layer
            if feature.GetGeometryRef() is not None
            and not feature.GetGeometryRef().IsEmpty()
        ]
    else:
        geojson = aoi.get_geojson() if hasattr(aoi, "get_geojson") else aoi
        if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
            raise ValueError("aoi must expose a GeoJSON FeatureCollection.")
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)
        source_geometries = []
        for feature in geojson.get("features", []):
            geometry_json = feature.get("geometry")
            if not geometry_json:
                continue
            geometry = ogr.CreateGeometryFromJson(json.dumps(geometry_json))
            if geometry is None or geometry.IsEmpty():
                continue
            source_geometries.append(geometry)
    if source_srs is None and aoi_srs_wkt:
        source_srs = osr.SpatialReference()
        if source_srs.SetFromUserInput(aoi_srs_wkt) != 0:
            raise ValueError("Invalid explicit AOI coordinate reference system.")
    if source_srs is None:
        raise ValueError("AOI has no CRS; provide aoi_srs explicitly.")
    target_srs = osr.SpatialReference()
    if not target_projection or target_srs.ImportFromWkt(target_projection) != 0:
        raise ValueError("Reference raster has no valid CRS.")
    source_srs = source_srs.Clone()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = (
        None
        if source_srs.IsSame(target_srs)
        else osr.CoordinateTransformation(source_srs, target_srs)
    )
    geometries = []
    geometries = []
    for geometry in source_geometries:
        if transform is not None:
            if geometry.Transform(transform) != 0:
                raise ValueError(
                    "Could not transform an AOI geometry to the raster CRS."
                )
        geometries.append(geometry)
    datasource = None
    if not geometries:
        raise ValueError("AOI contains no valid geometries.")
    return geometries


def _rasterize_aoi_block(
    geometries: Optional[List[ogr.Geometry]],
    gt: Tuple,
    projection: str,
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
) -> Optional[np.ndarray]:
    if geometries is None:
        return None
    block_gt = (
        gt[0] + x_off * gt[1],
        gt[1],
        gt[2],
        gt[3] + y_off * gt[5],
        gt[4],
        gt[5],
    )
    mask_ds = gdal.GetDriverByName("MEM").Create("", x_size, y_size, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(block_gt)
    mask_ds.SetProjection(projection)
    mask_ds.GetRasterBand(1).Fill(0)
    layer_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = layer_ds.CreateLayer("aoi", srs=srs, geom_type=ogr.wkbUnknown)
    layer_defn = layer.GetLayerDefn()
    for geometry in geometries:
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
    if gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1]) != 0:
        raise RuntimeError("Failed to rasterize AOI.")
    result = mask_ds.GetRasterBand(1).ReadAsArray() != 0
    mask_ds = None
    layer_ds = None
    return result


def _open_band(path: str, band_index: int = 1) -> Tuple[gdal.Dataset, gdal.Band]:
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(band_index)
    if band is None:
        raise RuntimeError(f"Band {band_index} not found in {path}")
    return ds, band


def _band_has_valid_values(band: gdal.Band) -> bool:
    nodata = band.GetNoDataValue()
    for y_off in range(0, band.YSize, BLOCK_SIZE):
        y_size = min(BLOCK_SIZE, band.YSize - y_off)
        values = band.ReadAsArray(0, y_off, band.XSize, y_size)
        if values is None:
            continue
        valid = np.isfinite(values) & (values != int(config.NODATA_VALUE))
        if nodata is not None:
            valid &= values != nodata
        if np.any(valid):
            return True
    return False


def _create_output_raster(
    output_path: str,
    ref_ds: gdal.Dataset,
    nodata: int = ARR_NODATA,
    dtype: int = gdal.GDT_Int16,
    bands: int = 1,
) -> gdal.Dataset:
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        bands,
        dtype,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    b = out_ds.GetRasterBand(1)
    b.SetNoDataValue(nodata)
    b.Fill(nodata)
    if bands > 1:
        b.SetDescription("ARR class")
        for index in range(2, bands + 1):
            reason_band = out_ds.GetRasterBand(index)
            reason_band.SetNoDataValue(nodata)
            reason_band.Fill(ARR_REASON_OUTSIDE_AOI)
            reason_band.SetDescription("ARR classification reason code")
    return out_ds


def _aligned_band(
    path: str,
    band_index: int,
    reference_ds: gdal.Dataset,
    resample_alg: int = gdal.GRA_NearestNeighbour,
) -> Tuple[gdal.Dataset, gdal.Band]:
    """Open or warp a band onto the exact reference raster grid."""
    source_ds, source_band = _open_band(path, band_index)
    if (
        source_ds.RasterXSize == reference_ds.RasterXSize
        and source_ds.RasterYSize == reference_ds.RasterYSize
        and source_ds.GetGeoTransform() == reference_ds.GetGeoTransform()
        and source_ds.GetProjection() == reference_ds.GetProjection()
    ):
        return source_ds, source_band
    if not source_ds.GetProjection() or not reference_ds.GetProjection():
        raise ValueError(
            f"Cannot align {path}: both source and reference rasters need a CRS."
        )
    gt = reference_ds.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError("Rotated reference grids are not supported.")
    bounds = (
        gt[0],
        gt[3] + gt[5] * reference_ds.RasterYSize,
        gt[0] + gt[1] * reference_ds.RasterXSize,
        gt[3],
    )
    selected_band = gdal.Translate("", path, format="VRT", bandList=[band_index])
    if selected_band is None:
        raise RuntimeError(f"Could not select band {band_index} from {path}.")
    warped = gdal.Warp(
        "",
        selected_band,
        format="VRT",
        width=reference_ds.RasterXSize,
        height=reference_ds.RasterYSize,
        outputBounds=bounds,
        outputBoundsSRS=reference_ds.GetProjection(),
        dstSRS=reference_ds.GetProjection(),
        resampleAlg=resample_alg,
    )
    del selected_band
    if warped is None:
        raise RuntimeError(f"Could not align raster {path}.")
    warped_band = warped.GetRasterBand(1)
    if warped_band is None:
        raise RuntimeError(f"Could not read aligned raster {path}.")
    no_aligned_values = not _band_has_valid_values(warped_band)
    has_source_values = (
        _band_has_valid_values(source_band) if no_aligned_values else True
    )
    del source_ds, source_band
    if no_aligned_values and has_source_values:
        raise ValueError(f"{path} does not overlap the reference raster.")
    return warped, warped_band


def _detect_status_type(
    ds: gdal.Dataset, band_index: int = 1, status_type: str = "auto"
) -> str:
    """Validate or infer status encoding; reject auto-detection ambiguity.

    The 3-class SDG 15.3.1 indicator encodes values in {-1, 0, 1} (degraded /
    stable / improved). The 7-class GPG "expanded status" band is all-positive,
    with values in {1..7} (1-3 degradation, 4 stability, 5-7 improvement).

    ``band_index`` MUST match the band actually used for the analysis — the SDG
    output is multi-band and the status band is generally not band 1.

    Inference masks nodata (both the band's declared nodata and the -32768
    sentinel). An all-1 band is ambiguous and requires an explicit declaration.
    """
    status_type = str(status_type).lower()
    if status_type not in {"auto", "3class", "7class"}:
        raise ValueError("status_type must be '3class', '7class', or 'auto'.")
    band = ds.GetRasterBand(band_index)
    if band is None:
        raise RuntimeError(f"Band {band_index} not found.")
    arr = band.ReadAsArray()
    if arr is None:
        raise ValueError("Could not read the status raster band.")
    arr = arr.ravel()
    nd = band.GetNoDataValue()
    mask = np.isfinite(arr) & (arr != int(config.NODATA_VALUE))
    if nd is not None:
        mask &= arr != nd
    valid = arr[mask]
    if valid.size == 0:
        if status_type == "auto":
            return "none"
        return status_type

    uniques = set(int(v) for v in np.unique(valid))
    if status_type != "auto":
        _validate_status_values(valid, status_type)
        return status_type
    is_3class = uniques <= {-1, 0, 1}
    is_7class = uniques <= set(range(1, 8))
    if is_3class and is_7class:
        raise ValueError(
            "Status values {1} are ambiguous between 3-class and 7-class "
            "encodings; declare status_type explicitly."
        )
    if not is_3class and not is_7class:
        raise ValueError(f"Status values {sorted(uniques)} are invalid.")
    return "3class" if is_3class else "7class"


BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# 1. ARR classification
# ---------------------------------------------------------------------------


def compute_arr_classification(
    status_path: str,
    status_band_index: int = 1,
    output_path: Optional[str] = None,
    trajectory_path: Optional[str] = None,
    trajectory_band_index: int = 1,
    risk_layer_path: Optional[str] = None,
    risk_band_index: int = 1,
    progress_callback=None,
    killed_callback=None,
    status_type: str = "auto",
    aoi_path: Optional[str] = None,
    aoi_srs: Optional[str] = None,
    aoi=None,
    risk_threshold: float = 0.0,
) -> Tuple[Dict[str, Any], str]:
    """Classify land into the LDN response hierarchy (Avoid / Reduce / Reverse).

    Rules (Cowie et al. 2018, Principle 12; GPG Addendum §2.2):
      - **Reverse** (3): pixel is overall degraded (any degradation category
        under the one-out-all-out principle already encoded in the status layer).
        These are the only pixels where restoration actions can generate
        counterbalancing GAINS.
      - **Reduce** (2): not overall degraded, but at risk — Land Productivity
        Dynamics class 3 "stable but stressed" (the only non-degraded LPD class
        that carries an early-warning signal; classes 1 and 2 are already
        degraded and hence Reverse) or risk_layer value above ``risk_threshold``.
        SLM here avoids losses.
      - **Avoid** (1): not degraded and not flagged at risk. Protection
        prevents losses but does not generate gains.
      - If no LPD or risk layer is supplied, all non-degraded land defaults to
        **Avoid**; users may designate Reduce zones via uploaded polygons in
        the scenario step.

    The ``trajectory_path`` argument expects a 5-class Land Productivity
    Dynamics (LPD) layer whose values follow PRODUCTIVITY_CLASS_KEY.

    Output band 1 values: 1=Avoid, 2=Reduce, 3=Reverse, -32768=NoData.
    Output band 2 contains reason codes documented in ``summary["reason_codes"]``.
    """
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(
            suffix="_ldn_arr.tif", delete=False
        ).name
    try:
        risk_threshold = float(risk_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("risk_threshold must be a finite number.") from error
    if not math.isfinite(risk_threshold):
        raise ValueError("risk_threshold must be a finite number.")

    status_ds, status_band = _open_band(status_path, status_band_index)
    xsize = status_ds.RasterXSize
    ysize = status_ds.RasterYSize
    gt = status_ds.GetGeoTransform()

    nodata_val = status_band.GetNoDataValue()
    if nodata_val is None:
        nodata_val = float(config.NODATA_VALUE)

    status_type = _detect_status_type(status_ds, status_band_index, status_type)
    degraded_set = (
        _STATUS_7CLASS_DEGRADED if status_type == "7class" else _STATUS_3CLASS_DEGRADED
    )
    logger.info(
        "ARR: status layer '%s' band %d — detected %s status, "
        "degraded classes = %s, nodata = %s",
        status_path,
        status_band_index,
        status_type,
        sorted(degraded_set),
        nodata_val,
    )

    traj_ds = traj_band = None
    if trajectory_path:
        traj_ds, traj_band = _aligned_band(
            trajectory_path, trajectory_band_index, status_ds
        )

    risk_ds = risk_band_obj = None
    if risk_layer_path:
        risk_ds, risk_band_obj = _aligned_band(
            risk_layer_path, risk_band_index, status_ds
        )
    aoi_geometries = _prepare_aoi_geometries(
        aoi_path, status_ds.GetProjection(), aoi_srs, aoi
    )

    out_ds = _create_output_raster(output_path, status_ds, bands=2)
    out_band = out_ds.GetRasterBand(1)
    reason_band = out_ds.GetRasterBand(2)

    avoid_km2 = reduce_km2 = reverse_km2 = nodata_km2 = 0.0
    reason_counts = {}
    reason_area_km2 = {}
    n_blocks_y = math.ceil(ysize / BLOCK_SIZE)
    n_blocks_x = math.ceil(xsize / BLOCK_SIZE)
    total_blocks = n_blocks_y * n_blocks_x
    block_count = 0
    cancelled = False
    block_reporter = ProgressReporter(progress_callback, start=10, end=100)

    for y_off in range(0, ysize, BLOCK_SIZE):
        if killed_callback and killed_callback():
            cancelled = True
            break
        y_blk = min(BLOCK_SIZE, ysize - y_off)

        for x_off in range(0, xsize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                cancelled = True
                break
            x_blk = min(BLOCK_SIZE, xsize - x_off)

            status_values = status_band.ReadAsArray(x_off, y_off, x_blk, y_blk)
            status_arr = np.where(
                np.isfinite(status_values), status_values, ARR_NODATA
            ).astype(np.int16)

            pixel_areas = _compute_pixel_areas_km2(
                gt, y_off, y_blk, x_blk, status_ds.GetProjection()
            )

            status_valid = (
                np.isfinite(status_values)
                & (status_values != nodata_val)
                & (status_values != int(config.NODATA_VALUE))
            )
            _validate_status_values(status_values[status_valid], status_type)
            aoi_mask = _rasterize_aoi_block(
                aoi_geometries,
                gt,
                status_ds.GetProjection(),
                x_off,
                y_off,
                x_blk,
                y_blk,
            )
            if aoi_mask is None:
                aoi_mask = np.ones(status_arr.shape, dtype=bool)
            valid = status_valid & aoi_mask

            # --- Reverse: overall degraded ---
            is_degraded = np.zeros(status_arr.shape, dtype=bool)
            for dv in degraded_set:
                is_degraded |= status_arr == dv

            # --- Reduce / Avoid: at-risk signal from trajectory or risk layer ---
            not_degraded = valid & ~is_degraded
            is_at_risk = np.zeros(status_arr.shape, dtype=bool)
            missing_required_input = np.zeros(status_arr.shape, dtype=bool)

            if traj_band is not None:
                traj_arr = traj_band.ReadAsArray(x_off, y_off, x_blk, y_blk).astype(
                    np.int16
                )
                traj_nd = traj_band.GetNoDataValue()
                if traj_nd is None:
                    traj_nd = float(config.NODATA_VALUE)
                traj_valid = np.isfinite(traj_arr) & (traj_arr != int(traj_nd))
                for rv in _LPD_STRESSED:
                    is_at_risk |= (traj_arr == rv) & traj_valid
                missing_required_input |= ~traj_valid

            if risk_band_obj is not None:
                risk_arr = risk_band_obj.ReadAsArray(x_off, y_off, x_blk, y_blk)
                risk_nd = risk_band_obj.GetNoDataValue()
                if risk_nd is None:
                    risk_nd = config.NODATA_VALUE
                risk_valid = np.isfinite(risk_arr) & (risk_arr != risk_nd)
                risk_at_risk = (risk_arr > risk_threshold) & risk_valid
                is_at_risk |= risk_at_risk
                missing_required_input |= ~risk_valid

            # Build output block
            arr = np.full(status_arr.shape, ARR_NODATA, dtype=np.int16)
            reason_arr = np.full(
                status_arr.shape, ARR_REASON_OUTSIDE_AOI, dtype=np.int16
            )
            reason_arr[aoi_mask] = ARR_REASON_UNKNOWN_STATUS
            arr[valid & is_degraded & aoi_mask] = ARR_REVERSE
            reason_arr[valid & is_degraded] = ARR_REASON_DEGRADED
            known_not_degraded = not_degraded & ~missing_required_input
            arr[known_not_degraded & is_at_risk] = ARR_REDUCE
            arr[known_not_degraded & ~is_at_risk] = ARR_AVOID
            reason_arr[not_degraded & missing_required_input] = (
                ARR_REASON_MISSING_REQUIRED_INPUT
            )
            if traj_band is not None:
                reason_arr[known_not_degraded & traj_valid & (traj_arr == 3)] |= (
                    ARR_REASON_LPD_STRESSED
                )
            if risk_band_obj is not None:
                reason_arr[
                    known_not_degraded & risk_valid & (risk_arr > risk_threshold)
                ] |= ARR_REASON_RISK_POSITIVE
            no_risk_sources = traj_band is None and risk_band_obj is None
            if no_risk_sources:
                reason_arr[known_not_degraded & ~is_at_risk] = ARR_REASON_NO_RISK_SOURCE
            else:
                reason_arr[known_not_degraded & ~is_at_risk] = ARR_REASON_NO_RISK_SIGNAL

            # Accumulate areas (km²)
            avoid_km2 += float(np.sum(pixel_areas[arr == ARR_AVOID]))
            reduce_km2 += float(np.sum(pixel_areas[arr == ARR_REDUCE]))
            reverse_km2 += float(np.sum(pixel_areas[arr == ARR_REVERSE]))
            nodata_km2 += float(np.sum(pixel_areas[aoi_mask & (arr == ARR_NODATA)]))

            out_band.WriteArray(arr, x_off, y_off)
            reason_band.WriteArray(reason_arr, x_off, y_off)
            for code in np.unique(reason_arr):
                if code == ARR_REASON_OUTSIDE_AOI:
                    continue
                code = int(code)
                mask = reason_arr == code
                reason_counts[code] = reason_counts.get(code, 0) + int(
                    np.count_nonzero(mask)
                )
                reason_area_km2[code] = reason_area_km2.get(code, 0.0) + float(
                    np.sum(pixel_areas[mask])
                )

            block_count += 1
            block_reporter.update(block_count / total_blocks)
        if cancelled:
            break

    out_band.FlushCache()
    reason_band.FlushCache()
    out_ds.FlushCache()
    del out_band, reason_band, out_ds, status_ds, status_band
    if traj_ds:
        del traj_ds, traj_band
    if risk_ds:
        del risk_ds, risk_band_obj

    if cancelled:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise RuntimeError("ARR classification cancelled; partial output was removed.")
    block_reporter.finish()

    summary = {
        "avoid_km2": avoid_km2,
        "reduce_km2": reduce_km2,
        "reverse_km2": reverse_km2,
        "nodata_km2": nodata_km2,
        "total_km2": avoid_km2 + reduce_km2 + reverse_km2,
        "status_type": status_type,
        "risk_threshold": risk_threshold,
        "trajectory_used": trajectory_path is not None,
        "risk_layer_used": risk_layer_path is not None,
        "aoi_used": aoi_path is not None or aoi is not None,
        "reason_band_index": 2,
        "reason_codes": {
            ARR_REASON_UNKNOWN_STATUS: "Input status is NoData",
            ARR_REASON_DEGRADED: "Degraded status assigned Reverse",
            ARR_REASON_LPD_STRESSED: "LPD class 3 stable but stressed",
            ARR_REASON_RISK_POSITIVE: "Risk value exceeds configured threshold",
            ARR_REASON_NO_RISK_SOURCE: "No risk source supplied; default Avoid",
            ARR_REASON_MISSING_REQUIRED_INPUT: (
                "Required risk/trajectory input missing; class unknown"
            ),
            ARR_REASON_NO_RISK_SIGNAL: "No configured at-risk signal; Avoid",
            ARR_REASON_OUTSIDE_AOI: "Outside the supplied AOI",
        },
        "reason_counts": reason_counts,
        "reason_area_km2": reason_area_km2,
    }
    logger.info(
        "ARR: Avoid=%.1f km², Reduce=%.1f km², Reverse=%.1f km²",
        avoid_km2,
        reduce_km2,
        reverse_km2,
    )
    return summary, output_path


def summarize_land_types(
    land_types_path: str,
    labels: Dict[int, str],
    layer_names: Optional[List[str]] = None,
    block_rows: int = 256,
) -> Dict[str, Any]:
    """Summarize the physical area of each land type in a land-types raster.

    Reads the raster in row blocks. Land type ``0`` and NoData are excluded.
    Types present in the raster but missing from ``labels`` are reported with
    their numeric id as the label.
    """
    ds, band = _open_band(land_types_path)
    gt = ds.GetGeoTransform()
    projection = ds.GetProjection()
    nodata = band.GetNoDataValue()
    pixels: Dict[int, int] = {}
    areas: Dict[int, float] = {}
    for y_off in range(0, ds.RasterYSize, block_rows):
        rows = min(block_rows, ds.RasterYSize - y_off)
        values = band.ReadAsArray(0, y_off, ds.RasterXSize, rows)
        valid = values != 0
        if nodata is not None:
            valid &= values != nodata
        if not np.any(valid):
            continue
        pixel_areas = _compute_pixel_areas_km2(
            gt, y_off, rows, ds.RasterXSize, projection
        )
        ids, inverse = np.unique(values[valid], return_inverse=True)
        block_areas = np.bincount(inverse, weights=pixel_areas[valid])
        block_pixels = np.bincount(inverse)
        for land_type, area, count in zip(ids, block_areas, block_pixels):
            land_type = int(land_type)
            areas[land_type] = areas.get(land_type, 0.0) + float(area)
            pixels[land_type] = pixels.get(land_type, 0) + int(count)
    del band, ds

    total = sum(areas.values())
    land_types = []
    for land_type in sorted(set(labels) | set(areas)):
        label = labels.get(land_type, str(land_type))
        area = areas.get(land_type, 0.0)
        land_types.append(
            {
                "land_type_id": land_type,
                "label": label,
                "values": label.split("_"),
                "pixels": pixels.get(land_type, 0),
                "area_km2": area,
                "fraction": area / total if total else 0.0,
            }
        )
    return {
        "layer_names": list(layer_names or []),
        "n_land_types": len(land_types),
        "total_area_km2": total,
        "land_types": land_types,
    }


# ---------------------------------------------------------------------------
# 2. Hotspot prioritization
# ---------------------------------------------------------------------------


def _generate_fishnet(
    gt: Tuple,
    xsize: int,
    ysize: int,
    grid_size_km: float,
    srs: osr.SpatialReference,
    output_path: str,
) -> None:
    """Create a fishnet polygon vector file from the raster extent."""
    # Convert grid_size_km to approximate degrees (rough: 1 deg ≈ 111 km)
    cell_deg = grid_size_km / 111.0
    min_x = gt[0]
    max_y = gt[3]
    max_x = gt[0] + gt[1] * xsize
    min_y = gt[3] + gt[5] * ysize

    drv = ogr.GetDriverByName("GPKG")
    # GPKG driver requires the file not to exist yet; delete before creating.
    import os as _os

    if _os.path.exists(output_path):
        _os.remove(output_path)
    ds = drv.CreateDataSource(output_path)
    lyr = ds.CreateLayer("zones", srs=srs, geom_type=ogr.wkbPolygon)
    fld = ogr.FieldDefn("zone_id", ogr.OFTInteger)
    lyr.CreateField(fld)

    fid = 0
    y = max_y
    while y > min_y:
        x = min_x
        while x < max_x:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x, y)
            ring.AddPoint(x + cell_deg, y)
            ring.AddPoint(x + cell_deg, y - cell_deg)
            ring.AddPoint(x, y - cell_deg)
            ring.AddPoint(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feat = ogr.Feature(lyr.GetLayerDefn())
            feat.SetGeometry(poly)
            feat.SetField("zone_id", fid)
            lyr.CreateFeature(feat)
            fid += 1
            x += cell_deg
        y -= cell_deg

    ds.FlushCache()
    del ds


def compute_hotspots(
    status_path: str,
    status_band_index: int = 1,
    zones_path: Optional[str] = None,
    grid_size_km: float = 50.0,
    output_vector_path: Optional[str] = None,
    output_raster_path: Optional[str] = None,
    progress_callback=None,
    killed_callback=None,
    status_type: str = "auto",
) -> Tuple[Dict[str, Any], str, str]:
    """Rank zones by degraded fraction (degradation hotspot prioritization).

    For each zone, computes the proportion of valid pixels that are degraded
    (status == -1 in 3-class, or values 1/2/3 in 7-class). Zones are ranked
    1 = highest degraded fraction (worst hotspot).

    Args:
        status_path: Path to SDG 15.3.1 status raster.
        status_band_index: Band index.
        zones_path: Optional vector layer (GPKG/SHP) whose features define zones.
            If None, a fishnet grid of ``grid_size_km`` cells is auto-generated.
        grid_size_km: Fishnet cell size in km (used when zones_path is None).
        output_vector_path: Output path for ranked polygon layer (.gpkg).
        output_raster_path: Output path for degraded-fraction raster (.tif).
        progress_callback: Optional callable(pct: float).
        killed_callback: Optional callable() → bool.
        status_type: ``3class``, ``7class``, or ``auto`` to infer from values.

    Returns:
        (summary_dict, vector_path, raster_path)
        summary_dict keys: n_zones, top10_zone_ids, mean_degraded_fraction
    """
    if output_vector_path is None:
        output_vector_path = tempfile.NamedTemporaryFile(
            suffix="_hotspots.gpkg", delete=False
        ).name
    if output_raster_path is None:
        output_raster_path = tempfile.NamedTemporaryFile(
            suffix="_hotspots_raster.tif", delete=False
        ).name

    status_ds, status_band = _open_band(status_path, status_band_index)
    gt = status_ds.GetGeoTransform()
    xsize = status_ds.RasterXSize
    ysize = status_ds.RasterYSize
    proj = status_ds.GetProjection()

    nodata_val = status_band.GetNoDataValue()
    if nodata_val is None:
        nodata_val = float(config.NODATA_VALUE)

    status_type = _detect_status_type(status_ds, status_band_index, status_type)
    degraded_set = (
        _STATUS_7CLASS_DEGRADED if status_type == "7class" else _STATUS_3CLASS_DEGRADED
    )

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)

    # --- Prepare zones ---
    temp_fishnet = None
    if zones_path is None:
        # mkstemp gives us a unique name; close+delete immediately so the
        # GDAL GPKG driver can create the file itself.
        import os as _os

        fd, temp_fishnet = tempfile.mkstemp(suffix="_fishnet.gpkg")
        _os.close(fd)
        _os.remove(temp_fishnet)
        _generate_fishnet(gt, xsize, ysize, grid_size_km, srs, temp_fishnet)
        zones_path = temp_fishnet

    zones_ds = ogr.Open(zones_path, 0)
    if zones_ds is None:
        raise RuntimeError(f"Cannot open zones layer: {zones_path}")
    zones_lyr = zones_ds.GetLayer(0)
    n_zones = zones_lyr.GetFeatureCount()

    # Build output vector layer (copy schema + add result fields)
    drv = ogr.GetDriverByName("GPKG")
    out_vec_ds = drv.CreateDataSource(output_vector_path)
    out_lyr = out_vec_ds.CreateLayer(
        "hotspots", srs=zones_lyr.GetSpatialRef() or srs, geom_type=ogr.wkbPolygon
    )

    # Copy source fields
    zones_lyr_defn = zones_lyr.GetLayerDefn()
    for i in range(zones_lyr_defn.GetFieldCount()):
        out_lyr.CreateField(zones_lyr_defn.GetFieldDefn(i))

    # Add result fields
    for fname, ftype in [
        ("total_pixels", ogr.OFTInteger),
        ("deg_pixels", ogr.OFTInteger),
        ("deg_fraction", ogr.OFTReal),
        ("deg_pct", ogr.OFTInteger),
        ("deg_area_km2", ogr.OFTReal),
        ("priority_rank", ogr.OFTInteger),
    ]:
        out_lyr.CreateField(ogr.FieldDefn(fname, ftype))

    out_lyr_defn = out_lyr.GetLayerDefn()

    # Per-zone stats (rasterize each zone and count)
    zone_stats = []  # list of (fid, deg_fraction, deg_area_km2)

    status_array_full = status_band.ReadAsArray().astype(np.int16)
    is_degraded_full = np.zeros(status_array_full.shape, dtype=bool)
    for dv in degraded_set:
        is_degraded_full |= status_array_full == dv
    is_valid_full = status_array_full != int(nodata_val)

    # Pre-compute pixel areas for the full raster
    pixel_areas_full = _compute_pixel_areas_km2(
        gt, 0, ysize, xsize, status_ds.GetProjection()
    )

    # Rasterize each zone to compute stats
    mem_drv = gdal.GetDriverByName("MEM")
    total_deg_fractions = []

    zones_lyr.ResetReading()
    for idx, feat in enumerate(zones_lyr):
        if killed_callback and killed_callback():
            break

        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        # Bounding box of this feature → pixel window
        env = geom.GetEnvelope()  # (min_x, max_x, min_y, max_y)
        col_min = max(0, int((env[0] - gt[0]) / gt[1]))
        col_max = min(xsize - 1, int((env[1] - gt[0]) / gt[1]))
        row_min = max(0, int((env[3] - gt[3]) / gt[5]))
        row_max = min(ysize - 1, int((env[2] - gt[3]) / gt[5]))

        if col_max < col_min or row_max < row_min:
            continue

        win_w = col_max - col_min + 1
        win_h = row_max - row_min + 1

        # Rasterize the geometry in the window
        mask_ds = mem_drv.Create("", win_w, win_h, 1, gdal.GDT_Byte)
        win_gt = (
            gt[0] + col_min * gt[1],
            gt[1],
            0.0,
            gt[3] + row_min * gt[5],
            0.0,
            gt[5],
        )
        mask_ds.SetGeoTransform(win_gt)
        mask_ds.SetProjection(proj)
        mask_band = mask_ds.GetRasterBand(1)
        mask_band.Fill(0)

        # Create temp single-feature layer
        tmp_ds = ogr.GetDriverByName("Memory").CreateDataSource("tmp")
        tmp_lyr = tmp_ds.CreateLayer("tmp", geom_type=ogr.wkbPolygon)
        tmp_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
        tmp_feat.SetGeometry(geom)
        tmp_lyr.CreateFeature(tmp_feat)

        gdal.RasterizeLayer(mask_ds, [1], tmp_lyr, burn_values=[1])
        mask_arr = mask_band.ReadAsArray()
        del mask_ds, tmp_ds

        # Extract pixel stats within this zone window
        deg_win = is_degraded_full[row_min : row_max + 1, col_min : col_max + 1]
        valid_win = is_valid_full[row_min : row_max + 1, col_min : col_max + 1]
        areas_win = pixel_areas_full[row_min : row_max + 1, col_min : col_max + 1]

        in_zone = mask_arr == 1
        total_px = int(np.sum(in_zone & valid_win))
        deg_px = int(np.sum(in_zone & valid_win & deg_win))
        deg_area = float(np.sum(areas_win[in_zone & valid_win & deg_win]))
        deg_frac = (deg_px / total_px) if total_px > 0 else 0.0

        zone_stats.append(
            {
                "fid": feat.GetFID(),
                "feat": feat.Clone(),
                "total_px": total_px,
                "deg_px": deg_px,
                "deg_frac": deg_frac,
                "deg_area_km2": deg_area,
            }
        )
        total_deg_fractions.append(deg_frac)

        if progress_callback:
            progress_callback(80.0 * (idx + 1) / max(n_zones, 1))

    del status_ds, status_band

    # Rank by descending deg_fraction (rank 1 = worst hotspot)
    zone_stats.sort(key=lambda z: z["deg_frac"], reverse=True)
    for rank, zs in enumerate(zone_stats, start=1):
        zs["priority_rank"] = rank

    # Write output vector features
    for zs in zone_stats:
        out_feat = ogr.Feature(out_lyr_defn)
        src_feat = zs["feat"]
        for i in range(zones_lyr_defn.GetFieldCount()):
            out_feat.SetField(i, src_feat.GetField(i))
        out_feat.SetGeometry(src_feat.GetGeometryRef())
        out_feat.SetField("total_pixels", zs["total_px"])
        out_feat.SetField("deg_pixels", zs["deg_px"])
        out_feat.SetField("deg_fraction", zs["deg_frac"])
        out_feat.SetField("deg_pct", int(round(zs["deg_frac"] * 100)))
        out_feat.SetField("deg_area_km2", zs["deg_area_km2"])
        out_feat.SetField("priority_rank", zs["priority_rank"])
        out_lyr.CreateFeature(out_feat)

    out_vec_ds.FlushCache()
    del out_vec_ds, zones_ds

    # Burn the degraded fraction (as an integer percent, 0-100) into the raster
    # so the map legend is interpretable. The ordinal priority rank is retained
    # in the vector layer's attributes and in the Excel report.
    out_ras_ds = _create_output_raster(
        output_raster_path,
        gdal.Open(status_path),
        nodata=int(config.NODATA_VALUE),
    )

    # Re-open output vector to burn
    out_vec_r = ogr.Open(output_vector_path, 0)
    out_lyr_r = out_vec_r.GetLayer(0)
    gdal.RasterizeLayer(
        out_ras_ds,
        [1],
        out_lyr_r,
        options=["ATTRIBUTE=deg_pct"],
    )
    out_ras_ds.FlushCache()
    del out_ras_ds, out_vec_r

    mean_frac = float(np.mean(total_deg_fractions)) if total_deg_fractions else 0.0
    top10 = [z["fid"] for z in zone_stats[:10]]

    summary = {
        "n_zones": len(zone_stats),
        "mean_degraded_fraction": mean_frac,
        "top10_zone_fids": top10,
        "status_type": status_type,
    }
    logger.info(
        "Hotspots: %d zones, mean degraded fraction=%.1f%%",
        len(zone_stats),
        100 * mean_frac,
    )
    return summary, output_vector_path, output_raster_path


# ---------------------------------------------------------------------------
# 3. BAU (Business-As-Usual) loss rate
# ---------------------------------------------------------------------------


def compute_bau_rate(
    status_baseline_path: str,
    status_baseline_band_index: int = 1,
    status_reporting_path: Optional[str] = None,
    status_reporting_band_index: int = 1,
    year_initial: int = 2000,
    year_final: int = 2015,
    target_year: int = 2030,
    zones_path: Optional[str] = None,
    zones_raster_path: Optional[str] = None,
    zones_raster_labels: Optional[Dict[Any, str]] = None,
    land_type_path: Optional[str] = None,
    land_type_band_index: int = 1,
    land_type_labels: Optional[Dict[Any, str]] = None,
    progress_callback=None,
    killed_callback=None,
    status_baseline_type: str = "auto",
    status_reporting_type: str = "auto",
    aoi_path: Optional[str] = None,
    aoi_srs: Optional[str] = None,
    aoi=None,
) -> Dict[str, Any]:
    """Compute BAU degradation trajectory and project to ``target_year``.

    The LDN frame of reference (Cowie et al. 2018, Module B):
      - Baseline (t₀ = 2015): the reference; LDN target = no net loss vs baseline.
      - Neutrality is the *minimum* objective; net gain is more ambitious.

    Uses total degraded area at the two time points to compute an annual loss
    rate and project linearly to ``target_year``.  The "target line" = baseline
    degraded area (no net loss objective).

    Args:
        status_baseline_path: SDG 15.3.1 status raster at baseline period.
        status_baseline_band_index: Band index in baseline raster.
        status_reporting_path: SDG 15.3.1 status raster at reporting period.
            If None, only the baseline stats are returned (no projection).
        status_reporting_band_index: Band index in reporting raster.
        year_initial: Start year of the baseline period (e.g. 2000).
        year_final: End year of the baseline / start of reporting (e.g. 2015).
        target_year: Year to project BAU to (e.g. 2030).
        zones_path: Optional path to a vector layer whose features define
            sub-national zones for per-zone BAU statistics.  When supplied,
            the summary dict includes a ``"jurisdictions"`` list.
        progress_callback: Optional callable(pct: float).
        killed_callback: Optional callable() → bool.

    Returns:
        summary_dict with keys:
            degraded_area_baseline_km2, degraded_area_reporting_km2,
            total_area_km2, pct_degraded_baseline, pct_degraded_reporting,
            annual_change_km2, bau_projection_{target_year}_km2,
            ldntarget_km2 (= degraded_area_baseline_km2),
            shortfall_km2 (BAU projection − target),
            zones (list of per-zone dicts, if zones_path supplied)
    """

    def _count_degraded(
        path, band_idx, declared_type, reference_ds=None, aoi_geometries=None
    ):
        if reference_ds is None:
            ds, band = _open_band(path, band_idx)
        else:
            ds, band = _aligned_band(path, band_idx, reference_ds)
        gt = ds.GetGeoTransform()
        xsize, ysize = ds.RasterXSize, ds.RasterYSize
        nd = band.GetNoDataValue()
        if nd is None:
            nd = float(config.NODATA_VALUE)
        status_type = _detect_status_type(
            ds, 1 if reference_ds is not None else band_idx, declared_type
        )
        degraded_set = (
            _STATUS_7CLASS_DEGRADED
            if status_type == "7class"
            else _STATUS_3CLASS_DEGRADED
        )
        deg_km2 = 0.0
        total_km2 = 0.0
        for y_off in range(0, ysize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                raise RuntimeError("BAU calculation cancelled.")
            y_blk = min(BLOCK_SIZE, ysize - y_off)
            status_values = band.ReadAsArray(0, y_off, xsize, y_blk)
            areas = _compute_pixel_areas_km2(
                gt, y_off, y_blk, xsize, ds.GetProjection()
            )
            aoi_mask = _rasterize_aoi_block(
                aoi_geometries,
                gt,
                ds.GetProjection(),
                0,
                y_off,
                xsize,
                y_blk,
            )
            valid = (
                np.isfinite(status_values)
                & (status_values != nd)
                & (status_values != int(config.NODATA_VALUE))
            )
            _validate_status_values(status_values[valid], status_type)
            if aoi_mask is not None:
                valid &= aoi_mask
            is_deg = np.zeros(status_values.shape, dtype=bool)
            for dv in degraded_set:
                is_deg |= status_values == dv
            deg_km2 += float(np.sum(areas[valid & is_deg]))
            total_km2 += float(np.sum(areas[valid]))
        if not (deg_km2 or total_km2):
            source_ds, source_band = _open_band(path, band_idx)
            source_stats = source_band.GetStatistics(0, 1)
            del source_ds, source_band
            if source_stats and source_stats[0] is not None and aoi_geometries is None:
                raise ValueError(f"{path} does not overlap the reference grid.")
        del ds, band
        return deg_km2, total_km2

    if progress_callback:
        progress_callback(10.0)

    baseline_ds, baseline_band = _open_band(
        status_baseline_path, status_baseline_band_index
    )
    aoi_geometries = _prepare_aoi_geometries(
        aoi_path, baseline_ds.GetProjection(), aoi_srs, aoi
    )
    bl_type = _detect_status_type(
        baseline_ds, status_baseline_band_index, status_baseline_type
    )
    deg_bl, total = _count_degraded(
        status_baseline_path,
        status_baseline_band_index,
        bl_type,
        aoi_geometries=aoi_geometries,
    )

    if progress_callback:
        progress_callback(50.0)

    deg_rep = None
    annual_change = None
    bau_proj = None
    shortfall = None
    n_years = year_final - year_initial

    if status_reporting_path is not None:
        rp_ds, rp_band = _aligned_band(
            status_reporting_path, status_reporting_band_index, baseline_ds
        )
        rp_type = _detect_status_type(rp_ds, 1, status_reporting_type)
        del rp_ds, rp_band
        deg_rep, _ = _count_degraded(
            status_reporting_path,
            status_reporting_band_index,
            rp_type,
            reference_ds=baseline_ds,
            aoi_geometries=aoi_geometries,
        )
        if n_years > 0:
            annual_change = (deg_rep - deg_bl) / n_years
            n_proj = target_year - year_final
            bau_proj = deg_rep + annual_change * n_proj
            shortfall = max(0.0, bau_proj - deg_bl)  # how much above baseline

    if progress_callback:
        progress_callback(100.0)
    del baseline_ds, baseline_band

    pct_bl = 100.0 * deg_bl / total if total > 0 else 0.0
    pct_rep = 100.0 * deg_rep / total if (deg_rep is not None and total > 0) else None

    summary = {
        "year_initial": year_initial,
        "year_final": year_final,
        "target_year": target_year,
        "total_area_km2": total,
        "degraded_area_baseline_km2": deg_bl,
        "pct_degraded_baseline": pct_bl,
        "degraded_area_reporting_km2": deg_rep,
        "pct_degraded_reporting": pct_rep,
        "annual_change_km2": annual_change,
        f"bau_projection_{target_year}_km2": bau_proj,
        "ldntarget_km2": deg_bl,  # target = no net loss vs baseline
        "shortfall_km2": shortfall,
        "status_baseline_type": bl_type,
        "status_reporting_type": (
            rp_type if status_reporting_path is not None else None
        ),
        "aoi_used": aoi_path is not None or aoi is not None,
    }
    logger.info(
        "BAU: baseline=%.1f km² (%.1f%%), annual change=%.2f km²/yr, BAU %d=%.1f km²",
        deg_bl,
        pct_bl,
        annual_change if annual_change is not None else 0.0,
        target_year,
        bau_proj if bau_proj is not None else 0.0,
    )

    # --- Per-jurisdiction BAU statistics ---
    if zones_path is not None or zones_raster_path is not None:
        summary["jurisdictions"] = _compute_bau_zones(
            status_baseline_path=status_baseline_path,
            status_baseline_band_index=status_baseline_band_index,
            status_reporting_path=status_reporting_path,
            status_reporting_band_index=status_reporting_band_index,
            year_initial=year_initial,
            year_final=year_final,
            target_year=target_year,
            zones_path=zones_path,
            zones_raster_path=zones_raster_path,
            zones_raster_labels=zones_raster_labels,
            killed_callback=killed_callback,
            status_baseline_type=bl_type,
            status_reporting_type=(
                rp_type if status_reporting_path is not None else status_reporting_type
            ),
            aoi_path=aoi_path,
            aoi_srs=aoi_srs,
            aoi=aoi,
        )
    else:
        summary["jurisdictions"] = []

    # --- Per-land-type BAU statistics ---
    if land_type_path is not None:
        summary["land_types_bau"] = _compute_bau_zones(
            status_baseline_path=status_baseline_path,
            status_baseline_band_index=status_baseline_band_index,
            status_reporting_path=status_reporting_path,
            status_reporting_band_index=status_reporting_band_index,
            year_initial=year_initial,
            year_final=year_final,
            target_year=target_year,
            zones_path=None,
            zones_raster_path=land_type_path,
            zones_raster_labels=land_type_labels,
            zones_raster_band_index=land_type_band_index,
            killed_callback=killed_callback,
            status_baseline_type=bl_type,
            status_reporting_type=(
                rp_type if status_reporting_path is not None else status_reporting_type
            ),
            aoi_path=aoi_path,
            aoi_srs=aoi_srs,
            aoi=aoi,
        )
    else:
        summary["land_types_bau"] = []

    return summary


def _compute_bau_zones(
    status_baseline_path: str,
    status_baseline_band_index: int,
    status_reporting_path: Optional[str],
    status_reporting_band_index: int,
    year_initial: int,
    year_final: int,
    target_year: int,
    zones_path: Optional[str] = None,
    zones_raster_path: Optional[str] = None,
    zones_raster_labels: Optional[Dict[Any, str]] = None,
    zones_raster_band_index: int = 1,
    killed_callback=None,
    status_baseline_type: str = "auto",
    status_reporting_type: str = "auto",
    aoi_path: Optional[str] = None,
    aoi_srs: Optional[str] = None,
    aoi=None,
) -> List[Dict[str, Any]]:
    """Compute BAU statistics per zone.

    Zones may come from an uploaded/admin vector (``zones_path``) or a saved
    raster (``zones_raster_path``). Statistics are grouped on a single
    zone-id array aligned to the baseline grid, so all zone sources behave
    identically.
    """
    bl_ds, bl_band = _open_band(status_baseline_path, status_baseline_band_index)
    bl_arr = bl_band.ReadAsArray()
    bl_gt = bl_ds.GetGeoTransform()
    proj = bl_ds.GetProjection()
    bl_nd = bl_band.GetNoDataValue()
    if bl_nd is None:
        bl_nd = float(config.NODATA_VALUE)
    bl_type = _detect_status_type(
        bl_ds, status_baseline_band_index, status_baseline_type
    )
    bl_deg_set = (
        _STATUS_7CLASS_DEGRADED if bl_type == "7class" else _STATUS_3CLASS_DEGRADED
    )
    xsize = bl_ds.RasterXSize
    ysize = bl_ds.RasterYSize
    pixel_areas = _compute_pixel_areas_km2(
        bl_gt, 0, ysize, xsize, bl_ds.GetProjection()
    )
    aoi_geometries = _prepare_aoi_geometries(aoi_path, proj, aoi_srs, aoi)
    aoi_mask = _rasterize_aoi_block(aoi_geometries, bl_gt, proj, 0, 0, xsize, ysize)

    rp_arr = rp_deg_set = rp_nd = None
    if status_reporting_path is not None:
        rp_ds, rp_band = _aligned_band(
            status_reporting_path, status_reporting_band_index, bl_ds
        )
        rp_arr = rp_band.ReadAsArray()
        rp_nd = rp_band.GetNoDataValue()
        if rp_nd is None:
            rp_nd = float(config.NODATA_VALUE)
        rp_type = _detect_status_type(rp_ds, 1, status_reporting_type)
        rp_deg_set = (
            _STATUS_7CLASS_DEGRADED if rp_type == "7class" else _STATUS_3CLASS_DEGRADED
        )
        del rp_ds, rp_band

    del bl_ds, bl_band

    zone_ids, id_to_name = zones.resolve_zone_ids(
        bl_gt,
        proj,
        xsize,
        ysize,
        zones_path=zones_path,
        zones_raster_path=zones_raster_path,
        zones_raster_labels=zones_raster_labels,
        zones_raster_band_index=zones_raster_band_index,
    )
    if zone_ids is None:
        return []

    bl_valid = (
        np.isfinite(bl_arr) & (bl_arr != bl_nd) & (bl_arr != int(config.NODATA_VALUE))
    )
    _validate_status_values(bl_arr[bl_valid], bl_type)
    if aoi_mask is not None:
        bl_valid &= aoi_mask
    bl_is_deg = np.isin(bl_arr, list(bl_deg_set))
    rp_valid = rp_is_deg = None
    if rp_arr is not None:
        rp_valid = (
            np.isfinite(rp_arr)
            & (rp_arr != rp_nd)
            & (rp_arr != int(config.NODATA_VALUE))
        )
        _validate_status_values(rp_arr[rp_valid], rp_type)
        if aoi_mask is not None:
            rp_valid &= aoi_mask
        rp_is_deg = np.isin(rp_arr, list(rp_deg_set))

    n_years = year_final - year_initial
    results = []
    for code in np.unique(zone_ids):
        if killed_callback and killed_callback():
            raise RuntimeError("BAU zone calculation cancelled.")
        code = int(code)
        if code == 0:
            continue
        gmask = zone_ids == code
        gbl_valid = gmask & bl_valid
        total = float(np.sum(pixel_areas[gbl_valid]))
        deg_bl = float(np.sum(pixel_areas[gbl_valid & bl_is_deg]))

        deg_rp = annual_change = bau_proj = shortfall = None
        if rp_arr is not None:
            grp_valid = gmask & rp_valid
            deg_rp = float(np.sum(pixel_areas[grp_valid & rp_is_deg]))
            if n_years > 0:
                annual_change = (deg_rp - deg_bl) / n_years
                bau_proj = deg_rp + annual_change * (target_year - year_final)
                shortfall = max(0.0, bau_proj - deg_bl)

        results.append(
            {
                "zone_name": id_to_name.get(code, str(code)),
                "zone_id": code,
                "total_area_km2": total,
                "degraded_area_baseline_km2": deg_bl,
                "pct_degraded_baseline": 100.0 * deg_bl / total if total > 0 else 0.0,
                "degraded_area_reporting_km2": deg_rp,
                "pct_degraded_reporting": (
                    100.0 * deg_rp / total
                    if (deg_rp is not None and total > 0)
                    else None
                ),
                "annual_change_km2": annual_change,
                f"bau_projection_{target_year}_km2": bau_proj,
                "ldntarget_km2": deg_bl,
                "shortfall_km2": shortfall,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 4. Scenario builder
# ---------------------------------------------------------------------------


def _validate_effectiveness(value: Any, target_index: int) -> float:
    try:
        effectiveness = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Target {target_index} effectiveness must be a number in [0, 1]."
        ) from error
    if not math.isfinite(effectiveness) or not 0 <= effectiveness <= 1:
        raise ValueError(
            f"Target {target_index} effectiveness must be finite and in [0, 1]."
        )
    return effectiveness


def _target_geometry_in_reference_crs(
    target: Dict[str, Any], target_index: int, target_projection: str
) -> Optional[ogr.Geometry]:
    wkt = target.get("wkt_geometry", "")
    geometry = ogr.CreateGeometryFromWkt(wkt) if wkt else None
    if geometry is None or geometry.IsEmpty() or not geometry.IsValid():
        return None
    geometry_crs = target.get("geometry_crs", target.get("srs"))
    if geometry_crs:
        source_srs = osr.SpatialReference()
        if source_srs.SetFromUserInput(str(geometry_crs)) != 0:
            raise ValueError(f"Target {target_index} has an invalid geometry CRS.")
        target_srs = osr.SpatialReference()
        if not target_projection or target_srs.ImportFromWkt(target_projection) != 0:
            raise ValueError("ARR raster has no valid CRS.")
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not source_srs.IsSame(target_srs):
            transform = osr.CoordinateTransformation(source_srs, target_srs)
            if geometry.Transform(transform) != 0:
                raise ValueError(
                    f"Target {target_index} geometry could not be transformed."
                )
    return geometry


def apply_scenario(
    arr_path: str,
    arr_band_index: int = 1,
    targets: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    land_type_path: Optional[str] = None,
    land_type_band_index: int = 1,
    land_type_labels: Optional[Dict[int, str]] = None,
    zones_path: Optional[str] = None,
    zones_raster_path: Optional[str] = None,
    zones_raster_labels: Optional[Dict[Any, str]] = None,
    progress_callback=None,
    killed_callback=None,
    aoi=None,
) -> Tuple[Dict[str, Any], str]:
    """Apply planning targets to the ARR layer using an expected-value model.

    Rather than flipping arbitrary pixels, each eligible pixel within a target
    is assigned a *probability* of being successfully treated (equal to the
    target's ``effectiveness``; overlapping targets take the maximum). The
    spatial output is a 3-band probability raster:

        Band 1: P(restoration effect)  — over Reverse (degraded) pixels
        Band 2: P(reduction effect)    — over Reduce (at-risk) pixels
        Band 3: P(avoidance effect)    — over Avoid (healthy) pixels

    Expected areas are then accumulated per land type and per jurisdiction
    (zone) for the balance sheet.

    Intervention roles (Cowie et al. 2018, §3.4; GPG Addendum rule 4):
      - *reverse*: restoration on degraded land → counterbalancing GAINS.
      - *reduce* : SLM on at-risk land          → AVOIDED LOSSES (not gains).
      - *avoid*  : protection of healthy land    → AVOIDED LOSSES (not gains).
      - *auto*   : per-pixel intervention follows the pixel's ARR class.

    Args:
        arr_path: ARR raster from ``compute_arr_classification``.
        arr_band_index: Band index.
        targets: List of dicts with keys wkt_geometry, intervention, effectiveness.
        output_path: Output probability raster path.
        land_type_path: Optional raster whose values define land types (e.g. the
            baseline land cover proxy, per GPG). Aligned to the ARR grid.
        land_type_band_index: Band index in the land-type raster.
        land_type_labels: Optional {code: name} mapping for land-type reporting.
        zones_path: Optional vector layer defining jurisdictions for per-zone stats.
        progress_callback: Optional callable(pct: float).
        killed_callback: Optional callable() → bool.

    Returns:
        (summary_dict, output_raster_path). summary_dict includes the national
        totals plus ``by_land_type`` and ``by_jurisdiction`` lists.

    Phase 1 note: avoided losses are reported as an *upper bound*
    (expected treated area of at-risk/healthy land); weighting by BAU
    degradation risk is a Phase 2 enhancement.
    """
    if targets is None:
        targets = []
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(
            suffix="_ldn_scenario.tif", delete=False
        ).name

    arr_ds, arr_band = _open_band(arr_path, arr_band_index)
    xsize = arr_ds.RasterXSize
    ysize = arr_ds.RasterYSize
    gt = arr_ds.GetGeoTransform()
    proj = arr_ds.GetProjection()
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError("ARR raster must use a north-up grid.")
    nd = arr_band.GetNoDataValue()
    if nd is None:
        nd = float(ARR_NODATA)
    nd = int(nd)

    arr_full = arr_band.ReadAsArray().astype(np.int16)
    pixel_areas = _compute_pixel_areas_km2(gt, 0, ysize, xsize, proj)
    aoi_geometries = _prepare_aoi_geometries(None, proj, aoi=aoi)

    valid = arr_full != nd
    aoi_mask = _rasterize_aoi_block(aoi_geometries, gt, proj, 0, 0, xsize, ysize)
    if aoi_mask is not None:
        valid &= aoi_mask
    rev_mask = valid & (arr_full == ARR_REVERSE)
    red_mask = valid & (arr_full == ARR_REDUCE)
    avo_mask = valid & (arr_full == ARR_AVOID)

    # Probability surfaces (int percent 0-100), nodata outside the relevant class
    p_reverse = np.full((ysize, xsize), ARR_NODATA, dtype=np.int16)
    p_reduce = np.full((ysize, xsize), ARR_NODATA, dtype=np.int16)
    p_avoid = np.full((ysize, xsize), ARR_NODATA, dtype=np.int16)
    p_reverse[rev_mask] = 0
    p_reduce[red_mask] = 0
    p_avoid[avo_mask] = 0

    band_for = {
        ARR_REVERSE: (p_reverse, rev_mask),
        ARR_REDUCE: (p_reduce, red_mask),
        ARR_AVOID: (p_avoid, avo_mask),
    }
    intervention_class = {
        "reverse": [ARR_REVERSE],
        "reduce": [ARR_REDUCE],
        "avoid": [ARR_AVOID],
        "auto": [ARR_REVERSE, ARR_REDUCE, ARR_AVOID],
    }

    per_target_stats = []
    skipped_targets = []
    mem_drv = gdal.GetDriverByName("MEM")

    for t_idx, target in enumerate(targets):
        if killed_callback and killed_callback():
            raise RuntimeError("Scenario calculation cancelled.")
        intervention = str(target.get("intervention", "reverse")).lower()
        if intervention not in intervention_class:
            raise ValueError(
                f"Target {t_idx} has unknown intervention '{intervention}'."
            )
        effectiveness = _validate_effectiveness(target.get("effectiveness", 1.0), t_idx)
        pct = int(round(effectiveness * 100))

        selected_mask_path = target.get("selected_mask_path")
        selected_mask = None
        if selected_mask_path:
            mask_ds, mask_band = _aligned_band(
                selected_mask_path,
                int(target.get("band", 1)),
                arr_ds,
            )
            selected_values = mask_band.ReadAsArray()
            mask_nd = mask_band.GetNoDataValue()
            if mask_nd is None:
                mask_nd = ARR_NODATA
            mask_valid = np.isfinite(selected_values) & (selected_values != mask_nd)
            if np.any(
                selected_values[mask_valid] != np.floor(selected_values[mask_valid])
            ):
                raise ValueError(
                    f"Target {t_idx} selected mask must use integer codes."
                )
            valid_mask_values = set(
                int(value) for value in np.unique(selected_values[mask_valid])
            )
            if not valid_mask_values <= {0, ARR_AVOID, ARR_REDUCE, ARR_REVERSE}:
                raise ValueError(
                    f"Target {t_idx} selected mask contains values outside 0, 1, 2, 3."
                )
            selected_mask = np.where(mask_valid, selected_values, ARR_NODATA).astype(
                np.int16
            )
            expected_classes = {
                "avoid": ARR_AVOID,
                "reduce": ARR_REDUCE,
                "reverse": ARR_REVERSE,
            }
            # One candidate mask may hold several classes; each class target
            # uses only its own pixels but must select at least one of them.
            if (
                intervention != "auto"
                and expected_classes[intervention] not in valid_mask_values
            ):
                raise ValueError(
                    f"Target {t_idx} intervention does not match its selected mask class."
                )
        else:
            geom = _target_geometry_in_reference_crs(target, t_idx, proj)
        if not selected_mask_path and geom is None:
            skipped_targets.append(
                {"target_index": t_idx, "reason": "invalid or empty geometry"}
            )
            logger.warning("Target %d: invalid or empty geometry, skipping", t_idx)
            continue

        if selected_mask_path:
            row_min, row_max, col_min, col_max = 0, ysize - 1, 0, xsize - 1
        else:
            env = geom.GetEnvelope()
            col_min = max(0, int((env[0] - gt[0]) / gt[1]))
            col_max = min(xsize - 1, int(math.ceil((env[1] - gt[0]) / gt[1])))
            row_min = max(0, int((env[3] - gt[3]) / gt[5]))
            row_max = min(ysize - 1, int(math.ceil((env[2] - gt[3]) / gt[5])))
            if col_max < col_min or row_max < row_min:
                skipped_targets.append(
                    {"target_index": t_idx, "reason": "outside ARR raster"}
                )
                continue
        win_w = col_max - col_min + 1
        win_h = row_max - row_min + 1

        if selected_mask_path:
            mask_win = selected_mask[row_min : row_max + 1, col_min : col_max + 1]
        else:
            mask_ds = mem_drv.Create("", win_w, win_h, 1, gdal.GDT_Byte)
            win_gt = (
                gt[0] + col_min * gt[1],
                gt[1],
                0.0,
                gt[3] + row_min * gt[5],
                0.0,
                gt[5],
            )
            mask_ds.SetGeoTransform(win_gt)
            mask_ds.SetProjection(proj)
            mask_ds.GetRasterBand(1).Fill(0)
            tmp_ds = ogr.GetDriverByName("Memory").CreateDataSource("tmp")
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj)
            tmp_lyr = tmp_ds.CreateLayer("tmp", srs=srs, geom_type=ogr.wkbUnknown)
            tmp_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
            tmp_feat.SetGeometry(geom)
            tmp_lyr.CreateFeature(tmp_feat)
            if gdal.RasterizeLayer(mask_ds, [1], tmp_lyr, burn_values=[1]) != 0:
                raise RuntimeError(f"Could not rasterize target {t_idx}.")
            mask_win = mask_ds.GetRasterBand(1).ReadAsArray() == 1
            del mask_ds, tmp_ds

        rs, re, cs, ce = row_min, row_max + 1, col_min, col_max + 1
        arr_win = arr_full[rs:re, cs:ce]
        areas_win = pixel_areas[rs:re, cs:ce]
        if selected_mask_path:
            selected = mask_win
            eligible_selected = (
                selected > 0
                if intervention == "auto"
                else selected == expected_classes[intervention]
            )
            mask_win = eligible_selected
            mask_values = selected
        else:
            mask_values = None
        if aoi_mask is not None:
            mask_win &= aoi_mask[rs:re, cs:ce]

        t_gains = 0.0
        t_avoided = 0.0
        if selected_mask_path and intervention == "auto":
            target_classes = [ARR_AVOID, ARR_REDUCE, ARR_REVERSE]
        elif selected_mask_path:
            target_classes = [expected_classes[intervention]]
        else:
            target_classes = intervention_class[intervention]
        for cls in target_classes:
            band, _ = band_for[cls]
            eligible_win = mask_win & (arr_win == cls)
            if mask_values is not None:
                eligible_win &= mask_values == cls
            if not np.any(eligible_win):
                continue
            band_win = band[rs:re, cs:ce]
            # Assign probability (max on overlap)
            band_win[eligible_win] = np.maximum(band_win[eligible_win], pct)
            treated_area = float(np.sum(areas_win[eligible_win]) * effectiveness)
            if cls == ARR_REVERSE:
                t_gains += treated_area
            else:
                t_avoided += treated_area

        per_target_stats.append(
            {
                "target_index": t_idx,
                "intervention": intervention,
                "effectiveness": effectiveness,
                "area_treated_km2": float(
                    np.sum(areas_win[mask_win & (arr_win != nd) & valid[rs:re, cs:ce]])
                ),
                "gains_km2": t_gains,
                "avoided_losses_km2": t_avoided,
            }
        )
        if progress_callback:
            progress_callback(70.0 * (t_idx + 1) / max(len(targets), 1))
        if selected_mask_path:
            del mask_ds, mask_band

    # Expected-area surfaces (km²) from the probability bands
    gain_area = np.where(rev_mask, pixel_areas * (p_reverse / 100.0), 0.0)
    avoid_reduce_area = np.where(red_mask, pixel_areas * (p_reduce / 100.0), 0.0)
    avoid_avoid_area = np.where(avo_mask, pixel_areas * (p_avoid / 100.0), 0.0)

    # National totals (overlap-safe: computed from the probability surfaces)
    gains_km2_reverse = float(np.sum(gain_area))
    avoided_losses_km2_reduce = float(np.sum(avoid_reduce_area))
    avoided_losses_km2_avoid = float(np.sum(avoid_avoid_area))

    # Land-type grid
    if land_type_path:
        lt_full = zones.align_raster_to_ref(
            land_type_path, land_type_band_index, gt, proj, xsize, ysize
        )
    else:
        lt_full = np.ones((ysize, xsize), dtype=np.int16)
    labels = land_type_labels or {}

    def _group_stats(group_arr, id_to_name, is_land_type):
        out = []
        codes = np.unique(group_arr[valid])
        for code in codes:
            code = int(code)
            if is_land_type and code == int(ARR_NODATA):
                continue
            if (not is_land_type) and code == 0:
                continue
            gmask = valid & (group_arr == code)
            g = float(np.sum(gain_area[gmask]))
            ar = float(np.sum(avoid_reduce_area[gmask]))
            aa = float(np.sum(avoid_avoid_area[gmask]))
            if is_land_type:
                name = labels.get(code, f"Land type {code}")
            else:
                name = id_to_name.get(code, str(code))
            out.append(
                {
                    "name": name,
                    "code": code,
                    "scenario_effects_are_observed": False,
                    "gains_km2": g,
                    "avoided_losses_reduce_km2": ar,
                    "avoided_losses_avoid_km2": aa,
                    "total_gains_km2": g,
                    "total_avoided_losses_km2": ar + aa,
                    "avoided_losses_are_upper_bound": True,
                }
            )
        out.sort(key=lambda r: r["total_gains_km2"], reverse=True)
        return out

    by_land_type = _group_stats(lt_full, {}, True)

    by_jurisdiction = []
    zone_ids, id_to_name = zones.resolve_zone_ids(
        gt,
        proj,
        xsize,
        ysize,
        zones_path=zones_path,
        zones_raster_path=zones_raster_path,
        zones_raster_labels=zones_raster_labels,
    )
    if zone_ids is not None:
        by_jurisdiction = _group_stats(zone_ids, id_to_name, False)

    if progress_callback:
        progress_callback(90.0)

    # Write the 3-band probability raster
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        xsize,
        ysize,
        3,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    for i, band_arr in enumerate((p_reverse, p_reduce, p_avoid), start=1):
        b = out_ds.GetRasterBand(i)
        b.SetNoDataValue(ARR_NODATA)
        b.WriteArray(band_arr)
        b.FlushCache()
    out_ds.FlushCache()
    del out_ds, arr_ds, arr_band

    total_gains = gains_km2_reverse
    total_avoided_losses = avoided_losses_km2_reduce + avoided_losses_km2_avoid

    summary = {
        "gains_km2_reverse": gains_km2_reverse,
        "avoided_losses_km2_reduce": avoided_losses_km2_reduce,
        "avoided_losses_km2_avoid": avoided_losses_km2_avoid,
        "total_gains_km2": total_gains,
        "total_avoided_losses_km2": total_avoided_losses,
        "net_balance_note": (
            "Expected-value planning model. Gains (from Reverse targets) are the "
            "only counterbalancing gains; Avoid/Reduce yield avoided losses "
            "(upper bound in this phase) — Cowie et al. 2018, rule 4. "
            "Per-pixel probabilities are in the output raster bands."
        ),
        "per_target": per_target_stats,
        "skipped_targets": skipped_targets,
        "by_land_type": by_land_type,
        "by_jurisdiction": by_jurisdiction,
        "net_balance_interpretation": (
            "National scenario totals are aggregate projected effects, not a "
            "like-for-like LDN determination. Treat avoided-loss estimates as "
            "upper bounds; review projected effects by land type separately."
        ),
        "scenario_effects_are_observed": False,
        "avoided_losses_are_upper_bound": True,
    }
    logger.info(
        "Scenario: gains=%.1f km² (Reverse), avoided_losses=%.1f km² "
        "(Reduce=%.1f + Avoid=%.1f); %d land types, %d jurisdictions",
        total_gains,
        total_avoided_losses,
        avoided_losses_km2_reduce,
        avoided_losses_km2_avoid,
        len(by_land_type),
        len(by_jurisdiction),
    )

    if progress_callback:
        progress_callback(100.0)

    return summary, output_path


# ---------------------------------------------------------------------------
# 5. BAU vs. scenario comparison (over the planning horizon)
# ---------------------------------------------------------------------------


def compare_scenarios(params, projection):
    """Compare a projection with the saved, input-compatible scenario in params."""
    previous = params.get("comparison")
    if not previous:
        return None
    prior_projection = previous.get("projection")
    if projection is None or prior_projection is None:
        raise ValueError(
            "Both scenarios need reporting-period projections for comparison."
        )
    old = previous["parameters"]
    if not params.get("input_checksums") or not old.get("input_checksums"):
        raise ValueError(
            "Scenario comparison requires recorded input checksums in both jobs."
        )
    for key in (
        "arr_layer_path",
        "arr_band_index",
        "status_baseline_path",
        "status_baseline_band_index",
        "status_reporting_path",
        "status_reporting_band_index",
        "status_baseline_type",
        "status_reporting_type",
        "year_initial",
        "year_final",
        "target_year",
        "land_type_path",
        "land_type_band_index",
        "land_type_labels",
        "jurisdictions_path",
        "input_checksums",
        "aoi_checksum",
    ):
        if old.get(key) != params.get(key):
            raise ValueError(f"Scenario comparison requires the same inputs: {key}")
    rows = []
    for key in (
        "scenario_gains_km2",
        "scenario_avoided_losses_km2",
        "scenario_degraded_km2",
        "remaining_shortfall_km2",
        "gap_closed_pct",
    ):
        current = projection[key]
        prior = prior_projection[key]
        rows.append(
            {
                "metric": key,
                "previous": prior,
                "current": current,
                "difference": current - prior,
            }
        )
    return {
        "previous_job_id": previous["job_id"],
        "previous_task_name": previous["task_name"],
        "metrics": rows,
        "note": "Planning estimates with upper-bound avoided losses, not observed neutrality.",
    }


def project_scenario_against_bau(
    bau_summary: Dict[str, Any],
    scenario_summary: Dict[str, Any],
    target_year: int,
) -> Optional[Dict[str, Any]]:
    """Compare a scenario against the BAU trajectory at the target year.

    The scenario's counterbalancing **gains** (from Reverse) and **avoided
    losses** (from Avoid/Reduce) are subtracted from the BAU-projected degraded
    area to estimate the scenario-adjusted degraded area at ``target_year``.
    Neutrality is achieved when that value is at or below the LDN target
    (= baseline degraded area; Cowie et al. 2018, Module B).

    Avoided losses are credited as an upper bound (all treated at-risk / healthy
    land assumed to degrade under BAU); BAU-risk weighting is a Phase-2 item.

    Returns ``None`` when the BAU summary lacks a reporting period (no
    projection is possible). Otherwise returns a dict with national trajectory
    points, the neutrality flag, and — when both sides expose per-zone data —
    a ``by_jurisdiction`` list joining BAU shortfall with scenario contributions.
    """
    proj_key = f"bau_projection_{target_year}_km2"
    bau_projection = bau_summary.get(proj_key)
    if bau_projection is None:
        return None

    ldntarget = bau_summary.get("ldntarget_km2", 0.0) or 0.0
    gains = scenario_summary.get("total_gains_km2", 0.0) or 0.0
    avoided = scenario_summary.get("total_avoided_losses_km2", 0.0) or 0.0
    contribution = gains + avoided

    scenario_degraded = max(0.0, bau_projection - contribution)
    bau_shortfall = max(0.0, bau_projection - ldntarget)
    remaining_shortfall = max(0.0, scenario_degraded - ldntarget)
    gap_closed_pct = (
        100.0 * min(contribution, bau_shortfall) / bau_shortfall
        if bau_shortfall > 0
        else 100.0
    )

    def _zone_projection(zone: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        z_bau = zone.get(proj_key)
        if z_bau is None:
            return None
        z_target = zone.get("ldntarget_km2", 0.0) or 0.0
        contrib = scen_contrib_by_jurisdiction.get(zone.get("zone_name"))
        contrib = 0.0 if contrib is None else contrib
        z_scen = max(0.0, z_bau - contrib)
        z_shortfall = max(0.0, z_bau - z_target)
        return {
            "zone_name": zone.get("zone_name"),
            "bau_projection_km2": z_bau,
            "ldntarget_km2": z_target,
            "scenario_contribution_km2": contrib,
            "scenario_degraded_km2": z_scen,
            "neutral": z_scen <= z_target + 1e-9,
            "neutrality_is_like_for_like": False,
            "gap_closed_pct": (
                100.0 * min(contrib, z_shortfall) / z_shortfall
                if z_shortfall > 0
                else 100.0
            ),
        }

    by_jurisdiction_out: List[Dict[str, Any]] = []
    bau_jurisdictions = bau_summary.get("jurisdictions") or []
    scen_jurisdictions = scenario_summary.get("by_jurisdiction") or []
    if bau_jurisdictions and scen_jurisdictions:
        scen_contrib_by_jurisdiction = {
            z.get("name"): (z.get("total_gains_km2", 0.0) or 0.0)
            + (z.get("total_avoided_losses_km2", 0.0) or 0.0)
            for z in scen_jurisdictions
        }
        for zone in bau_jurisdictions:
            zp = _zone_projection(zone)
            if zp is not None:
                by_jurisdiction_out.append(zp)

    return {
        "target_year": target_year,
        "year_initial": bau_summary.get("year_initial"),
        "year_final": bau_summary.get("year_final"),
        "degraded_area_baseline_km2": bau_summary.get("degraded_area_baseline_km2"),
        "degraded_area_reporting_km2": bau_summary.get("degraded_area_reporting_km2"),
        "bau_projection_km2": bau_projection,
        "ldntarget_km2": ldntarget,
        "scenario_gains_km2": gains,
        "scenario_avoided_losses_km2": avoided,
        "scenario_contribution_km2": contribution,
        "scenario_degraded_km2": scenario_degraded,
        "bau_shortfall_km2": bau_shortfall,
        "remaining_shortfall_km2": remaining_shortfall,
        "gap_closed_pct": gap_closed_pct,
        "neutral": scenario_degraded <= ldntarget + 1e-9,
        "neutrality_is_like_for_like": False,
        "by_jurisdiction": by_jurisdiction_out,
        "note": (
            "Scenario-adjusted degraded area = BAU projection - gains - avoided "
            "losses. Avoided losses are an upper bound (all treated at-risk / "
            "healthy land assumed to degrade under BAU). Neutrality: "
            "scenario-adjusted degraded area <= baseline (LDN target). This is a "
            "simplified aggregate indicator, not a like-for-like LDN determination; "
            "review results by land type separately."
        ),
    }
