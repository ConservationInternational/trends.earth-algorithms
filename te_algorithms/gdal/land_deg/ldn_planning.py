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

import logging
import math
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

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

# Status values that indicate "overall degraded" in each status type
# 3-class: -1 = degraded (0 = stable, 1 = improved)
# 7-class (expanded status matrix):
#   1 = persistent degradation, 2 = recent degradation, 3 = baseline degradation
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
) -> np.ndarray:
    """Compute per-pixel area in km² for a raster block on WGS84.

    Uses the geodetically correct ``calc_cell_area`` function so that areas
    are accurate across latitudes.
    """
    pixel_w_deg = abs(gt[1])
    areas = np.empty((y_size, x_size), dtype=np.float64)
    for row in range(y_size):
        # Top latitude of this pixel row
        lat_max = gt[3] + (y_off + row) * gt[5]
        lat_min = lat_max + gt[5]
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        area_m2 = calc_cell_area(lat_min, lat_max, pixel_w_deg)
        areas[row, :] = area_m2 / 1e6  # m² → km²
    return areas


def _open_band(path: str, band_index: int = 1) -> Tuple[gdal.Dataset, gdal.Band]:
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(band_index)
    if band is None:
        raise RuntimeError(f"Band {band_index} not found in {path}")
    return ds, band


def _create_output_raster(
    output_path: str,
    ref_ds: gdal.Dataset,
    nodata: int = ARR_NODATA,
    dtype: int = gdal.GDT_Int16,
) -> gdal.Dataset:
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        dtype,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    b = out_ds.GetRasterBand(1)
    b.SetNoDataValue(nodata)
    b.Fill(nodata)
    return out_ds


def _detect_status_type(ds: gdal.Dataset, band_index: int = 1) -> str:
    """Return ``'3class'`` or ``'7class'`` from the raster band's actual values.

    The 3-class SDG 15.3.1 indicator encodes values in {-1, 0, 1} (degraded /
    stable / improved). The 7-class GPG "expanded status" band is all-positive,
    with values in {1..7} (1-3 degradation, 4 stability, 5-7 improvement).

    ``band_index`` MUST match the band actually used for the analysis — the SDG
    output is multi-band and the status band is generally not band 1.

    Detection reads the band and explicitly masks nodata (both the band's
    declared nodata and the -32768 sentinel). This avoids the failure mode
    where ``ComputeStatistics`` reports ``min == -32768`` for a band whose
    nodata value is not set in metadata, which previously caused a 7-class
    status band to be misdetected as 3-class (so no degradation was found).
    """
    band = ds.GetRasterBand(band_index)
    arr = band.ReadAsArray()
    if arr is None:
        return "7class"
    arr = arr.ravel()
    nd = band.GetNoDataValue()
    mask = arr != int(config.NODATA_VALUE)
    if nd is not None:
        mask &= arr != int(nd)
    valid = arr[mask]
    if valid.size == 0:
        return "7class"

    uniques = set(int(v) for v in np.unique(valid))
    # The values -1 and 0 only exist in the 3-class indicator.
    if -1 in uniques or 0 in uniques:
        return "3class"
    # The values 4-7 only exist in the 7-class expanded status.
    if any(v > 3 for v in uniques):
        return "7class"
    if any(v < 0 for v in uniques):
        return "3class"
    # Only values within {1, 2, 3} are present (no 0/-1 and nothing above 3).
    # This is the 7-class expanded status showing only degradation classes,
    # which is exactly the case where Reverse pixels must be produced.
    return "7class"


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
        degraded and hence Reverse) or risk_layer value > 0. SLM here avoids
        losses.
      - **Avoid** (1): not degraded and not flagged at risk. Protection
        prevents losses but does not generate gains.
      - If no LPD or risk layer is supplied, all non-degraded land defaults to
        **Avoid**; users may designate Reduce zones via uploaded polygons in
        the scenario step.

    The ``trajectory_path`` argument expects a 5-class Land Productivity
    Dynamics (LPD) layer whose values follow PRODUCTIVITY_CLASS_KEY.

    Output raster values: 1=Avoid, 2=Reduce, 3=Reverse, -32768=NoData.
    """
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(
            suffix="_ldn_arr.tif", delete=False
        ).name

    status_ds, status_band = _open_band(status_path, status_band_index)
    xsize = status_ds.RasterXSize
    ysize = status_ds.RasterYSize
    gt = status_ds.GetGeoTransform()

    nodata_val = status_band.GetNoDataValue()
    if nodata_val is None:
        nodata_val = float(config.NODATA_VALUE)

    status_type = _detect_status_type(status_ds, status_band_index)
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
        traj_ds, traj_band = _open_band(trajectory_path, trajectory_band_index)

    risk_ds = risk_band_obj = None
    if risk_layer_path:
        risk_ds, risk_band_obj = _open_band(risk_layer_path, risk_band_index)

    out_ds = _create_output_raster(output_path, status_ds)
    out_band = out_ds.GetRasterBand(1)

    avoid_km2 = reduce_km2 = reverse_km2 = nodata_km2 = 0.0
    n_blocks_y = math.ceil(ysize / BLOCK_SIZE)
    n_blocks_x = math.ceil(xsize / BLOCK_SIZE)
    total_blocks = n_blocks_y * n_blocks_x
    block_count = 0

    for y_off in range(0, ysize, BLOCK_SIZE):
        if killed_callback and killed_callback():
            break
        y_blk = min(BLOCK_SIZE, ysize - y_off)

        for x_off in range(0, xsize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                break
            x_blk = min(BLOCK_SIZE, xsize - x_off)

            status_arr = status_band.ReadAsArray(x_off, y_off, x_blk, y_blk).astype(
                np.int16
            )

            pixel_areas = _compute_pixel_areas_km2(gt, y_off, y_blk, x_blk)

            valid = status_arr != int(nodata_val)

            # --- Reverse: overall degraded ---
            is_degraded = np.zeros(status_arr.shape, dtype=bool)
            for dv in degraded_set:
                is_degraded |= status_arr == dv

            # --- Reduce / Avoid: at-risk signal from trajectory or risk layer ---
            not_degraded = valid & ~is_degraded
            is_at_risk = np.zeros(status_arr.shape, dtype=bool)

            if traj_band is not None:
                traj_arr = traj_band.ReadAsArray(x_off, y_off, x_blk, y_blk).astype(
                    np.int16
                )
                traj_nd = traj_band.GetNoDataValue()
                if traj_nd is None:
                    traj_nd = float(config.NODATA_VALUE)
                for rv in _LPD_STRESSED:
                    is_at_risk |= (traj_arr == rv) & (traj_arr != int(traj_nd))

            if risk_band_obj is not None:
                risk_arr = risk_band_obj.ReadAsArray(x_off, y_off, x_blk, y_blk).astype(
                    np.int16
                )
                risk_nd = risk_band_obj.GetNoDataValue()
                if risk_nd is None:
                    risk_nd = float(config.NODATA_VALUE)
                is_at_risk |= (risk_arr > 0) & (risk_arr != int(risk_nd))

            # Build output block
            arr = np.full(status_arr.shape, ARR_NODATA, dtype=np.int16)
            arr[valid & is_degraded] = ARR_REVERSE
            arr[not_degraded & is_at_risk] = ARR_REDUCE
            arr[not_degraded & ~is_at_risk] = ARR_AVOID

            # Accumulate areas (km²)
            avoid_km2 += float(np.sum(pixel_areas[arr == ARR_AVOID]))
            reduce_km2 += float(np.sum(pixel_areas[arr == ARR_REDUCE]))
            reverse_km2 += float(np.sum(pixel_areas[arr == ARR_REVERSE]))
            nodata_km2 += float(np.sum(pixel_areas[~valid]))

            out_band.WriteArray(arr, x_off, y_off)

            block_count += 1
            if progress_callback:
                progress_callback(100.0 * block_count / total_blocks)

    out_band.FlushCache()
    out_ds.FlushCache()
    del out_band, out_ds, status_ds, status_band
    if traj_ds:
        del traj_ds, traj_band
    if risk_ds:
        del risk_ds, risk_band_obj

    summary = {
        "avoid_km2": avoid_km2,
        "reduce_km2": reduce_km2,
        "reverse_km2": reverse_km2,
        "nodata_km2": nodata_km2,
        "total_km2": avoid_km2 + reduce_km2 + reverse_km2,
        "status_type": status_type,
        "trajectory_used": trajectory_path is not None,
        "risk_layer_used": risk_layer_path is not None,
    }
    logger.info(
        "ARR: Avoid=%.1f km², Reduce=%.1f km², Reverse=%.1f km²",
        avoid_km2,
        reduce_km2,
        reverse_km2,
    )
    return summary, output_path


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

    status_type = _detect_status_type(status_ds, status_band_index)
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
    pixel_areas_full = _compute_pixel_areas_km2(gt, 0, ysize, xsize)

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
    progress_callback=None,
    killed_callback=None,
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
            the summary dict includes a ``"zones"`` list.
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

    def _count_degraded(path, band_idx):
        ds, band = _open_band(path, band_idx)
        gt = ds.GetGeoTransform()
        xsize, ysize = ds.RasterXSize, ds.RasterYSize
        nd = band.GetNoDataValue()
        if nd is None:
            nd = float(config.NODATA_VALUE)
        status_type = _detect_status_type(ds, band_idx)
        degraded_set = (
            _STATUS_7CLASS_DEGRADED
            if status_type == "7class"
            else _STATUS_3CLASS_DEGRADED
        )
        deg_km2 = 0.0
        total_km2 = 0.0
        for y_off in range(0, ysize, BLOCK_SIZE):
            y_blk = min(BLOCK_SIZE, ysize - y_off)
            arr = band.ReadAsArray(0, y_off, xsize, y_blk).astype(np.int16)
            areas = _compute_pixel_areas_km2(gt, y_off, y_blk, xsize)
            valid = arr != int(nd)
            is_deg = np.zeros(arr.shape, dtype=bool)
            for dv in degraded_set:
                is_deg |= arr == dv
            deg_km2 += float(np.sum(areas[valid & is_deg]))
            total_km2 += float(np.sum(areas[valid]))
        del ds, band
        return deg_km2, total_km2

    if progress_callback:
        progress_callback(10.0)

    deg_bl, total = _count_degraded(status_baseline_path, status_baseline_band_index)

    if progress_callback:
        progress_callback(50.0)

    deg_rep = None
    annual_change = None
    bau_proj = None
    shortfall = None
    n_years = year_final - year_initial

    if status_reporting_path is not None:
        deg_rep, _ = _count_degraded(status_reporting_path, status_reporting_band_index)
        if n_years > 0:
            annual_change = (deg_rep - deg_bl) / n_years
            n_proj = target_year - year_final
            bau_proj = deg_rep + annual_change * n_proj
            shortfall = max(0.0, bau_proj - deg_bl)  # how much above baseline

    if progress_callback:
        progress_callback(100.0)

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
    }
    logger.info(
        "BAU: baseline=%.1f km² (%.1f%%), annual change=%.2f km²/yr, BAU %d=%.1f km²",
        deg_bl,
        pct_bl,
        annual_change if annual_change is not None else 0.0,
        target_year,
        bau_proj if bau_proj is not None else 0.0,
    )

    # --- Per-zone BAU statistics ---
    if zones_path is not None or zones_raster_path is not None:
        summary["zones"] = _compute_bau_zones(
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
        )
    else:
        summary["zones"] = []

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
    killed_callback=None,
) -> List[Dict[str, Any]]:
    """Compute BAU statistics per zone.

    Zones may come from an uploaded/admin vector (``zones_path``) or a saved
    zones raster (``zones_raster_path``). Statistics are grouped on a single
    zone-id array aligned to the baseline grid, so all zone sources behave
    identically.
    """
    bl_ds, bl_band = _open_band(status_baseline_path, status_baseline_band_index)
    bl_arr = bl_band.ReadAsArray().astype(np.int16)
    bl_gt = bl_ds.GetGeoTransform()
    proj = bl_ds.GetProjection()
    bl_nd = bl_band.GetNoDataValue()
    if bl_nd is None:
        bl_nd = float(config.NODATA_VALUE)
    bl_type = _detect_status_type(bl_ds, status_baseline_band_index)
    bl_deg_set = (
        _STATUS_7CLASS_DEGRADED if bl_type == "7class" else _STATUS_3CLASS_DEGRADED
    )
    xsize = bl_ds.RasterXSize
    ysize = bl_ds.RasterYSize
    pixel_areas = _compute_pixel_areas_km2(bl_gt, 0, ysize, xsize)

    rp_arr = rp_deg_set = rp_nd = None
    if status_reporting_path is not None:
        rp_ds, rp_band = _open_band(status_reporting_path, status_reporting_band_index)
        if (
            rp_ds.RasterXSize != xsize
            or rp_ds.RasterYSize != ysize
            or rp_ds.GetGeoTransform() != bl_gt
        ):
            warp_path = tempfile.NamedTemporaryFile(
                suffix="_rp_warp.vrt", delete=False
            ).name
            gdal.Warp(
                warp_path,
                status_reporting_path,
                format="VRT",
                width=xsize,
                height=ysize,
                outputBounds=(
                    bl_gt[0],
                    bl_gt[3] + bl_gt[5] * ysize,
                    bl_gt[0] + bl_gt[1] * xsize,
                    bl_gt[3],
                ),
                resampleAlg=gdal.GRA_NearestNeighbour,
            )
            rp_ds2, rp_band2 = _open_band(warp_path, 1)
            rp_arr = rp_band2.ReadAsArray().astype(np.int16)
            del rp_ds2, rp_band2
        else:
            rp_arr = rp_band.ReadAsArray().astype(np.int16)
        rp_nd = rp_band.GetNoDataValue()
        if rp_nd is None:
            rp_nd = float(config.NODATA_VALUE)
        rp_type = _detect_status_type(rp_ds, status_reporting_band_index)
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
    )
    if zone_ids is None:
        return []

    bl_valid = bl_arr != int(bl_nd)
    bl_is_deg = np.isin(bl_arr, list(bl_deg_set))
    rp_valid = rp_is_deg = None
    if rp_arr is not None:
        rp_valid = rp_arr != int(rp_nd)
        rp_is_deg = np.isin(rp_arr, list(rp_deg_set))

    n_years = year_final - year_initial
    results = []
    for code in np.unique(zone_ids):
        if killed_callback and killed_callback():
            break
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
        totals plus ``by_land_type`` and ``by_zone`` lists.

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
    nd = arr_band.GetNoDataValue()
    if nd is None:
        nd = float(ARR_NODATA)
    nd = int(nd)

    arr_full = arr_band.ReadAsArray().astype(np.int16)
    pixel_areas = _compute_pixel_areas_km2(gt, 0, ysize, xsize)

    valid = arr_full != nd
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
    mem_drv = gdal.GetDriverByName("MEM")

    for t_idx, target in enumerate(targets):
        if killed_callback and killed_callback():
            break
        wkt = target.get("wkt_geometry", "")
        intervention = str(target.get("intervention", "reverse")).lower()
        effectiveness = max(0.0, min(1.0, float(target.get("effectiveness", 1.0))))
        pct = int(round(effectiveness * 100))

        geom = ogr.CreateGeometryFromWkt(wkt)
        if geom is None:
            logger.warning("Target %d: invalid WKT geometry, skipping", t_idx)
            continue

        env = geom.GetEnvelope()
        col_min = max(0, int((env[0] - gt[0]) / gt[1]))
        col_max = min(xsize - 1, int(math.ceil((env[1] - gt[0]) / gt[1])))
        row_min = max(0, int((env[3] - gt[3]) / gt[5]))
        row_max = min(ysize - 1, int(math.ceil((env[2] - gt[3]) / gt[5])))
        if col_max < col_min or row_max < row_min:
            continue
        win_w = col_max - col_min + 1
        win_h = row_max - row_min + 1

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
        tmp_lyr = tmp_ds.CreateLayer("tmp", geom_type=ogr.wkbPolygon)
        tmp_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
        tmp_feat.SetGeometry(geom)
        tmp_lyr.CreateFeature(tmp_feat)
        gdal.RasterizeLayer(mask_ds, [1], tmp_lyr, burn_values=[1])
        mask_win = mask_ds.GetRasterBand(1).ReadAsArray() == 1
        del mask_ds, tmp_ds

        rs, re, cs, ce = row_min, row_max + 1, col_min, col_max + 1
        arr_win = arr_full[rs:re, cs:ce]
        areas_win = pixel_areas[rs:re, cs:ce]

        t_gains = 0.0
        t_avoided = 0.0
        for cls in intervention_class.get(intervention, [ARR_REVERSE]):
            band, _ = band_for[cls]
            eligible_win = mask_win & (arr_win == cls)
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
                    np.sum(areas_win[mask_win & (arr_win != nd)])
                ),
                "gains_km2": t_gains,
                "avoided_losses_km2": t_avoided,
            }
        )
        if progress_callback:
            progress_callback(70.0 * (t_idx + 1) / max(len(targets), 1))

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
                    "gains_km2": g,
                    "avoided_losses_reduce_km2": ar,
                    "avoided_losses_avoid_km2": aa,
                    "total_gains_km2": g,
                    "total_avoided_losses_km2": ar + aa,
                }
            )
        out.sort(key=lambda r: r["total_gains_km2"], reverse=True)
        return out

    by_land_type = _group_stats(lt_full, {}, True)

    by_zone = []
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
        by_zone = _group_stats(zone_ids, id_to_name, False)

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
        "by_land_type": by_land_type,
        "by_zone": by_zone,
    }
    logger.info(
        "Scenario: gains=%.1f km² (Reverse), avoided_losses=%.1f km² "
        "(Reduce=%.1f + Avoid=%.1f); %d land types, %d zones",
        total_gains,
        total_avoided_losses,
        avoided_losses_km2_reduce,
        avoided_losses_km2_avoid,
        len(by_land_type),
        len(by_zone),
    )

    if progress_callback:
        progress_callback(100.0)

    return summary, output_path


# ---------------------------------------------------------------------------
# 5. BAU vs. scenario comparison (over the planning horizon)
# ---------------------------------------------------------------------------


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
    a ``by_zone`` list joining BAU shortfall with scenario contributions.
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
        contrib = scen_contrib_by_zone.get(zone.get("zone_name"))
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
            "gap_closed_pct": (
                100.0 * min(contrib, z_shortfall) / z_shortfall
                if z_shortfall > 0
                else 100.0
            ),
        }

    by_zone_out: List[Dict[str, Any]] = []
    bau_zones = bau_summary.get("zones") or []
    scen_zones = scenario_summary.get("by_zone") or []
    if bau_zones and scen_zones:
        scen_contrib_by_zone = {
            z.get("name"): (z.get("total_gains_km2", 0.0) or 0.0)
            + (z.get("total_avoided_losses_km2", 0.0) or 0.0)
            for z in scen_zones
        }
        for zone in bau_zones:
            zp = _zone_projection(zone)
            if zp is not None:
                by_zone_out.append(zp)

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
        "by_zone": by_zone_out,
        "note": (
            "Scenario-adjusted degraded area = BAU projection - gains - avoided "
            "losses. Avoided losses are an upper bound (all treated at-risk / "
            "healthy land assumed to degrade under BAU). Neutrality: "
            "scenario-adjusted degraded area <= baseline (LDN target)."
        ),
    }
