"""Orchestration for LDN counterbalancing assessment.

Implements the GPG Addendum counterbalancing procedure:
  Steps 1-3 are provided by the existing SDG 15.3.1 workflow (7-class status).
  Step 4: Calculate gain and loss areas per land type.
  Step 5: Assess LDN per land type (delta_i >= 0 means achieved).
"""

import dataclasses
import logging
import multiprocessing
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

from .. import util, workers
from ..util_numba import calc_cell_area
from . import config, models
from .counterbalancing_numba import (
    classify_gains_losses,
    zonal_gains_losses,
)

if TYPE_CHECKING:
    from te_schemas.aoi import AOI

logger = logging.getLogger(__name__)


def _get_aoi_bounds(aoi: "AOI") -> Tuple[float, float, float, float]:
    """Extract bounding box from AOI geojson as (minx, miny, maxx, maxy)."""

    def _collect(obj, result):
        if isinstance(obj[0], (int, float)):
            result.append(obj)
        else:
            for item in obj:
                _collect(item, result)

    coords: list = []
    for feature in aoi.geojson.get("features", [aoi.geojson]):
        geom = feature.get("geometry", feature)
        _collect(geom["coordinates"], coords)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def prepare_land_type_vrts(
    layer_paths: List[str],
    reference_raster: str,
    aoi: Optional["AOI"] = None,
) -> Tuple[List[str], int, tuple]:
    """Create warped VRTs for each land-type layer and compute the multiplier.

    When *aoi* is supplied the VRTs (and therefore all downstream
    processing) are clipped to the AOI bounding box.

    Args:
        layer_paths: Paths to raster layers defining land types.
        reference_raster: Raster whose pixel grid defines alignment.
        aoi: Optional area of interest for bbox clipping.

    Returns:
        Tuple of ``(warped_vrt_paths, multiplier, (gt, proj, xsize, ysize))``.
    """
    gdal.UseExceptions()

    ref_ds = gdal.Open(reference_raster)
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_xsize = ref_ds.RasterXSize
    ref_ysize = ref_ds.RasterYSize
    del ref_ds

    px = ref_gt[1]  # pixel width (positive)
    py = ref_gt[5]  # pixel height (negative)
    ref_minx = ref_gt[0]
    ref_maxy = ref_gt[3]
    ref_maxx = ref_minx + px * ref_xsize
    ref_miny = ref_maxy + py * ref_ysize

    if aoi is not None:
        aoi_bounds = _get_aoi_bounds(aoi)
        # Intersect AOI bbox with reference extent
        minx = max(aoi_bounds[0], ref_minx)
        miny = max(aoi_bounds[1], ref_miny)
        maxx = min(aoi_bounds[2], ref_maxx)
        maxy = min(aoi_bounds[3], ref_maxy)

        # Snap to reference pixel grid (expand outward)
        col_start = int(np.floor((minx - ref_minx) / px))
        col_end = int(np.ceil((maxx - ref_minx) / px))
        row_start = int(np.floor((ref_maxy - maxy) / (-py)))
        row_end = int(np.ceil((ref_maxy - miny) / (-py)))

        col_start = max(0, col_start)
        col_end = min(ref_xsize, col_end)
        row_start = max(0, row_start)
        row_end = min(ref_ysize, row_end)

        xsize = col_end - col_start
        ysize = row_end - row_start
        out_minx = ref_minx + col_start * px
        out_maxy = ref_maxy + row_start * py
        out_maxx = ref_minx + col_end * px
        out_miny = ref_maxy + row_end * py
        gt = (out_minx, px, 0.0, out_maxy, 0.0, py)
        bounds = (out_minx, out_miny, out_maxx, out_maxy)

        logger.info(
            "AOI clipping: %dx%d pixels (%.1f%% of reference %dx%d)",
            xsize,
            ysize,
            100.0 * xsize * ysize / (ref_xsize * ref_ysize),
            ref_xsize,
            ref_ysize,
        )
    else:
        xsize = ref_xsize
        ysize = ref_ysize
        gt = ref_gt
        bounds = (ref_minx, ref_miny, ref_maxx, ref_maxy)

    # Create warped VRTs
    warped_vrts: List[str] = []
    for lp in layer_paths:
        vrt_path = tempfile.NamedTemporaryFile(suffix="_lt_warp.vrt", delete=False).name
        _ds = gdal.Warp(
            vrt_path,
            str(Path(lp).resolve()),
            format="VRT",
            outputBounds=bounds,
            width=xsize,
            height=ysize,
            resampleAlg=gdal.GRA_NearestNeighbour,
            dstSRS=ref_proj,
        )
        _ds.FlushCache()
        del _ds
        warped_vrts.append(vrt_path)

    # Compute multiplier from the VRTs
    max_vals = []
    for vrt in warped_vrts:
        ds = gdal.Open(vrt)
        band = ds.GetRasterBand(1)
        stats = band.ComputeStatistics(False)
        max_vals.append(max(int(abs(stats[1])), 1))
        del ds
    multiplier = max(max_vals) + 1

    return warped_vrts, multiplier, (gt, ref_proj, xsize, ysize)


# Timeout for total processing (matches land_deg.py)
TOTAL_PROCESSING_TIMEOUT = 24 * 3600  # 24 hours in seconds


@dataclasses.dataclass()
class CounterbalancingTileInputs:
    """Parameters for processing a single counterbalancing tile.

    The tile is an (N+2)-band raster produced by CutTiles from a combined VRT:
      Band 1 = 7-class expanded status
      Bands 2..N+1 = individual land-type layers (warped VRTs)
      Band N+2 = AOI mask
    """

    in_file: Path
    n_land_type_bands: int
    multiplier: int
    layer_nodata: List[Optional[int]]


def _summarize_counterbalancing_tile(
    inputs: CounterbalancingTileInputs,
) -> Tuple[
    Optional[models.SummaryTableCounterbalancing],
    Optional[str],
    Optional[str],
    Optional[Dict[int, str]],
    Optional[str],
]:
    """Process a single tile for the counterbalancing assessment.

    Returns:
        Tuple of (summary_table, gl_out_path, lt_out_path, spu_labels,
        error_message).  On error, the first four elements are None.
    """
    gdal.UseExceptions()
    error_message = None
    tile_name = inputs.in_file.name
    logger.info("Processing counterbalancing tile: %s", tile_name)

    tile_ds = gdal.Open(str(inputs.in_file))
    if tile_ds is None:
        return None, None, None, None, f"Cannot open tile {tile_name}."

    n_lt = inputs.n_land_type_bands
    multiplier = inputs.multiplier

    status_band = tile_ds.GetRasterBand(1)
    lt_bands = [tile_ds.GetRasterBand(2 + i) for i in range(n_lt)]
    mask_band = tile_ds.GetRasterBand(2 + n_lt)

    xsize = tile_ds.RasterXSize
    ysize = tile_ds.RasterYSize
    src_gt = tile_ds.GetGeoTransform()
    proj = tile_ds.GetProjection()
    long_width = src_gt[1]
    lat = src_gt[3]
    pixel_height = src_gt[5]

    block_ysize = 256

    # Pre-compute cell areas per row
    all_cell_areas = np.empty(ysize, dtype=np.float64)
    current_lat = lat
    for row in range(ysize):
        all_cell_areas[row] = (
            calc_cell_area(current_lat, current_lat + pixel_height, long_width) * 1e-6
        )
        current_lat += pixel_height

    drv = gdal.GetDriverByName("GTiff")
    tif_opts = ["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"]

    # Output gains/losses tile (always GeoTIFF regardless of input format)
    gl_out_file = str(inputs.in_file.parent / (inputs.in_file.stem + "_cb_gl.tif"))
    gl_ds = drv.Create(gl_out_file, xsize, ysize, 1, gdal.GDT_Int16, options=tif_opts)
    gl_ds.SetGeoTransform(src_gt)
    gl_ds.SetProjection(proj)
    gl_band = gl_ds.GetRasterBand(1)
    gl_band.SetNoDataValue(float(config.NODATA_VALUE))

    # Output encoded land-type tile (always GeoTIFF)
    lt_out_file = str(inputs.in_file.parent / (inputs.in_file.stem + "_cb_lt.tif"))
    lt_ds = drv.Create(lt_out_file, xsize, ysize, 1, gdal.GDT_Int32, options=tif_opts)
    lt_ds.SetGeoTransform(src_gt)
    lt_ds.SetProjection(proj)
    lt_out_band = lt_ds.GetRasterBand(1)
    lt_out_band.SetNoDataValue(float(config.NODATA_VALUE))

    all_gains: Dict[int, float] = {}
    all_losses: Dict[int, float] = {}
    all_spu_labels: Dict[int, str] = {}

    for y in range(0, ysize, block_ysize):
        win_y = min(block_ysize, ysize - y)

        status_arr = status_band.ReadAsArray(0, y, xsize, win_y).astype(np.int16)
        mask_arr = mask_band.ReadAsArray(0, y, xsize, win_y)
        mask_bool = mask_arr == config.MASK_VALUE

        # Encode land-type layers
        combined = np.zeros((win_y, xsize), dtype=np.int64)
        nodata_mask = np.zeros((win_y, xsize), dtype=bool)

        for idx in range(n_lt):
            arr = lt_bands[idx].ReadAsArray(0, y, xsize, win_y).astype(np.int64)
            nd = inputs.layer_nodata[idx]
            if nd is not None:
                nodata_mask |= arr == nd
            power = n_lt - 1 - idx
            combined += arr * (multiplier**power)

        combined[nodata_mask] = config.NODATA_VALUE

        # Collect unique labels for land types seen in this block
        for u in np.unique(combined):
            u_int = int(u)
            if u_int == config.NODATA_VALUE:
                continue
            if u_int not in all_spu_labels:
                parts = []
                remainder = u_int
                for i in range(n_lt):
                    power = n_lt - 1 - i
                    divisor = multiplier**power
                    parts.append(str(remainder // divisor))
                    remainder = remainder % divisor
                all_spu_labels[u_int] = "_".join(parts)

        lt_arr = np.clip(combined, -2147483647, 2147483647).astype(np.int32)
        lt_out_band.WriteArray(lt_arr, 0, y)

        # Also mask pixels where any land-type layer is nodata
        mask_bool |= nodata_mask

        cell_areas = all_cell_areas[y : y + win_y].copy().reshape(-1, 1)
        cell_area_2d = np.broadcast_to(cell_areas, (win_y, xsize)).copy()

        gl_arr = classify_gains_losses(status_arr, mask_bool)
        gl_band.WriteArray(gl_arr, 0, y)

        # Zonal accumulation (Step 4)
        gains, losses = zonal_gains_losses(status_arr, lt_arr, cell_area_2d, mask_bool)
        all_gains = util.accumulate_dicts([all_gains, dict(gains)])
        all_losses = util.accumulate_dicts([all_losses, dict(losses)])

    del gl_ds, lt_ds, tile_ds

    summary_table = models.SummaryTableCounterbalancing(
        gains_by_land_type=all_gains,
        losses_by_land_type=all_losses,
    )
    summary_table.cast_to_cpython()  # needed for multiprocessing pickling

    logger.info("Completed counterbalancing tile: %s", tile_name)
    return summary_table, gl_out_file, lt_out_file, all_spu_labels, error_message


def _cb_process_multiprocess(
    inputs, n_cpus, parallel_backend="process", progress_callback=None
):
    """Dispatch counterbalancing tiles to a multiprocessing pool."""
    current_thread = threading.current_thread()
    is_in_thread_pool = getattr(
        current_thread, "_is_in_thread_pool", False
    ) or current_thread.name.startswith("ThreadPoolExecutor")

    chunksize = max(1, len(inputs) // (n_cpus * 2))
    total_tiles = len(inputs)
    worker_type = "threads" if parallel_backend == "thread" else "processes"

    logger.info(
        f"Processing {total_tiles} counterbalancing tiles with {n_cpus} {worker_type}"
    )

    if parallel_backend == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_cpus) as executor:
            summary_tables = []
            gl_out_files = []
            lt_out_files = []
            all_spu_labels: Dict[int, str] = {}

            for n, output in enumerate(
                executor.map(_summarize_counterbalancing_tile, inputs)
            ):
                summary_tbl, gl_file, lt_file, spu_labels, error_msg = output
                if error_msg is not None:
                    logger.error("Error %s", error_msg)
                    return None
                summary_tables.append(summary_tbl)
                gl_out_files.append(gl_file)
                lt_out_files.append(lt_file)
                all_spu_labels.update(spu_labels)

                completed = n + 1
                if (
                    completed % max(5, total_tiles // 20) == 0
                ) or completed == total_tiles:
                    util.log_progress(
                        completed / total_tiles,
                        message=(
                            f"Counterbalancing: {completed}/{total_tiles} "
                            f"tiles ({100 * completed / total_tiles:.1f}%)"
                        ),
                    )
                    if progress_callback is not None:
                        progress_callback(10 + int(80 * completed / total_tiles))

        summary_table = models.accumulate_summary_table_counterbalancing(summary_tables)
        return summary_table, gl_out_files, lt_out_files, all_spu_labels

    with multiprocessing.get_context("spawn").Pool(n_cpus) as pool:
        summary_tables = []
        gl_out_files = []
        lt_out_files = []
        all_spu_labels: Dict[int, str] = {}

        try:
            if is_in_thread_pool:
                results_iter = pool.imap_unordered(
                    _summarize_counterbalancing_tile,
                    inputs,
                    chunksize=chunksize,
                )
            else:
                async_result = pool.map_async(
                    _summarize_counterbalancing_tile,
                    inputs,
                    chunksize=chunksize,
                )
                results_iter = async_result.get(timeout=TOTAL_PROCESSING_TIMEOUT)

            for n, output in enumerate(results_iter):
                summary_tbl, gl_file, lt_file, spu_labels, error_msg = output
                if error_msg is not None:
                    logger.error("Error %s", error_msg)
                    pool.terminate()
                    return None
                summary_tables.append(summary_tbl)
                gl_out_files.append(gl_file)
                lt_out_files.append(lt_file)
                all_spu_labels.update(spu_labels)

                completed = n + 1
                if (
                    completed % max(5, total_tiles // 20) == 0
                ) or completed == total_tiles:
                    util.log_progress(
                        completed / total_tiles,
                        message=(
                            f"Counterbalancing: {completed}/{total_tiles} "
                            f"tiles ({100 * completed / total_tiles:.1f}%)"
                        ),
                    )
                    if progress_callback is not None:
                        progress_callback(10 + int(80 * completed / total_tiles))

        except multiprocessing.TimeoutError:
            logger.error(
                "Counterbalancing timed out after "
                f"{TOTAL_PROCESSING_TIMEOUT // 3600} hours"
            )
            pool.terminate()
            pool.join()
            return None
        except Exception as e:
            logger.error(f"Error in counterbalancing pool: {e}")
            pool.terminate()
            pool.join()
            return None

    summary_table = models.accumulate_summary_table_counterbalancing(summary_tables)
    return summary_table, gl_out_files, lt_out_files, all_spu_labels


def _cb_process_sequential(inputs, progress_callback=None, killed_callback=None):
    """Process counterbalancing tiles sequentially."""
    summary_tables = []
    gl_out_files = []
    lt_out_files = []
    all_spu_labels: Dict[int, str] = {}
    total_tiles = len(inputs)
    logger.info(f"Processing {total_tiles} counterbalancing tiles sequentially")

    for n, item in enumerate(inputs):
        if killed_callback is not None and killed_callback():
            logger.warning("Counterbalancing cancelled by user")
            return None

        output = _summarize_counterbalancing_tile(item)
        summary_tbl, gl_file, lt_file, spu_labels, error_msg = output

        if error_msg is not None:
            logger.error("Error %s", error_msg)
            break

        summary_tables.append(summary_tbl)
        gl_out_files.append(gl_file)
        lt_out_files.append(lt_file)
        all_spu_labels.update(spu_labels)

        completed = n + 1
        frac = completed / total_tiles
        util.log_progress(
            frac,
            message=(
                f"Counterbalancing: {completed}/{total_tiles} tiles ({100 * frac:.1f}%)"
            ),
        )
        if progress_callback is not None:
            # Map tile progress to 10-90% of overall progress
            progress_callback(10 + 80 * frac)

    summary_table = models.accumulate_summary_table_counterbalancing(summary_tables)
    return summary_table, gl_out_files, lt_out_files, all_spu_labels


def compute_counterbalancing(
    status_path: str,
    status_band_index: int,
    land_type_layer_paths: List[str],
    output_path: str,
    aoi: "AOI",
    n_cpus: int = max(1, multiprocessing.cpu_count() - 1),
    progress_callback=None,
    killed_callback=None,
    parallel_backend: str = "process",
) -> Tuple[
    models.SummaryTableCounterbalancing,
    List[models.CounterbalancingLandTypeResult],
    str,
    str,
    str,
    Dict[int, str],
]:
    """Run the full counterbalancing assessment.

    Args:
        status_path: Path to the 7-class expanded status raster.
        status_band_index: 1-based band index for the status layer.
        land_type_layer_paths: Paths to one or more raster layers whose
            intersection defines the land types for counterbalancing.
        output_path: Base path for output rasters (gains/losses and land type
            achievement rasters will be written alongside).
        aoi: Area of interest (used for masking and optional bbox clipping).
        n_cpus: Number of CPU cores for parallel tile processing.
            Defaults to ``cpu_count() - 1``.

    Returns:
        Tuple of:
          - SummaryTableCounterbalancing (accumulated stats)
          - List of CounterbalancingLandTypeResult (per-land-type results)
          - Path to gains/losses raster
          - Path to land type achievement raster
          - Path to spatial units (land type) raster
          - Dict mapping spatial unit code to label string
    """
    gdal.UseExceptions()

    logger.info(
        "Starting counterbalancing assessment with %d land type layer(s)",
        len(land_type_layer_paths),
    )

    # Adjust CPU count based on available memory (same heuristic as land_deg)
    try:
        import psutil

        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        memory_per_core_gb = 1.0
        max_cpus_by_memory = max(1, int(available_memory_gb / memory_per_core_gb))
        effective_n_cpus = min(n_cpus, max_cpus_by_memory, multiprocessing.cpu_count())
        logger.info(
            f"Counterbalancing using {effective_n_cpus} CPUs "
            f"(requested: {n_cpus}, memory-limited: {max_cpus_by_memory})"
        )
    except ImportError:
        effective_n_cpus = n_cpus

    output_base = Path(output_path)

    logger.info("Preparing land-type VRTs from %d layer(s)", len(land_type_layer_paths))
    warped_vrts, multiplier, (proc_gt, proc_proj, proc_xsize, proc_ysize) = (
        prepare_land_type_vrts(land_type_layer_paths, status_path, aoi)
    )

    # Collect per-layer nodata values
    layer_nodata: List[Optional[int]] = []
    for vrt in warped_vrts:
        ds = gdal.Open(vrt)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        layer_nodata.append(int(nd) if nd is not None else None)
        del ds

    logger.info(
        "Land-type VRTs ready (multiplier=%d, %d layers, extent=%dx%d)",
        multiplier,
        len(warped_vrts),
        proc_xsize,
        proc_ysize,
    )

    logger.info(
        "Extracting status band %d as VRT at processing extent", status_band_index
    )
    status_extracted = str(
        output_base.parent / f"{output_base.stem}_status_b{status_band_index}.vrt"
    )
    proc_bounds = (
        proc_gt[0],
        proc_gt[3] + proc_gt[5] * proc_ysize,
        proc_gt[0] + proc_gt[1] * proc_xsize,
        proc_gt[3],
    )
    _ds = gdal.Translate(
        status_extracted,
        status_path,
        format="VRT",
        bandList=[status_band_index],
        projWin=[proc_bounds[0], proc_bounds[3], proc_bounds[2], proc_bounds[1]],
    )
    _ds.FlushCache()
    del _ds

    # --- 3. Create mask at processing extent ---------------------------------
    logger.info("Creating AOI mask raster")
    mask_path = str(output_base.parent / f"{output_base.stem}_mask.tif")
    mask_worker = workers.Mask(
        out_file=Path(mask_path),
        geojson=aoi.geojson,
        model_file=Path(status_extracted),
    )
    if not mask_worker.work():
        raise RuntimeError("Failed to create AOI mask raster.")

    #  (N+2) bands: status | lt_layer_1 | ... | lt_layer_N | mask
    n_combined = 2 + len(warped_vrts)
    logger.info("Building %d-band combined VRT", n_combined)
    combined_vrt = str(output_base.parent / f"{output_base.stem}_cb_combined.vrt")
    _ds = gdal.BuildVRT(
        combined_vrt,
        [status_extracted] + warped_vrts + [mask_path],
        separate=True,
    )
    _ds.FlushCache()
    del _ds

    # Cut into tiles and process in parallel
    logger.info("Cutting input rasters into tiles")
    tile_base = output_base.parent / f"{output_base.stem}_cb_tiles.vrt"
    cutter = workers.CutTiles(
        in_file=Path(combined_vrt),
        n_cpus=effective_n_cpus,
        out_file=tile_base,
        datatype=gdal.GDT_Int32,
        parallel_backend=parallel_backend,
        output_format="VRT",
    )
    tiles = cutter.work()

    if not tiles:
        raise RuntimeError("Error splitting counterbalancing inputs into tiles.")

    logger.info(f"Created {len(tiles)} tiles for counterbalancing processing")

    inputs = [
        CounterbalancingTileInputs(
            in_file=tile,
            n_land_type_bands=len(warped_vrts),
            multiplier=multiplier,
            layer_nodata=layer_nodata,
        )
        for tile in tiles
    ]

    if progress_callback is not None:
        progress_callback(10)

    if killed_callback is not None and killed_callback():
        raise RuntimeError("Cancelled by user.")

    if effective_n_cpus > 1 and len(tiles) > 1:
        result = _cb_process_multiprocess(
            inputs, effective_n_cpus, parallel_backend, progress_callback
        )
    else:
        result = _cb_process_sequential(inputs, progress_callback, killed_callback)

    if result is None:
        raise RuntimeError("Error during counterbalancing tile processing.")

    summary_table, gl_tile_paths, lt_tile_paths, land_type_labels = result
    logger.info("Tile processing complete")

    # Combine output tiles into VRTs
    gains_losses_path = str(output_base.parent / f"{output_base.stem}_gains_losses.vrt")
    gdal.BuildVRT(gains_losses_path, gl_tile_paths)

    lt_raster_path = str(output_base.parent / f"{output_base.stem}_land_type.vrt")
    gdal.BuildVRT(lt_raster_path, lt_tile_paths)

    if progress_callback is not None:
        progress_callback(92)

    # Compute per-land-type results (Steps 4-5 of GPG)
    logger.info("Computing per-land-type results")
    all_gains = summary_table.gains_by_land_type
    all_losses = summary_table.losses_by_land_type

    all_lt_codes = sorted(set(list(all_gains.keys()) + list(all_losses.keys())))

    land_type_results: List[models.CounterbalancingLandTypeResult] = []

    for lt_code in all_lt_codes:
        g = all_gains.get(lt_code, 0.0)
        lo = all_losses.get(lt_code, 0.0)
        delta = g - lo
        achieved = delta >= 0
        total = g + lo
        pct = (delta / total * 100.0) if total > 0 else 0.0

        land_type_results.append(
            models.CounterbalancingLandTypeResult(
                land_type_code=int(lt_code),
                land_type_name=land_type_labels.get(lt_code, f"Land type {lt_code}"),
                gains_area_sqkm=float(g),
                losses_area_sqkm=float(lo),
                delta_ldn=float(delta),
                ldn_achieved=bool(achieved),
                ldn_pct=float(pct),
            )
        )

    if progress_callback is not None:
        progress_callback(95)

    # Write land type achievement raster
    logger.info("Writing land type achievement raster")
    achievement_path = str(
        output_base.parent / f"{output_base.stem}_lt_achievement.tif"
    )
    # Store percentage * 100 as Int16 (range -10000 to +10000, 0.01% precision)
    lt_achievement_pct_map = {
        r.land_type_code: int(round(r.ldn_pct * 100)) for r in land_type_results
    }

    lt_ds = gdal.Open(lt_raster_path)
    lt_band_r = lt_ds.GetRasterBand(1)
    mask_ds = gdal.Open(mask_path)
    mask_band_r = mask_ds.GetRasterBand(1)

    drv = gdal.GetDriverByName("GTiff")
    ach_ds = drv.Create(
        achievement_path,
        proc_xsize,
        proc_ysize,
        1,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    ach_ds.SetGeoTransform(proc_gt)
    ach_ds.SetProjection(proc_proj)
    ach_band = ach_ds.GetRasterBand(1)
    ach_band.SetNoDataValue(float(config.NODATA_VALUE))
    ach_band.SetScale(0.01)
    ach_band.SetOffset(0.0)

    block_ysize = 256
    for y_off in range(0, proc_ysize, block_ysize):
        win_y = min(block_ysize, proc_ysize - y_off)
        lt_arr = lt_band_r.ReadAsArray(0, y_off, proc_xsize, win_y).astype(np.int32)
        m_arr = mask_band_r.ReadAsArray(0, y_off, proc_xsize, win_y)
        ach_arr = np.full((win_y, proc_xsize), config.NODATA_VALUE, dtype=np.int16)

        for lt_code, scaled_val in lt_achievement_pct_map.items():
            ach_arr[lt_arr == lt_code] = np.int16(scaled_val)

        # Mask pixels outside the AOI
        ach_arr[m_arr == config.MASK_VALUE] = config.NODATA_VALUE

        ach_band.WriteArray(ach_arr, 0, y_off)

    del lt_ds, mask_ds, ach_ds

    logger.info(
        "Counterbalancing assessment complete - %d land types processed",
        len(land_type_results),
    )

    return (
        summary_table,
        land_type_results,
        gains_losses_path,
        achievement_path,
        lt_raster_path,
        land_type_labels,
    )
