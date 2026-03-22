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


def derive_land_type_raster(
    layer_paths: List[str],
    output_path: str,
    reference_raster: str,
) -> Dict[int, str]:
    """Intersect multiple raster layers to produce a single land type ID raster.

    Each unique combination of values across the input layers gets a unique
    integer ID.  Uses a multiplicative encoding: for N layers the land type ID
    is computed as ``v0 * M^(N-1) + v1 * M^(N-2) + ... + v(N-1)`` where M is
    the maximum value across all layers (clamped to a safe range).

    Args:
        layer_paths: Paths to raster layers that define land types.
        output_path: Where to write the output int32 land type raster.
        reference_raster: A raster whose extent, resolution, and projection
            define the output grid.

    Returns:
        Mapping of land type integer ID to a descriptive label string
        (``"v0_v1_..."``) for each unique combination found.
    """
    gdal.UseExceptions()

    ref_ds = gdal.Open(reference_raster)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize
    del ref_ds

    # Warp every input layer to the reference grid
    warped = []
    for lp in layer_paths:
        tmp = tempfile.NamedTemporaryFile(suffix="_lt_warp.tif", delete=False)
        gdal.Warp(
            tmp.name,
            lp,
            outputBounds=(
                gt[0],
                gt[3] + gt[5] * ysize,
                gt[0] + gt[1] * xsize,
                gt[3],
            ),
            width=xsize,
            height=ysize,
            resampleAlg=gdal.GRA_NearestNeighbour,
            dstSRS=proj,
        )
        warped.append(tmp.name)

    # Determine multiplier
    max_vals = []
    for wp in warped:
        ds = gdal.Open(wp)
        band = ds.GetRasterBand(1)
        stats = band.ComputeStatistics(False)
        max_vals.append(max(int(abs(stats[1])), 1))
        del ds
    multiplier = max(max_vals) + 1

    # Create output
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        output_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Int32,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(float(config.NODATA_VALUE))

    spu_labels: Dict[int, str] = {}
    block_ysize = 256

    datasets = [gdal.Open(wp) for wp in warped]

    for y in range(0, ysize, block_ysize):
        win_y = min(block_ysize, ysize - y)
        combined = np.zeros((win_y, xsize), dtype=np.int64)

        for idx, ds in enumerate(datasets):
            arr = ds.GetRasterBand(1).ReadAsArray(0, y, xsize, win_y).astype(np.int64)
            power = len(datasets) - 1 - idx
            combined += arr * (multiplier**power)

        # Collect unique labels
        uniques = np.unique(combined)
        for u in uniques:
            u_int = int(u)
            if u_int not in spu_labels:
                parts = []
                remainder = u_int
                for idx in range(len(datasets)):
                    power = len(datasets) - 1 - idx
                    divisor = multiplier**power
                    parts.append(str(remainder // divisor))
                    remainder = remainder % divisor
                spu_labels[u_int] = "_".join(parts)

        # Clamp to int32 range
        combined = np.clip(combined, -2147483647, 2147483647).astype(np.int32)
        dst_band.WriteArray(combined, 0, y)

    del datasets, dst_ds
    return spu_labels


# Timeout for total processing (matches land_deg.py)
TOTAL_PROCESSING_TIMEOUT = 24 * 3600  # 24 hours in seconds


@dataclasses.dataclass()
class CounterbalancingTileInputs:
    """Parameters for processing a single counterbalancing tile.

    The tile is a 3-band raster produced by CutTiles from a combined VRT:
      Band 1 = 7-class expanded status
      Band 2 = land type (from intersection of input layers)
      Band 3 = AOI mask
    """

    in_file: Path


def _summarize_counterbalancing_tile(
    inputs: CounterbalancingTileInputs,
) -> Tuple[Optional[models.SummaryTableCounterbalancing], Optional[str], Optional[str]]:
    """Process a single tile for the counterbalancing assessment.

    Returns:
        Tuple of (summary_table, out_file_path, error_message).
        On error, summary_table is None and error_message is set.
    """
    gdal.UseExceptions()
    error_message = None
    tile_name = inputs.in_file.name
    logger.info("Processing counterbalancing tile: %s", tile_name)

    tile_ds = gdal.Open(str(inputs.in_file))
    if tile_ds is None:
        return None, None, f"Cannot open tile {tile_name}."

    status_band = tile_ds.GetRasterBand(1)
    lt_band = tile_ds.GetRasterBand(2)
    mask_band = tile_ds.GetRasterBand(3)

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

    # Output gains/losses tile
    out_file = str(
        inputs.in_file.parent / (inputs.in_file.stem + "_cb_gl" + inputs.in_file.suffix)
    )
    drv = gdal.GetDriverByName("GTiff")
    gl_ds = drv.Create(
        out_file,
        xsize,
        ysize,
        1,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    gl_ds.SetGeoTransform(src_gt)
    gl_ds.SetProjection(proj)
    gl_band = gl_ds.GetRasterBand(1)
    gl_band.SetNoDataValue(float(config.NODATA_VALUE))

    all_gains: Dict[int, float] = {}
    all_losses: Dict[int, float] = {}

    for y in range(0, ysize, block_ysize):
        win_y = min(block_ysize, ysize - y)

        status_arr = status_band.ReadAsArray(0, y, xsize, win_y).astype(np.int16)
        lt_arr = lt_band.ReadAsArray(0, y, xsize, win_y).astype(np.int32)
        mask_arr = mask_band.ReadAsArray(0, y, xsize, win_y)
        mask_bool = mask_arr == config.MASK_VALUE

        cell_areas = all_cell_areas[y : y + win_y].copy().reshape(-1, 1)
        cell_area_2d = np.broadcast_to(cell_areas, (win_y, xsize)).copy()

        gl_arr = classify_gains_losses(status_arr, mask_bool)
        gl_band.WriteArray(gl_arr, 0, y)

        # Zonal accumulation (Step 4)
        gains, losses = zonal_gains_losses(status_arr, lt_arr, cell_area_2d, mask_bool)
        all_gains = util.accumulate_dicts([all_gains, dict(gains)])
        all_losses = util.accumulate_dicts([all_losses, dict(losses)])

    del gl_ds, tile_ds

    summary_table = models.SummaryTableCounterbalancing(
        gains_by_land_type=all_gains,
        losses_by_land_type=all_losses,
    )
    summary_table.cast_to_cpython()  # needed for multiprocessing pickling

    logger.info("Completed counterbalancing tile: %s", tile_name)
    return summary_table, out_file, error_message


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
            out_files = []

            for n, output in enumerate(
                executor.map(_summarize_counterbalancing_tile, inputs)
            ):
                summary_tbl, out_file, error_msg = output
                if error_msg is not None:
                    logger.error("Error %s", error_msg)
                    return None
                summary_tables.append(summary_tbl)
                out_files.append(out_file)

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
        return summary_table, out_files

    with multiprocessing.get_context("spawn").Pool(n_cpus) as pool:
        summary_tables = []
        out_files = []

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
                summary_tbl, out_file, error_msg = output
                if error_msg is not None:
                    logger.error("Error %s", error_msg)
                    pool.terminate()
                    return None
                summary_tables.append(summary_tbl)
                out_files.append(out_file)

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
    return summary_table, out_files


def _cb_process_sequential(inputs, progress_callback=None, killed_callback=None):
    """Process counterbalancing tiles sequentially."""
    summary_tables = []
    out_files = []
    total_tiles = len(inputs)
    logger.info(f"Processing {total_tiles} counterbalancing tiles sequentially")

    for n, item in enumerate(inputs):
        if killed_callback is not None and killed_callback():
            logger.warning("Counterbalancing cancelled by user")
            return None

        output = _summarize_counterbalancing_tile(item)
        summary_tbl, out_file, error_msg = output

        if error_msg is not None:
            logger.error("Error %s", error_msg)
            break

        summary_tables.append(summary_tbl)
        out_files.append(out_file)

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
    return summary_table, out_files


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
        aoi: Area of interest (used for masking).
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
        memory_per_core_gb = 4.0
        max_cpus_by_memory = max(1, int(available_memory_gb / memory_per_core_gb))
        effective_n_cpus = min(n_cpus, max_cpus_by_memory, multiprocessing.cpu_count())
        logger.info(
            f"Counterbalancing using {effective_n_cpus} CPUs "
            f"(requested: {n_cpus}, memory-limited: {max_cpus_by_memory})"
        )
    except ImportError:
        effective_n_cpus = n_cpus

    output_base = Path(output_path)

    # --- 1. Derive land type raster from input layers ------------------------
    logger.info("Deriving land type raster from input layers")
    lt_raster_path = str(output_base.parent / f"{output_base.stem}_land_type.tif")
    land_type_labels = derive_land_type_raster(
        land_type_layer_paths, lt_raster_path, status_path
    )
    logger.info(
        "Land type raster complete - %d unique land types found",
        len(land_type_labels),
    )

    # --- 2. Create mask from AOI ---------------------------------------------
    logger.info("Creating AOI mask raster")
    mask_path = str(output_base.parent / f"{output_base.stem}_mask.tif")
    mask_worker = workers.Mask(
        out_file=Path(mask_path),
        geojson=aoi.geojson,
        model_file=Path(status_path),
    )
    if not mask_worker.work():
        raise RuntimeError("Failed to create AOI mask raster.")

    # --- 3. Extract status band and build combined VRT -----------------------
    logger.info(
        "Extracting status band %d and building combined VRT", status_band_index
    )
    #  Build a 3-band VRT: status | land_type | mask
    #  so CutTiles can split all inputs into aligned tile files.
    status_extracted = str(
        output_base.parent / f"{output_base.stem}_status_b{status_band_index}.tif"
    )
    gdal.Translate(
        status_extracted,
        status_path,
        bandList=[status_band_index],
        creationOptions=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )

    combined_vrt = str(output_base.parent / f"{output_base.stem}_cb_combined.vrt")
    gdal.BuildVRT(
        combined_vrt,
        [status_extracted, lt_raster_path, mask_path],
        separate=True,
    )

    # --- 4. Cut into tiles and process in parallel ---------------------------
    logger.info("Cutting input rasters into tiles")
    tile_base = output_base.parent / f"{output_base.stem}_cb_tiles.tif"
    cutter = workers.CutTiles(
        in_file=Path(combined_vrt),
        n_cpus=effective_n_cpus,
        out_file=tile_base,
        datatype=gdal.GDT_Int32,
        parallel_backend=parallel_backend,
    )
    tiles = cutter.work()

    if not tiles:
        raise RuntimeError("Error splitting counterbalancing inputs into tiles.")

    logger.info(f"Created {len(tiles)} tiles for counterbalancing processing")

    inputs = [CounterbalancingTileInputs(in_file=tile) for tile in tiles]

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

    summary_table, gl_tile_paths = result
    logger.info("Tile processing complete")

    # --- 5. Combine gains/losses tiles into a VRT ----------------------------
    gains_losses_path = str(output_base.parent / f"{output_base.stem}_gains_losses.vrt")
    gdal.BuildVRT(gains_losses_path, gl_tile_paths)

    if progress_callback is not None:
        progress_callback(92)

    # --- 6. Compute per-land-type results (Steps 4-5 of GPG) ----------------
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

    # --- 7. Write land type achievement raster --------------------------------
    logger.info("Writing land type achievement raster")
    achievement_path = str(
        output_base.parent / f"{output_base.stem}_lt_achievement.tif"
    )
    # Store percentage * 100 as Int16 (range -10000 to +10000, 0.01% precision)
    lt_achievement_pct_map = {
        r.land_type_code: int(round(r.ldn_pct * 100)) for r in land_type_results
    }

    ref_ds = gdal.Open(status_path)
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize
    del ref_ds

    lt_ds = gdal.Open(lt_raster_path)
    lt_band_r = lt_ds.GetRasterBand(1)
    mask_ds = gdal.Open(mask_path)
    mask_band_r = mask_ds.GetRasterBand(1)

    drv = gdal.GetDriverByName("GTiff")
    ach_ds = drv.Create(
        achievement_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
    )
    ach_ds.SetGeoTransform(ref_gt)
    ach_ds.SetProjection(ref_proj)
    ach_band = ach_ds.GetRasterBand(1)
    ach_band.SetNoDataValue(float(config.NODATA_VALUE))
    ach_band.SetScale(0.01)
    ach_band.SetOffset(0.0)

    block_ysize = 256
    for y_off in range(0, ysize, block_ysize):
        win_y = min(block_ysize, ysize - y_off)
        lt_arr = lt_band_r.ReadAsArray(0, y_off, xsize, win_y).astype(np.int32)
        m_arr = mask_band_r.ReadAsArray(0, y_off, xsize, win_y)
        ach_arr = np.full((win_y, xsize), config.NODATA_VALUE, dtype=np.int16)

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
