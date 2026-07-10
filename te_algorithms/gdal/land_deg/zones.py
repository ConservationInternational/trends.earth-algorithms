"""Analysis zones for LDN Planning.

A *zone* layer partitions space into mutually exclusive units used to break
down statistics (per-jurisdiction, per-land-type, per-management-unit, ...).

Two things live here:

  1. A reusable **Create Zones** algorithm (``create_zones``) that combines one
     or more categorical rasters into a single Int32 zone-id raster, mirroring
     the multi-raster "spatial unit" logic previously embedded in the LDN
     Counterbalancing tool. Each unique combination of input class values
     becomes one zone, remapped to a compact sequential id (1..K). A JSON
     "key" sidecar records the mapping ``{id: {label, values}}`` so the zones
     can be re-labelled and re-styled later without recomputation.

  2. Zone **resolution** helpers (``resolve_zone_ids``) that turn any supported
     zone source â€” a saved zones raster, an uploaded vector, or nothing â€” into
     a ``(zone_id_array, {id: name})`` pair aligned to an arbitrary analysis
     grid. This lets every analysis (BAU, scenario, counterbalancing) consume
     the *same* zones without caring how they were defined.

This module is pure GDAL/OGR/NumPy â€” no QGIS imports â€” so it can be reused by
the algorithms library and called from the plugin's local-execution handlers.

Memory safety
-------------
``create_zones`` is designed to handle large or global rasters safely:

  * **AOI clipping** (``aoi_bounds``): the output is cropped to the AOI before
    any array is allocated.  Always pass an AOI when running on datasets larger
    than the study area.
  * **Block-by-block processing**: the algorithm never allocates a full-raster
    array.  It uses two streaming passes over warped VRTs â€” one to discover all
    unique class-value combinations, one to write the remapped zone IDs â€” each
    reading at most ``ZONE_BLOCK_SIZE`` rows Ã— ``xsize`` columns at a time.
"""

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr

from . import config

logger = logging.getLogger(__name__)

gdal.UseExceptions()

NODATA = int(config.NODATA_VALUE)

# Rows processed per block in the streaming two-pass algorithm.
# 128 rows Ã— 172 800 cols (global 30â€³) Ã— 8 bytes (int64) â‰ˆ 176 MB per block
# per layer â€” acceptable even for global inputs.  Increase for faster I/O on
# small AOIs; decrease if memory is very constrained.
ZONE_BLOCK_SIZE = 128

# Vector attribute fields tried (in order) when deriving a human-readable
# zone name from an uploaded polygon layer.
_NAME_FIELD_CANDIDATES = (
    "name",
    "NAME",
    "shapeName",
    "NAME_1",
    "ADM1_NAME",
    "zone_name",
    "zone_id",
)


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def _ref_bounds(ref_gt: Tuple, xsize: int, ysize: int) -> Tuple:
    """(minx, miny, maxx, maxy) for a north-up geotransform."""
    minx = ref_gt[0]
    maxy = ref_gt[3]
    maxx = ref_gt[0] + ref_gt[1] * xsize
    miny = ref_gt[3] + ref_gt[5] * ysize
    return (minx, miny, maxx, maxy)


def align_raster_to_ref(
    src_path: str,
    band_index: int,
    ref_gt: Tuple,
    ref_proj: str,
    xsize: int,
    ysize: int,
) -> np.ndarray:
    """Read *src_path* band resampled (nearest) onto the reference grid.

    Returns an int32 array of shape (ysize, xsize). Nearest-neighbour keeps
    categorical class values intact.

    Note: this function loads the entire output into memory.  The reference
    grid should already be clipped to the analysis AOI (as the ARR raster
    always is).  Do **not** pass a global raster here without first clipping
    the reference grid â€” use ``create_zones`` with an ``aoi_bounds`` argument
    instead.
    """
    src_ds = gdal.Open(str(src_path))
    if src_ds is None:
        raise RuntimeError(f"Cannot open raster: {src_path}")
    same_grid = (
        src_ds.RasterXSize == xsize
        and src_ds.RasterYSize == ysize
        and src_ds.GetGeoTransform() == ref_gt
    )
    if same_grid:
        arr = src_ds.GetRasterBand(band_index).ReadAsArray().astype(np.int64)
        del src_ds
        return arr.astype(np.int32)

    warp_path = tempfile.NamedTemporaryFile(suffix="_zone_align.vrt", delete=False).name
    gdal.Warp(
        warp_path,
        str(src_path),
        format="VRT",
        width=xsize,
        height=ysize,
        outputBounds=_ref_bounds(ref_gt, xsize, ysize),
        dstSRS=ref_proj,
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcBands=[band_index],
    )
    warp_ds = gdal.Open(warp_path)
    arr = warp_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    del warp_ds, src_ds
    return arr


# ---------------------------------------------------------------------------
# Multi-raster zone combination (in-memory, for already-clipped grids)
# ---------------------------------------------------------------------------


def combine_arrays_to_zones(
    arrays: List[np.ndarray],
    nodata_values: List[int],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Combine aligned categorical arrays into sequential zone ids.

    Each unique combination of input values becomes one zone. Combinations are
    first encoded with a positional multiplier (``value_i * multiplier**power``)
    then remapped to compact sequential ids (1..K); id 0 marks nodata/outside.

    Args:
        arrays: list of equal-shape int arrays (already aligned to one grid).
        nodata_values: nodata value for each corresponding array.

    Returns:
        (zone_ids int32 array, labels {id: "v1_v2_..."}).
    """
    if not arrays:
        raise ValueError("combine_arrays_to_zones requires at least one array")

    shape = arrays[0].shape
    valid = np.ones(shape, dtype=bool)
    for arr, nd in zip(arrays, nodata_values):
        valid &= arr != int(nd)

    # Multiplier must exceed the largest class value in any layer.
    max_vals = []
    for arr in arrays:
        if np.any(valid):
            max_vals.append(int(np.max(np.abs(arr[valid]))))
        else:
            max_vals.append(0)
    multiplier = max(max(max_vals), 1) + 1

    n = len(arrays)
    combined = np.zeros(shape, dtype=np.int64)
    for idx, arr in enumerate(arrays):
        power = n - 1 - idx
        combined += arr.astype(np.int64) * (multiplier**power)

    zone_ids = np.zeros(shape, dtype=np.int32)
    labels: Dict[int, str] = {}
    unique_codes = np.unique(combined[valid]) if np.any(valid) else np.array([])
    for seq_id, raw in enumerate(sorted(int(c) for c in unique_codes), start=1):
        zone_ids[valid & (combined == raw)] = seq_id
        # Decode the raw code back into per-layer values for the label.
        parts = []
        remainder = raw
        for i in range(n):
            divisor = multiplier ** (n - 1 - i)
            parts.append(str(remainder // divisor))
            remainder = remainder % divisor
        labels[seq_id] = "_".join(parts)

    return zone_ids, labels


def combine_rasters_to_zones(
    raster_paths: List[str],
    ref_gt: Tuple,
    ref_proj: str,
    xsize: int,
    ysize: int,
    band_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[int, str], List[str]]:
    """Align and combine multiple rasters into a sequential zone-id array.

    Loads the full output into memory â€” only use when the reference grid is
    already clipped to the AOI (e.g. the ARR raster extent).

    Returns (zone_ids int32, labels {id: "v1_v2_.."}, layer_names).
    """
    if band_indices is None:
        band_indices = [1] * len(raster_paths)

    arrays = []
    nodata_values = []
    for path, bidx in zip(raster_paths, band_indices):
        arr = align_raster_to_ref(path, bidx, ref_gt, ref_proj, xsize, ysize)
        src_ds = gdal.Open(str(path))
        nd = src_ds.GetRasterBand(bidx).GetNoDataValue()
        del src_ds
        nodata_values.append(int(nd) if nd is not None else NODATA)
        arrays.append(arr)

    zone_ids, labels = combine_arrays_to_zones(arrays, nodata_values)
    layer_names = [Path(p).stem for p in raster_paths]
    return zone_ids, labels, layer_names


# ---------------------------------------------------------------------------
# Vector zones
# ---------------------------------------------------------------------------


def rasterize_vector_zones(
    zones_path: str,
    ref_gt: Tuple,
    ref_proj: str,
    xsize: int,
    ysize: int,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Rasterize polygon features to sequential zone ids on the reference grid.

    Returns (zone_ids int32 [1..N, 0 = outside], {id: feature_name}).
    """
    zones_ds = ogr.Open(zones_path, 0)
    if zones_ds is None:
        return np.zeros((ysize, xsize), dtype=np.int32), {}
    zones_lyr = zones_ds.GetLayer(0)

    mem_drv = gdal.GetDriverByName("MEM")
    zid_ds = mem_drv.Create("", xsize, ysize, 1, gdal.GDT_Int32)
    zid_ds.SetGeoTransform(ref_gt)
    zid_ds.SetProjection(ref_proj)
    zid_ds.GetRasterBand(1).Fill(0)

    id_to_name: Dict[int, str] = {}
    defn = zones_lyr.GetLayerDefn()
    name_field = None
    for candidate in _NAME_FIELD_CANDIDATES:
        if defn.GetFieldIndex(candidate) >= 0:
            name_field = candidate
            break

    zones_lyr.ResetReading()
    for i, feat in enumerate(zones_lyr, start=1):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        name = feat.GetField(name_field) if name_field else str(feat.GetFID())
        id_to_name[i] = name if name is not None else str(feat.GetFID())
        tmp_ds = ogr.GetDriverByName("Memory").CreateDataSource("z")
        tmp_lyr = tmp_ds.CreateLayer("z", geom_type=ogr.wkbPolygon)
        tmp_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
        tmp_feat.SetGeometry(geom)
        tmp_lyr.CreateFeature(tmp_feat)
        gdal.RasterizeLayer(zid_ds, [1], tmp_lyr, burn_values=[i])
        del tmp_ds

    arr = zid_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    del zid_ds, zones_ds
    return arr, id_to_name


# ---------------------------------------------------------------------------
# Saved zones raster
# ---------------------------------------------------------------------------


def load_zones_raster(
    zones_raster_path: str,
    ref_gt: Tuple,
    ref_proj: str,
    xsize: int,
    ysize: int,
    label_key: Optional[Dict[Any, str]] = None,
    band_index: int = 1,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Load a saved zones raster aligned to the reference grid.

    Args:
        zones_raster_path: Int32 zone-id raster produced by ``create_zones``.
        label_key: optional {id: label} mapping (keys may be str or int).

    Returns (zone_ids int32 [0 = outside], {id: name}).
    """
    arr = align_raster_to_ref(
        zones_raster_path, band_index, ref_gt, ref_proj, xsize, ysize
    )
    arr = arr.astype(np.int32)
    arr[arr < 0] = 0  # nodata -> outside

    id_to_name: Dict[int, str] = {}
    present = (int(c) for c in np.unique(arr) if int(c) != 0)
    for code in present:
        name = None
        if label_key:
            name = label_key.get(code, label_key.get(str(code)))
        id_to_name[code] = name if name is not None else str(code)
    return arr, id_to_name


def resolve_zone_ids(
    ref_gt: Tuple,
    ref_proj: str,
    xsize: int,
    ysize: int,
    zones_path: Optional[str] = None,
    zones_raster_path: Optional[str] = None,
    zones_raster_labels: Optional[Dict[Any, str]] = None,
    zones_raster_band_index: int = 1,
) -> Tuple[Optional[np.ndarray], Dict[int, str]]:
    """Resolve any supported zone source to (zone_id_array, {id: name}).

    Precedence: saved zones raster > uploaded vector > none. Returns
    ``(None, {})`` when no zone source is supplied.
    """
    if zones_raster_path:
        return load_zones_raster(
            zones_raster_path,
            ref_gt,
            ref_proj,
            xsize,
            ysize,
            label_key=zones_raster_labels,
            band_index=zones_raster_band_index,
        )
    if zones_path:
        return rasterize_vector_zones(zones_path, ref_gt, ref_proj, xsize, ysize)
    return None, {}


# ---------------------------------------------------------------------------
# Create Zones â€” memory-safe, streaming, AOI-aware public algorithm
# ---------------------------------------------------------------------------


def _approx_band_max(path: str, band_index: int) -> int:
    """Fast approximate maximum absolute class value for a raster band.

    Uses cached/overview statistics when available; falls back to an
    approximate scan (GDAL samples the raster rather than reading all pixels).
    Always returns at least 1.
    """
    ds = gdal.Open(str(path))
    if ds is None:
        return 1
    band = ds.GetRasterBand(band_index)
    # Try metadata-cached stats first (instant)
    stats = band.GetStatistics(True, False)  # approxOK=True, force=False
    max_val = abs(stats[1]) if stats and stats[1] != 0 else 0
    if max_val == 0:
        # Approximate computation â€” uses overviews/subsampling (fast)
        try:
            stats = band.ComputeStatistics(True)
            max_val = abs(stats[1])
        except RuntimeError:
            max_val = 0
    del ds
    return max(int(max_val), 1)


def create_zones(
    raster_paths: List[str],
    output_path: str,
    band_indices: Optional[List[int]] = None,
    aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
    progress_callback=None,
    killed_callback=None,
) -> Tuple[str, Dict[int, str], List[str]]:
    """Combine categorical rasters into a reusable Int32 zones raster.

    **Memory-safe and streaming** — works for any input size including
    Brazil at 30 m resolution.  The algorithm:

    1. Clips the output grid to ``aoi_bounds`` before any data is read
       (eliminates out-of-memory errors for global/continental inputs).
    2. Chooses an adaptive block height so each block occupies at most
       ~200 MB regardless of raster width or number of input layers.
    3. Processes the grid in a **single forward pass**:
       - For each row-block, each source raster is warped into a fresh
         in-memory (MEM) dataset via ``gdal.Warp``.  Each warp is a
         self-contained, stateless operation — no VRT file, no GDAL
         block-cache re-read, no Windows caching artefact.
       - The combined class-value encoding is computed in NumPy and zone
         IDs are assigned **incrementally** (new combinations get the next
         sequential ID as they are first encountered in the forward pass).
       - The zone block is written to the output GeoTIFF immediately.
       Because zone IDs are assigned in first-occurrence order rather than
       a global sort, the sidecar key is built at the end from the
       accumulated mapping.

    Args:
        raster_paths: categorical rasters to combine; the first defines the
            reference pixel grid.
        output_path: destination Int32 GeoTIFF.
        band_indices: band to read per raster (default band 1 for all).
        aoi_bounds: (minx, miny, maxx, maxy) in the rasters CRS.  Strongly
            recommended for inputs larger than the study area.
        progress_callback: optional callable(pct: float).
        killed_callback: optional callable() -> bool.

    Returns:
        (output_path, labels {id: label_str}, layer_names).
    """
    if not raster_paths:
        raise ValueError("create_zones requires at least one raster")
    if band_indices is None:
        band_indices = [1] * len(raster_paths)

    # ── Reference grid ──────────────────────────────────────────────────────
    ref_ds = gdal.Open(str(raster_paths[0]))
    if ref_ds is None:
        raise RuntimeError(f"Cannot open raster: {raster_paths[0]}")
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_xsize = ref_ds.RasterXSize
    ref_ysize = ref_ds.RasterYSize
    del ref_ds

    px = ref_gt[1]  # pixel width  (positive)
    py = ref_gt[5]  # pixel height (negative)
    ref_minx = ref_gt[0]
    ref_maxy = ref_gt[3]
    ref_maxx = ref_minx + px * ref_xsize
    ref_miny = ref_maxy + py * ref_ysize

    # ── AOI clipping ────────────────────────────────────────────────────────
    if aoi_bounds is not None:
        aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi_bounds
        clip_minx = max(aoi_minx, ref_minx)
        clip_miny = max(aoi_miny, ref_miny)
        clip_maxx = min(aoi_maxx, ref_maxx)
        clip_maxy = min(aoi_maxy, ref_maxy)
        col_start = max(0, int(math.floor((clip_minx - ref_minx) / px)))
        col_end = min(ref_xsize, int(math.ceil((clip_maxx - ref_minx) / px)))
        row_start = max(0, int(math.floor((ref_maxy - clip_maxy) / (-py))))
        row_end = min(ref_ysize, int(math.ceil((ref_maxy - clip_miny) / (-py))))
        xsize = max(1, col_end - col_start)
        ysize = max(1, row_end - row_start)
        out_minx = ref_minx + col_start * px
        out_maxy = ref_maxy + row_start * py
        out_gt = (out_minx, px, 0.0, out_maxy, 0.0, py)
        reduction = 1.0 - (xsize * ysize) / max(ref_xsize * ref_ysize, 1)
        logger.info(
            "Create Zones: AOI clip -> %dx%d px (%.0f%% reduction from %dx%d)",
            xsize,
            ysize,
            100.0 * reduction,
            ref_xsize,
            ref_ysize,
        )
    else:
        xsize, ysize = ref_xsize, ref_ysize
        out_gt = ref_gt
        total_mpx = xsize * ysize / 1_000_000
        if total_mpx > 50:
            logger.warning(
                "Create Zones: no AOI set; processing %dx%d (%.0f Mpx). "
                "Set an AOI to speed up processing and reduce memory use.",
                xsize,
                ysize,
                total_mpx,
            )

    if progress_callback:
        progress_callback(5.0)
    if killed_callback and killed_callback():
        return str(output_path), {}, []

    # ── Nodata values from source rasters ───────────────────────────────────
    n = len(raster_paths)
    nodata_values: List[int] = []
    for path, bidx in zip(raster_paths, band_indices):
        src_ds = gdal.Open(str(path))
        nd = src_ds.GetRasterBand(bidx).GetNoDataValue() if src_ds else None
        del src_ds
        nodata_values.append(int(nd) if nd is not None else NODATA)

    # ── Approximate multiplier from band statistics (fast) ──────────────────
    max_vals = [_approx_band_max(p, b) for p, b in zip(raster_paths, band_indices)]
    multiplier = max(max(max_vals), 1) + 1

    if progress_callback:
        progress_callback(10.0)

    # ── Adaptive block height ────────────────────────────────────────────────
    # Budget ~200 MB peak per block (n+1 int64 arrays + 1 int32 + 1 bool).
    bytes_per_pixel = (n + 1) * 8 + 4 + 1
    target_rows = max(1, int(200 * 1_000_000 / max(xsize * bytes_per_pixel, 1)))
    zone_block_rows = min(target_rows, ysize, 256)
    n_blocks = math.ceil(ysize / zone_block_rows)

    # ── Create output GeoTIFF ────────────────────────────────────────────────
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_path),
        xsize,
        ysize,
        1,
        gdal.GDT_Int32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(ref_proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    # ── Single forward pass ──────────────────────────────────────────────────
    # Zone IDs are assigned incrementally as new combinations are first
    # encountered.  Each source block is read exactly once via a fresh
    # gdal.Warp-to-MEM call, which is stateless and always returns correct
    # data regardless of GDAL block-cache state.
    raw_to_seq: Dict[int, int] = {}
    next_id = 1
    layer_names = [Path(p).stem for p in raster_paths]

    for blk_idx, y_off in enumerate(range(0, ysize, zone_block_rows)):
        if killed_callback and killed_callback():
            break
        y_blk = min(zone_block_rows, ysize - y_off)

        # Geographic extent of this row-block (outputBounds = xmin,ymin,xmax,ymax)
        blk_maxy = out_gt[3] + out_gt[5] * y_off
        blk_miny = out_gt[3] + out_gt[5] * (y_off + y_blk)
        blk_bounds = (out_gt[0], blk_miny, out_gt[0] + out_gt[1] * xsize, blk_maxy)

        combined = np.zeros((y_blk, xsize), dtype=np.int64)
        valid = np.ones((y_blk, xsize), dtype=bool)

        for idx, (path, bidx, nd) in enumerate(
            zip(raster_paths, band_indices, nodata_values)
        ):
            # Fresh warp to MEM for this block — stateless, no caching issues.
            mem_ds = gdal.Warp(
                "",
                str(path),
                format="MEM",
                width=xsize,
                height=y_blk,
                outputBounds=blk_bounds,
                dstSRS=ref_proj,
                resampleAlg=gdal.GRA_NearestNeighbour,
                srcBands=[bidx],
                dstNodata=nd,
            )
            if mem_ds is None:
                valid[:] = False
                continue
            arr = mem_ds.GetRasterBand(1).ReadAsArray()
            del mem_ds
            if arr is None:
                valid[:] = False
                continue
            arr = arr.astype(np.int64)
            valid &= arr != nd
            combined += arr * (multiplier ** (n - 1 - idx))

        # Assign sequential IDs to any new combinations in this block.
        if np.any(valid):
            for code in sorted(int(c) for c in np.unique(combined[valid])):
                if code not in raw_to_seq:
                    raw_to_seq[code] = next_id
                    next_id += 1

        # Write zone IDs for this block.
        zone_block = np.zeros((y_blk, xsize), dtype=np.int32)
        for raw, seq in raw_to_seq.items():
            zone_block[valid & (combined == raw)] = seq
        out_band.WriteArray(zone_block, 0, y_off)

        if progress_callback:
            progress_callback(10.0 + 85.0 * (blk_idx + 1) / n_blocks)

    out_band.FlushCache()
    out_ds.FlushCache()
    del out_band, out_ds

    # ── Build labels from accumulated mapping ────────────────────────────────
    labels: Dict[int, str] = {}
    for raw, seq in raw_to_seq.items():
        parts: List[str] = []
        remainder = raw
        for i in range(n):
            div = multiplier ** (n - 1 - i)
            parts.append(str(remainder // div))
            remainder = remainder % div
        labels[seq] = "_".join(parts)

    # ── Sidecar key ──────────────────────────────────────────────────────────
    key = {
        "description": "Maps each zone id to the combination of source-raster class values.",
        "layer_names": layer_names,
        "units": {
            str(zid): {"label": label, "values": label.split("_")}
            for zid, label in sorted(labels.items())
        },
    }
    key_path = str(output_path) + ".key.json"
    try:
        with open(key_path, "w", encoding="utf-8") as fh:
            json.dump(key, fh, indent=2)
    except OSError as exc:
        logger.warning("Could not write zones key sidecar %s: %s", key_path, exc)

    if progress_callback:
        progress_callback(100.0)

    logger.info(
        "Create Zones: %d zone(s) from %d raster(s) -> %s",
        len(labels),
        len(raster_paths),
        output_path,
    )
    return str(output_path), labels, layer_names


def read_zones_key(zones_raster_path: str) -> Optional[Dict[str, Any]]:
    """Load the ``<raster>.key.json`` sidecar if present."""
    key_path = str(zones_raster_path) + ".key.json"
    if not os.path.exists(key_path):
        return None
    try:
        with open(key_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read zones key %s: %s", key_path, exc)
        return None


def labels_from_key(key: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Extract an {id: label} mapping from a zones key dict."""
    if not key:
        return {}
    units = key.get("units", {})
    out: Dict[int, str] = {}
    for k, v in units.items():
        try:
            out[int(k)] = v.get("label", str(k)) if isinstance(v, dict) else str(v)
        except (TypeError, ValueError):
            continue
    return out
