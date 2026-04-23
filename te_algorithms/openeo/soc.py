"""openEO-based Soil Organic Carbon (SOC) algorithm for Trends.Earth.

This module mirrors the signature of :func:`te_algorithms.gee.soc.soc` as
closely as possible so that the dispatch layer in
``trends.earth/gee/soil-organic-carbon/src/main.py`` can call either
implementation with the same arguments.

The openEO implementation loads SoilGrids baseline SOC from Terrascope
(``TERRASCOPE_WORLD_SOILGRIDS``, band ``SOC``), applies the same IPCC
stock-change-factor approach as the GEE version, and writes GeoTIFF results
to S3 via openEO's ``save_result`` process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .util import bbox_from_geojsons, get_backend_normalizations, save_result_to_s3

if TYPE_CHECKING:
    import openeo  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPCC stock-change factor tables (identical values to gee/soc.py)
# ---------------------------------------------------------------------------

# fmt: off
# Transition codes follow the same convention as gee/soc.py:
# first digit = baseline IPCC class, second digit = final IPCC class
# Values: 1 = no change in factor
_SOC_FACTOR_MANAGEMENT = (
    [11, 12, 13, 14, 15, 16, 17,
     21, 22, 23, 24, 25, 26, 27,
     31, 32, 33, 34, 35, 36, 37,
     41, 42, 43, 44, 45, 46, 47,
     51, 52, 53, 54, 55, 56, 57,
     61, 62, 63, 64, 65, 66, 67,
     71, 72, 73, 74, 75, 76, 77],
    [1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1],
)

_SOC_FACTOR_ORGANIC_MATTER = (
    [11, 12, 13, 14, 15, 16, 17,
     21, 22, 23, 24, 25, 26, 27,
     31, 32, 33, 34, 35, 36, 37,
     41, 42, 43, 44, 45, 46, 47,
     51, 52, 53, 54, 55, 56, 57,
     61, 62, 63, 64, 65, 66, 67,
     71, 72, 73, 74, 75, 76, 77],
    [1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1],
)

_SOC_FACTOR_LAND_USE = (
    [11, 12, 13, 14, 15, 16, 17,
     21, 22, 23, 24, 25, 26, 27,
     31, 32, 33, 34, 35, 36, 37,
     41, 42, 43, 44, 45, 46, 47,
     51, 52, 53, 54, 55, 56, 57,
     61, 62, 63, 64, 65, 66, 67,
     71, 72, 73, 74, 75, 76, 77],
    [1,    1,   99,       1, 0.1,  0.1, 1,
     1,    1,   99,       1, 0.1,  0.1, 1,
     -99, -99,   1, 1/0.71, 0.1,  0.1, 1,
     1,    1, 0.71,       1, 0.1,  0.1, 1,
     2,    2,    2,       2, 1,    1,   1,
     2,    2,    2,       2, 1,    1,   1,
     1,    1,    1,       1, 1,    1,   1],
)
# fmt: on

# IPCC climate zone remapping tables (matching GEE toolbox_datasets/ipcc_climate_zones)
# Step 1: raw zone codes 0–12 → IPCC climate index 0–5
_CLIMATE_ZONE_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_CLIMATE_ZONE_TO_IPCC_IDX = [0, 2, 1, 2, 1, 2, 1, 2, 1, 5, 4, 4, 3]
# Step 2: IPCC climate index → Fl value
# Indices: 0=nodata, 1=tropical dry, 2=tropical moist, 3=temperate cool,
#          4=boreal, 5=tropical montane
_IPCC_IDX_FL_VALUES = [0.0, 0.80, 0.69, 0.58, 0.48, 0.64]


def soc(
    year_initial: int,
    year_final: int,
    fl: Any,  # float or "per pixel"
    esa_to_custom_nesting: Any,
    ipcc_nesting: Any,
    annual_lc: bool,
    annual_soc: bool,
    logger: Any,
    fake_data: bool = False,
    # openEO-specific arguments
    connection: "openeo.Connection | None" = None,
    geojsons: list | None = None,
    execution_id: str | None = None,
):
    """Build and submit an openEO batch job to compute the SOC indicator.

    Parameters mirror :func:`te_algorithms.gee.soc.soc` for the shared
    arguments.  The openEO-specific parameters are:

    Args:
        connection: An authenticated :class:`openeo.Connection`.  Required
            when running on openEO.
        geojsons: List of GeoJSON dicts defining the area of interest.
        execution_id: Trends.Earth execution UUID used to name S3 outputs.
        fake_data: When ``True`` and ``year_final`` exceeds the last available
            year in the land cover collection, the temporal extent is clamped
            to the last available year and a warning is logged — matching the
            behaviour of the GEE implementation.

    Returns:
        The openEO :class:`openeo.BatchJob` that was started.

    Raises:
        ValueError: When required openEO parameters are missing.
    """
    if connection is None:
        raise ValueError("'connection' is required for the openEO SOC implementation.")
    if execution_id is None:
        raise ValueError(
            "'execution_id' is required for the openEO SOC implementation."
        )

    if fl == "per pixel":
        logger.info(
            "Building openEO SOC process graph for years %d\u2013%d (Fl=per pixel).",
            year_initial,
            year_final,
        )
    else:
        logger.info(
            "Building openEO SOC process graph for years %d\u2013%d (Fl=%.3f).",
            year_initial,
            year_final,
            float(fl),
        )

    # Resolve collection IDs for this backend
    backend_url = connection.root_url
    norm = get_backend_normalizations(backend_url)
    col_aliases = norm.get("collection_aliases", {})

    soc_collection = col_aliases.get("SOC_BASELINE", "TERRASCOPE_WORLD_SOILGRIDS")
    soc_band = col_aliases.get("SOC_BASELINE_BAND", "SOC")
    lc_collection = col_aliases.get("LAND_COVER", "ESA_CCI_LC")
    crs = norm.get("crs_default", "EPSG:4326")
    lc_last_year = norm.get("lc_last_year", 2022)

    logger.debug(
        "Using collections: SOC=%s (band=%s), LC=%s, CRS=%s",
        soc_collection,
        soc_band,
        lc_collection,
        crs,
    )

    # Build bounding box from geojsons (take union envelope)
    bbox = bbox_from_geojsons(geojsons) if geojsons else None

    # ---------------------------------------------------------------------------
    # Process graph construction
    # ---------------------------------------------------------------------------
    import openeo  # noqa: PLC0415 – deferred to allow import without openeo installed

    # 1. Load baseline SOC (single-band, reference year 2000)
    soc_baseline = connection.load_collection(
        soc_collection,
        bands=[soc_band],
        spatial_extent=bbox,
        temporal_extent=["1999-01-01", "2001-12-31"],
    )
    # Reduce to a single reference image (mean over the window)
    soc_baseline_reduced = soc_baseline.reduce_temporal("mean")

    # 2. Load land cover for the analysis period
    lc_year_final = year_final
    if fake_data and year_final > lc_last_year:
        logger.warning(
            "year_final=%d exceeds the last available land cover year (%d). "
            "Clamping to %d (fake_data=True).",
            year_final,
            lc_last_year,
            lc_last_year,
        )
        lc_year_final = lc_last_year
    lc_data = connection.load_collection(
        lc_collection,
        bands=["lccs_class"],
        spatial_extent=bbox,
        temporal_extent=[f"{year_initial - 1}-01-01", f"{lc_year_final}-12-31"],
    )
    # Resample to annual time steps
    lc_annual = lc_data.aggregate_temporal_period(
        period="year",
        reducer="first",
    )

    # 3a. Load per-pixel Fl raster from climate zones (only when fl == "per pixel")
    if fl == "per pixel":
        cz_collection = col_aliases.get("CLIMATE_ZONES", "IPCC_CLIMATE_ZONES")
        logger.debug("Loading climate zones from collection %s.", cz_collection)
        climate_zones_raw = connection.load_collection(
            cz_collection,
            spatial_extent=bbox,
        )
        climate_zones_single = climate_zones_raw.reduce_temporal("first")
        fl_raster = climate_zones_single.apply_neighborhood(
            process=openeo.UDF(
                code=_make_climate_fl_udf(),
                version="1.0.0",
                runtime="Python",
            ),
            size=[
                {"dimension": "x", "value": 256, "unit": "px"},
                {"dimension": "y", "value": 256, "unit": "px"},
            ],
            overlap=[],
        )

    # 3b. Apply IPCC stock-change factor logic
    #    openEO process graphs are lazy; we build the computation as a datacube
    #    pipeline.  For the full multi-year loop we construct the factor
    #    multiplier analytically (scalar Fl when fl is a float, or per-pixel
    #    when fl == "per pixel").
    #
    #    The approach: at each annual time step compute the land-cover transition
    #    code, look up Fl × Fm × Fo, derive the annual SOC change, and accumulate.

    class_codes = sorted([c.code for c in esa_to_custom_nesting.parent.key])
    class_positions = list(range(1, len(class_codes) + 1))

    # Reclassify LC to 7-class custom scheme then to IPCC positions

    # We build a UDF-free graph where possible.  For the stock-change factors
    # we use a lookup-table via openEO's `reclassify_bin` / `vector_to_regular_cube`
    # equivalents.  Because openEO backends vary in UDF support, we encode the
    # factor tables as constant cubes and use arithmetic masking.

    lc_custom = lc_annual.apply_neighborhood(
        process=openeo.UDF(
            code=_make_lc_reclassify_udf(
                esa_to_custom_nesting.get_list()[0],
                esa_to_custom_nesting.get_list()[1],
                class_codes,
                class_positions,
            ),
            version="1.0.0",
            runtime="Python",
        ),
        size=[
            {"dimension": "x", "value": 256, "unit": "px"},
            {"dimension": "y", "value": 256, "unit": "px"},
        ],
        overlap=[],
    )

    # Merge the reclassified LC cube with the SOC baseline so the SOC-change UDF
    # can access both in the same neighbourhood window.  When per-pixel Fl is
    # requested the Fl raster is appended as the last band.
    lc_with_soc = lc_custom.merge_cubes(soc_baseline_reduced)
    per_pixel_fl = fl == "per pixel"
    if per_pixel_fl:
        lc_with_soc = lc_with_soc.merge_cubes(fl_raster)

    soc_final = lc_with_soc.apply_neighborhood(
        process=openeo.UDF(
            code=_make_soc_change_udf(
                year_initial=year_initial,
                year_final=year_final,
                fl=float(fl) if not per_pixel_fl else 1.0,
                per_pixel_fl=per_pixel_fl,
                soc_factor_land_use=_SOC_FACTOR_LAND_USE,
                soc_factor_management=_SOC_FACTOR_MANAGEMENT,
                soc_factor_organic_matter=_SOC_FACTOR_ORGANIC_MATTER,
                ipcc_nesting=ipcc_nesting,
            ),
            version="1.0.0",
            runtime="Python",
        ),
        size=[
            {"dimension": "x", "value": 256, "unit": "px"},
            {"dimension": "y", "value": 256, "unit": "px"},
        ],
        overlap=[],
    )

    # 4. Save results to S3
    result = save_result_to_s3(soc_final, execution_id, "soc")

    # 5. Submit as a batch job
    job = result.create_job(
        title=f"Trends.Earth SOC {year_initial}–{year_final} [{execution_id}]",
        description=(
            f"Soil Organic Carbon indicator computation via openEO, "
            f"execution {execution_id}, years {year_initial}–{year_final}."
        ),
    )
    job.start_job()
    logger.info(
        "Submitted openEO SOC batch job %s for execution %s.",
        job.job_id,
        execution_id,
    )
    return job


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_lc_reclassify_udf(
    from_codes: list,
    to_custom: list,
    class_codes: list,
    class_positions: list,
) -> str:
    """Return a Python UDF string that reclassifies LC to IPCC class positions."""
    return f"""\
import numpy as np
from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    from_codes = {from_codes!r}
    to_custom = {to_custom!r}
    class_codes = {class_codes!r}
    class_positions = {class_positions!r}

    arr = cube.get_array().values.copy()
    # Step 1: remap ESA codes to custom 7-class
    result = np.full_like(arr, -1, dtype=np.float32)
    for src, dst in zip(from_codes, to_custom):
        result[arr == src] = dst
    # Step 2: remap custom 7-class to IPCC positions (1-based)
    remapped = np.full_like(result, -1, dtype=np.float32)
    for code, pos in zip(class_codes, class_positions):
        remapped[result == code] = pos
    import xarray as xr
    da = cube.get_array().copy(data=remapped)
    return XarrayDataCube(da)
"""


def _make_climate_fl_udf() -> str:
    """Return a Python UDF string that remaps IPCC climate zone codes to Fl values."""
    return f"""\
import numpy as np
from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    \"\"\"Remap IPCC climate zone integers to per-pixel Fl values.\"\"\"
    zone_codes = {_CLIMATE_ZONE_CODES!r}
    zone_to_ipcc = {_CLIMATE_ZONE_TO_IPCC_IDX!r}
    ipcc_to_fl = {_IPCC_IDX_FL_VALUES!r}

    arr = cube.get_array().values.astype(np.float32)
    # Two-step remap: raw zone code \u2192 IPCC index \u2192 Fl value
    ipcc_idx = np.zeros_like(arr, dtype=np.int16)
    for src, dst in zip(zone_codes, zone_to_ipcc):
        ipcc_idx[arr == src] = dst
    fl_arr = np.zeros_like(arr, dtype=np.float32)
    for idx, fl_val in enumerate(ipcc_to_fl):
        fl_arr[ipcc_idx == idx] = fl_val
    import xarray as xr
    da = cube.get_array().copy(data=fl_arr)
    return XarrayDataCube(da)
"""


def _make_soc_change_udf(
    year_initial: int,
    year_final: int,
    fl: float,
    soc_factor_land_use: tuple,
    soc_factor_management: tuple,
    soc_factor_organic_matter: tuple,
    ipcc_nesting: Any,
    per_pixel_fl: bool = False,
) -> str:
    """Return a Python UDF string that computes SOC percent change."""
    # Serialise factor tables so they can be embedded in the UDF string
    from te_algorithms.common.soc import trans_factors_for_custom_legend

    lc_tr_fl_codes, lc_tr_fl_vals = trans_factors_for_custom_legend(
        soc_factor_land_use, ipcc_nesting
    )
    lc_tr_fm_codes, lc_tr_fm_vals = trans_factors_for_custom_legend(
        soc_factor_management, ipcc_nesting
    )
    lc_tr_fo_codes, lc_tr_fo_vals = trans_factors_for_custom_legend(
        soc_factor_organic_matter, ipcc_nesting
    )

    return f"""\
import numpy as np
from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    \"\"\"Compute annual SOC change and return percent change over the analysis period.\"\"\"
    year_initial = {year_initial!r}
    year_final = {year_final!r}
    fl = {fl!r}
    per_pixel_fl = {per_pixel_fl!r}
    lc_tr_fl_codes = {lc_tr_fl_codes!r}
    lc_tr_fl_vals = {lc_tr_fl_vals!r}
    lc_tr_fm_codes = {lc_tr_fm_codes!r}
    lc_tr_fm_vals = {lc_tr_fm_vals!r}
    lc_tr_fo_codes = {lc_tr_fo_codes!r}
    lc_tr_fo_vals = {lc_tr_fo_vals!r}

    da = cube.get_array()
    # Band layout:
    #   per_pixel_fl=False: [lc_year0 .. lc_yearN, soc_t0]
    #   per_pixel_fl=True:  [lc_year0 .. lc_yearN, soc_t0, fl_values]
    if per_pixel_fl:
        lc_bands = da.isel(bands=slice(None, -2)).values   # shape (T, Y, X)
        soc_t0 = da.isel(bands=-2).values                  # shape (Y, X)
        fl_band = da.isel(bands=-1).values                 # shape (Y, X)
        if soc_t0.ndim == 3:
            soc_t0 = soc_t0[0]
        if fl_band.ndim == 3:
            fl_band = fl_band[0]
    else:
        lc_bands = da.isel(bands=slice(None, -1)).values   # shape (T, Y, X)
        soc_t0 = da.isel(bands=-1).values                  # shape (Y, X)

    def remap(arr, codes, vals):
        out = np.full_like(arr, 1.0, dtype=np.float64)
        for c, v in zip(codes, vals):
            out[arr == c] = v
        # Resolve Fl sentinels (99 = Fl, -99 = 1/Fl) using scalar or per-pixel
        if per_pixel_fl:
            mask_99 = out == 99
            mask_n99 = out == -99
            with np.errstate(divide="ignore", invalid="ignore"):
                fl_inv = np.where(fl_band != 0, 1.0 / fl_band, 1.0)
            out = np.where(mask_99, fl_band.astype(np.float64), out)
            out = np.where(mask_n99, fl_inv, out)
        else:
            out[out == 99] = fl
            out[out == -99] = 1.0 / fl if fl != 0 else 1.0
        return out

    n_years = year_final - year_initial
    soc_stack = [soc_t0.copy()]
    soc_chg = np.zeros_like(soc_t0)
    lc_tr_prev = None
    tr_time = np.full_like(soc_t0, 2, dtype=np.float64)

    soc_t0_year = 2000

    for k in range(year_final - soc_t0_year):
        lc_t0 = lc_bands[k]
        lc_t1 = lc_bands[min(k + 1, lc_bands.shape[0] - 1)]

        lc_tr = lc_t0 * (max(lc_t0.max(), 7) + 1) + lc_t1

        if k == 0:
            lc_tr_prev = lc_tr.copy()
            tr_time = np.where(lc_t0 != lc_t1, 1, 2).astype(np.float64)
        else:
            tr_time = np.where(lc_t0 == lc_t1, tr_time + 1, 1).astype(np.float64)
            lc_tr_prev = np.where(lc_t0 != lc_t1, lc_tr, lc_tr_prev)

        lc_tr_fl = remap(lc_tr_prev, lc_tr_fl_codes, lc_tr_fl_vals)
        lc_tr_fm = remap(lc_tr_prev, lc_tr_fm_codes, lc_tr_fm_vals)
        lc_tr_fo = remap(lc_tr_prev, lc_tr_fo_codes, lc_tr_fo_vals)

        if k == 0:
            soc_chg = (soc_t0 - soc_t0 * lc_tr_fl * lc_tr_fm * lc_tr_fo) / 20.0
            soc_t1 = soc_t0 - soc_chg
            soc_stack.append(soc_t1)
        else:
            prev_soc = soc_stack[k]
            new_chg = np.where(
                lc_t0 != lc_t1,
                (prev_soc - prev_soc * lc_tr_fl * lc_tr_fm * lc_tr_fo) / 20.0,
                soc_chg,
            )
            new_chg = np.where(tr_time > 20, 0, new_chg)
            soc_chg = new_chg
            soc_stack.append(prev_soc - soc_chg)

    soc_initial = soc_stack[max(year_initial - soc_t0_year, 0)]
    soc_final_arr = soc_stack[year_final - soc_t0_year]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = np.where(
            soc_initial != 0,
            (soc_final_arr - soc_initial) / soc_initial * 100.0,
            0.0,
        )

    import xarray as xr
    # Return single-band result with the same spatial coordinates
    out_da = da.isel(bands=0).copy(data=pct_change.astype(np.float32))
    out_da = out_da.assign_coords(bands=["soc_pct_change"])
    return XarrayDataCube(out_da.expand_dims("bands", axis=0))
"""
