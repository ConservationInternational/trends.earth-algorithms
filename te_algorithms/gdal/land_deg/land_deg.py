import dataclasses
import datetime as dt
import json
import logging
import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal
from te_schemas.datafile import DataFile, combine_data_files
from te_schemas.land_cover import LCLegendNesting, LCTransitionDefinitionDeg
from te_schemas.productivity import ProductivityMode
from te_schemas.results import (
    URI,
    Band,
    DataType,
    Raster,
    RasterFileType,
    RasterResults,
)

from .. import util, workers
from ..util_numba import bizonal_total, zonal_total, zonal_total_weighted
from . import config, models, worker
from .land_deg_numba import (
    calc_deg_sdg,
    calc_lc_trans,
    calc_prod5,
    prod5_to_prod3,
    recode_deg_soc,
    recode_indicator_errors,
    recode_state,
    recode_traj,
)
from .land_deg_progress import compute_status_summary
from .land_deg_report import save_reporting_json, save_summary_table_excel

# Timeout constants for land degradation processing
TOTAL_PROCESSING_TIMEOUT = 24 * 3600  # 24 hours in seconds
PER_TILE_TIMEOUT = 4 * 3600  # 4 hours in seconds

if TYPE_CHECKING:
    from te_schemas.aoi import AOI
    from te_schemas.jobs import Job

logger = logging.getLogger(__name__)


def _prepare_land_cover_dfs(params: Dict) -> List[DataFile]:
    lc_path = params["layer_lc_path"]
    lc_dfs = [
        DataFile(
            path=util.save_vrt(lc_path, params["layer_lc_deg_band_index"]),
            bands=[Band(**params["layer_lc_deg_band"])],
        )
    ]

    for (
        lc_aux_band,
        lc_aux_band_index,
    ) in zip(params["layer_lc_aux_bands"], params["layer_lc_aux_band_indexes"]):
        lc_dfs.append(
            DataFile(
                path=util.save_vrt(lc_path, lc_aux_band_index),
                bands=[Band(**lc_aux_band)],
            )
        )
    lc_dfs.append(
        DataFile(
            path=util.save_vrt(
                params["layer_lc_trans_path"],
                params["layer_lc_trans_band_index"],
            ),
            bands=[Band(**params["layer_lc_trans_band"])],
        )
    )

    return lc_dfs


def _prepare_population_dfs(params: Dict) -> DataFile:
    population_dfs = []

    for population_band, population_band_index, path in zip(
        params["layer_population_bands"],
        params["layer_population_band_indexes"],
        params["layer_population_paths"],
    ):
        population_dfs.append(
            DataFile(
                path=util.save_vrt(path, population_band_index),
                bands=[Band(**population_band)],
            )
        )

    return population_dfs


def _prepare_soil_organic_carbon_dfs(params: Dict) -> List[DataFile]:
    soc_path = params["layer_soc_path"]
    soc_dfs = [
        DataFile(
            path=util.save_vrt(soc_path, params["layer_soc_deg_band_index"]),
            bands=[Band(**params["layer_soc_deg_band"])],
        )
    ]

    for (
        soc_aux_band,
        soc_aux_band_index,
    ) in zip(params["layer_soc_aux_bands"], params["layer_soc_aux_band_indexes"]):
        soc_dfs.append(
            DataFile(
                path=util.save_vrt(soc_path, soc_aux_band_index),
                bands=[Band(**soc_aux_band)],
            )
        )

    return soc_dfs


def _prepare_trends_earth_mode_dfs(params: Dict) -> Tuple[DataFile, DataFile, DataFile]:
    traj_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_traj_path"],
            params["layer_traj_band_index"],
        ),
        bands=[Band(**params["layer_traj_band"])],
    )
    perf_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_perf_path"],
            params["layer_perf_band_index"],
        ),
        bands=[Band(**params["layer_perf_band"])],
    )
    state_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_state_path"],
            params["layer_state_band_index"],
        ),
        bands=[Band(**params["layer_state_band"])],
    )

    return traj_vrt_df, perf_vrt_df, state_vrt_df


def _prepare_precalculated_lpd_df(params: Dict) -> DataFile:
    return DataFile(
        path=util.save_vrt(params["layer_lpd_path"], params["layer_lpd_band_index"]),
        bands=[Band(**params["layer_lpd_band"])],
    )


def _process_single_period_with_schemas(
    period, ldn_job, aoi, job_output_path, n_cpus, prepared_schemas
):
    """Process a single period with pre-loaded schemas - for parallel processing"""
    period_name = period["name"]
    period_params = period["params"]
    logger.debug("preparing land cover dfs")
    lc_dfs = _prepare_land_cover_dfs(period_params)
    logger.debug("preparing soil organic carbon dfs")
    soc_dfs = _prepare_soil_organic_carbon_dfs(period_params)
    logger.debug("preparing population dfs")
    population_dfs = _prepare_population_dfs(period_params)
    logger.debug("len(population_dfs) %s", len(population_dfs))
    logger.debug("population_dfs %s", population_dfs)
    sub_job_output_path = (
        job_output_path.parent / f"{job_output_path.stem}_{period_name}.json"
    )
    prod_mode = period_params["prod_mode"]

    period_params["periods"] = {
        "land_cover": period_params["layer_lc_deg_years"],
        "soc": period_params["layer_soc_deg_years"],
    }

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        period_params["periods"]["productivity"] = period_params["layer_traj_years"]
    elif prod_mode in (
        ProductivityMode.JRC_5_CLASS_LPD.value,
        ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
    ):
        period_params["periods"]["productivity"] = period_params["layer_lpd_years"]
    else:
        raise Exception(f"Unknown productivity mode {prod_mode}")

    # Add in period start/end if it isn't already in the parameters
    if "period" not in period_params:
        period_params["period"] = {
            "name": period_name,
            "year_initial": period_params["periods"]["productivity"]["year_initial"],
            "year_final": period_params["periods"]["productivity"]["year_final"],
        }

    # Use pre-loaded schemas instead of loading them here
    summary_table_stable_kwargs = {
        "aoi": aoi,
        "lc_legend_nesting": prepared_schemas["nesting"],
        "lc_trans_matrix": prepared_schemas["lc_trans_matrix"],
        "output_job_path": sub_job_output_path,
        "period_name": period_name,
        "periods": period_params["periods"],
        "n_cpus": n_cpus,
    }

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        traj, perf, state = _prepare_trends_earth_mode_dfs(period_params)
        compute_bbs_from = traj.path
        in_dfs = lc_dfs + soc_dfs + [traj, perf, state] + population_dfs
        summary_table, output_path, reproj_path = _compute_ld_summary_table(
            in_dfs=in_dfs,
            prod_mode=ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
            compute_bbs_from=compute_bbs_from,
            **summary_table_stable_kwargs,
        )
    elif prod_mode in (
        ProductivityMode.JRC_5_CLASS_LPD.value,
        ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
    ):
        lpd_df = _prepare_precalculated_lpd_df(period_params)
        compute_bbs_from = lpd_df.path
        in_dfs = lc_dfs + soc_dfs + [lpd_df] + population_dfs
        summary_table, output_path, reproj_path = _compute_ld_summary_table(
            in_dfs=in_dfs,
            prod_mode=prod_mode,
            compute_bbs_from=compute_bbs_from,
            **summary_table_stable_kwargs,
        )
    else:
        raise RuntimeError(f"Invalid prod_mode: {prod_mode!r}")

    sdg_band = Band(
        name=config.SDG_BAND_NAME,
        no_data_value=config.NODATA_VALUE.item(),
        metadata={
            "year_initial": period_params["period"]["year_initial"],
            "year_final": period_params["period"]["year_final"],
        },
        activated=True,
    )
    output_df = DataFile(output_path, [sdg_band])

    so3_band_total = _get_so3_band_instance(
        "total", period_params["periods"]["productivity"]
    )
    output_df.bands.append(so3_band_total)

    if _have_pop_by_sex(population_dfs):
        so3_band_female = _get_so3_band_instance(
            "female", period_params["periods"]["productivity"]
        )
        output_df.bands.append(so3_band_female)
        so3_band_male = _get_so3_band_instance(
            "male", period_params["periods"]["productivity"]
        )
        output_df.bands.append(so3_band_male)

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        prod_band = Band(
            name=config.TE_LPD_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),
            metadata={
                "year_initial": period_params["periods"]["productivity"][
                    "year_initial"
                ],
                "year_final": period_params["periods"]["productivity"]["year_final"],
            },
            activated=True,
        )
        output_df.bands.append(prod_band)

    reproj_df = combine_data_files(reproj_path, in_dfs)
    for band in reproj_df.bands:
        band.add_to_map = False

    period_vrt = job_output_path.parent / f"{sub_job_output_path.stem}_rasterdata.vrt"
    util.combine_all_bands_into_vrt([output_path, reproj_path], period_vrt)

    period_df = combine_data_files(period_vrt, [output_df, reproj_df])
    for band in period_df.bands:
        band.metadata["period"] = period_name

    summary_table_output_path = (
        sub_job_output_path.parent / f"{sub_job_output_path.stem}.xlsx"
    )
    save_summary_table_excel(
        summary_table_output_path,
        summary_table,
        period_params["periods"],
        period_params["layer_lc_years"],
        period_params["layer_soc_years"],
        summary_table_stable_kwargs["lc_legend_nesting"],
        summary_table_stable_kwargs["lc_trans_matrix"],
        period_name,
    )

    return period_df, period_vrt, summary_table, period_name


def _process_single_period(
    period, ldn_job, aoi, job_output_path, n_cpus, prepared_schemas=None
):
    """Process a single period - updated to accept pre-loaded schemas for thread safety"""
    if prepared_schemas is not None:
        # Use pre-loaded schemas (for threaded execution)
        return _process_single_period_with_schemas(
            period, ldn_job, aoi, job_output_path, n_cpus, prepared_schemas
        )

    # Original non-threaded execution path (load schemas here)
    period_name = period["name"]
    period_params = period["params"]
    logger.debug("preparing land cover dfs")
    lc_dfs = _prepare_land_cover_dfs(period_params)
    logger.debug("preparing soil organic carbon dfs")
    soc_dfs = _prepare_soil_organic_carbon_dfs(period_params)
    logger.debug("preparing population dfs")
    population_dfs = _prepare_population_dfs(period_params)
    logger.debug("len(population_dfs) %s", len(population_dfs))
    logger.debug("population_dfs %s", population_dfs)
    sub_job_output_path = (
        job_output_path.parent / f"{job_output_path.stem}_{period_name}.json"
    )
    prod_mode = period_params["prod_mode"]

    period_params["periods"] = {
        "land_cover": period_params["layer_lc_deg_years"],
        "soc": period_params["layer_soc_deg_years"],
    }

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        period_params["periods"]["productivity"] = period_params["layer_traj_years"]
    elif prod_mode in (
        ProductivityMode.JRC_5_CLASS_LPD.value,
        ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
    ):
        period_params["periods"]["productivity"] = period_params["layer_lpd_years"]
    else:
        raise Exception(f"Unknown productivity mode {prod_mode}")

    # Add in period start/end if it isn't already in the parameters
    if "period" not in period_params:
        period_params["period"] = {
            "name": period_name,
            "year_initial": period_params["periods"]["productivity"]["year_initial"],
            "year_final": period_params["periods"]["productivity"]["year_final"],
        }

    # Load schemas in main thread (non-threaded execution)
    nesting = period_params["layer_lc_deg_band"]["metadata"].get("nesting")
    if nesting:
        nesting = LCLegendNesting.Schema().loads(nesting)

    trans_matrix_data = period_params["layer_lc_deg_band"]["metadata"]["trans_matrix"]

    summary_table_stable_kwargs = {
        "aoi": aoi,
        "lc_legend_nesting": nesting,
        "lc_trans_matrix": LCTransitionDefinitionDeg.Schema().loads(trans_matrix_data),
        "output_job_path": sub_job_output_path,
        "period_name": period_name,
        "periods": period_params["periods"],
        "n_cpus": n_cpus,
    }

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        traj, perf, state = _prepare_trends_earth_mode_dfs(period_params)
        compute_bbs_from = traj.path
        in_dfs = lc_dfs + soc_dfs + [traj, perf, state] + population_dfs
        summary_table, output_path, reproj_path = _compute_ld_summary_table(
            in_dfs=in_dfs,
            prod_mode=ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
            compute_bbs_from=compute_bbs_from,
            **summary_table_stable_kwargs,
        )
    elif prod_mode in (
        ProductivityMode.JRC_5_CLASS_LPD.value,
        ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
    ):
        lpd_df = _prepare_precalculated_lpd_df(period_params)
        compute_bbs_from = lpd_df.path
        in_dfs = lc_dfs + soc_dfs + [lpd_df] + population_dfs
        summary_table, output_path, reproj_path = _compute_ld_summary_table(
            in_dfs=in_dfs,
            prod_mode=prod_mode,
            compute_bbs_from=compute_bbs_from,
            **summary_table_stable_kwargs,
        )
    else:
        raise RuntimeError(f"Invalid prod_mode: {prod_mode!r}")

    sdg_band = Band(
        name=config.SDG_BAND_NAME,
        no_data_value=config.NODATA_VALUE.item(),
        metadata={
            "year_initial": period_params["period"]["year_initial"],
            "year_final": period_params["period"]["year_final"],
        },
        activated=True,
    )
    output_df = DataFile(output_path, [sdg_band])

    so3_band_total = _get_so3_band_instance(
        "total", period_params["periods"]["productivity"]
    )
    output_df.bands.append(so3_band_total)

    if _have_pop_by_sex(population_dfs):
        so3_band_female = _get_so3_band_instance(
            "female", period_params["periods"]["productivity"]
        )
        output_df.bands.append(so3_band_female)
        so3_band_male = _get_so3_band_instance(
            "male", period_params["periods"]["productivity"]
        )
        output_df.bands.append(so3_band_male)

    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        prod_band = Band(
            name=config.TE_LPD_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),
            metadata={
                "year_initial": period_params["periods"]["productivity"][
                    "year_initial"
                ],
                "year_final": period_params["periods"]["productivity"]["year_final"],
            },
            activated=True,
        )
        output_df.bands.append(prod_band)

    reproj_df = combine_data_files(reproj_path, in_dfs)
    for band in reproj_df.bands:
        band.add_to_map = False

    period_vrt = job_output_path.parent / f"{sub_job_output_path.stem}_rasterdata.vrt"
    util.combine_all_bands_into_vrt([output_path, reproj_path], period_vrt)

    period_df = combine_data_files(period_vrt, [output_df, reproj_df])
    for band in period_df.bands:
        band.metadata["period"] = period_name

    summary_table_output_path = (
        sub_job_output_path.parent / f"{sub_job_output_path.stem}.xlsx"
    )
    save_summary_table_excel(
        summary_table_output_path,
        summary_table,
        period_params["periods"],
        period_params["layer_lc_years"],
        period_params["layer_soc_years"],
        summary_table_stable_kwargs["lc_legend_nesting"],
        summary_table_stable_kwargs["lc_trans_matrix"],
        period_name,
    )

    return period_df, period_vrt, summary_table, period_name


def summarise_land_degradation(
    ldn_job: "Job",
    aoi: "AOI",
    job_output_path: Path,
    n_cpus: int = max(1, multiprocessing.cpu_count() - 1),
) -> "Job":
    """Calculate final SDG 15.3.1 indicator and save to disk"""
    logger.debug("at top of compute_ldn")

    # Adaptive CPU scaling based on data characteristics
    import psutil

    # Check available memory and adjust CPU count accordingly
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Estimate memory per CPU core needed (rough heuristic)
    memory_per_core_gb = 4.0  # Conservative estimate for raster processing
    max_cpus_by_memory = int(available_memory_gb / memory_per_core_gb)

    # Use the minimum of requested CPUs and memory-constrained CPUs
    effective_n_cpus = min(n_cpus, max_cpus_by_memory, multiprocessing.cpu_count())
    logger.info(
        f"Using {effective_n_cpus} CPUs (requested: {n_cpus}, "
        f"memory-limited: {max_cpus_by_memory})"
    )

    summary_tables = {}
    summary_table_stable_kwargs = {}

    period_dfs = []
    period_vrts = []

    # Process periods in parallel when there are multiple periods
    if len(ldn_job.params["periods"]) > 1 and effective_n_cpus > 2:
        # Reserve CPUs for period-level parallelization
        cpus_per_period = max(1, effective_n_cpus // len(ldn_job.params["periods"]))
        period_cpus = min(
            cpus_per_period, effective_n_cpus // 2
        )  # Don't use all CPUs for periods

        logger.info(
            f"Processing {len(ldn_job.params['periods'])} periods in parallel "
            f"with {period_cpus} CPUs each"
        )

        import concurrent.futures

        def process_period_wrapper_with_thread_marker(period_data):
            """Wrapper that marks the current thread as being in a ThreadPoolExecutor"""
            current_thread = threading.current_thread()
            current_thread._is_in_thread_pool = True
            try:
                return process_period_wrapper(period_data)
            finally:
                # Clean up the marker
                if hasattr(current_thread, "_is_in_thread_pool"):
                    delattr(current_thread, "_is_in_thread_pool")

        def process_period_wrapper(period_data):
            period, cpus, prepared_schemas = period_data
            # Process single period with allocated CPU count and pre-loaded schemas
            return _process_single_period_with_schemas(
                period, ldn_job, aoi, job_output_path, cpus, prepared_schemas
            )

        # Pre-load schemas in main thread to avoid thread-safety issues
        prepared_schemas_list = []
        for period in ldn_job.params["periods"]:
            period_params = period["params"]
            nesting = period_params["layer_lc_deg_band"]["metadata"].get("nesting")
            if nesting:
                nesting = LCLegendNesting.Schema().loads(nesting)

            # Use .loads() for JSON string data
            trans_matrix_data = period_params["layer_lc_deg_band"]["metadata"][
                "trans_matrix"
            ]
            lc_trans_matrix = LCTransitionDefinitionDeg.Schema().loads(
                trans_matrix_data
            )

            prepared_schemas_list.append(
                {"nesting": nesting, "lc_trans_matrix": lc_trans_matrix}
            )

        period_data = [
            (period, period_cpus, schemas)
            for period, schemas in zip(ldn_job.params["periods"], prepared_schemas_list)
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(ldn_job.params["periods"])
        ) as executor:
            period_results = list(
                executor.map(process_period_wrapper_with_thread_marker, period_data)
            )

        for result, period, prepared_schemas in zip(
            period_results, ldn_job.params["periods"], prepared_schemas_list
        ):
            if result is None:
                raise RuntimeError("Error processing period in parallel")
            period_df, period_vrt, summary_table, period_name = result
            period_dfs.append(period_df)
            period_vrts.append(period_vrt)
            summary_tables[period_name] = summary_table

            # Populate summary_table_stable_kwargs for each period (needed for reporting)
            sub_job_output_path = (
                job_output_path.parent / f"{job_output_path.stem}_{period_name}.json"
            )
            period_params = period["params"]

            # Set up periods dictionary like in _process_single_period
            periods = {
                "land_cover": period_params["layer_lc_deg_years"],
                "soc": period_params["layer_soc_deg_years"],
            }

            prod_mode = period_params["prod_mode"]
            if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
                periods["productivity"] = period_params["layer_traj_years"]
            elif prod_mode in (
                ProductivityMode.JRC_5_CLASS_LPD.value,
                ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
            ):
                periods["productivity"] = period_params["layer_lpd_years"]

            summary_table_stable_kwargs[period_name] = {
                "aoi": aoi,
                "lc_legend_nesting": prepared_schemas["nesting"],
                "lc_trans_matrix": prepared_schemas["lc_trans_matrix"],
                "output_job_path": sub_job_output_path,
                "period_name": period_name,
                "periods": periods,
                "n_cpus": period_cpus,
            }
    else:
        # Sequential processing for single period or when CPU count is low
        for period in ldn_job.params["periods"]:
            result = _process_single_period(
                period, ldn_job, aoi, job_output_path, effective_n_cpus
            )
            if result is None:
                raise RuntimeError("Error processing period")
            period_df, period_vrt, summary_table, period_name = result
            period_dfs.append(period_df)
            period_vrts.append(period_vrt)
            summary_tables[period_name] = summary_table

            # Populate summary_table_stable_kwargs for each period (needed for reporting)
            sub_job_output_path = (
                job_output_path.parent / f"{job_output_path.stem}_{period_name}.json"
            )
            period_params = period["params"]

            # Load schemas for this period
            nesting = period_params["layer_lc_deg_band"]["metadata"].get("nesting")
            if nesting:
                nesting = LCLegendNesting.Schema().loads(nesting)

            trans_matrix_data = period_params["layer_lc_deg_band"]["metadata"][
                "trans_matrix"
            ]
            lc_trans_matrix = LCTransitionDefinitionDeg.Schema().loads(
                trans_matrix_data
            )

            # Set up periods dictionary like in _process_single_period
            periods = {
                "land_cover": period_params["layer_lc_deg_years"],
                "soc": period_params["layer_soc_deg_years"],
            }

            prod_mode = period_params["prod_mode"]
            if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
                periods["productivity"] = period_params["layer_traj_years"]
            elif prod_mode in (
                ProductivityMode.JRC_5_CLASS_LPD.value,
                ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
            ):
                periods["productivity"] = period_params["layer_lpd_years"]

            summary_table_stable_kwargs[period_name] = {
                "aoi": aoi,
                "lc_legend_nesting": nesting,
                "lc_trans_matrix": lc_trans_matrix,
                "output_job_path": sub_job_output_path,
                "period_name": period_name,
                "periods": periods,
                "n_cpus": effective_n_cpus,
            }

    if len(ldn_job.params["periods"]) > 1:
        # Make temporary combined VRT and DataFile just for this reporting period
        # calculations. Don't save these in the output folder as at end of this
        # process all the DFs will be combined and referenced to a VRT in that
        # folder
        temp_overall_vrt = Path(
            tempfile.NamedTemporaryFile(suffix=".vrt", delete=False).name
        )
        util.combine_all_bands_into_vrt(period_vrts, temp_overall_vrt)
        temp_df = combine_data_files(temp_overall_vrt, period_dfs)

        # Ensure the same lc legend and nesting are used for both the
        # baseline and each reporting period (at least in terms of codes
        # and their nesting)
        baseline_nesting = LCLegendNesting.Schema().loads(
            ldn_job.params["periods"][0]["params"]["layer_lc_deg_band"]["metadata"].get(
                "nesting"
            )
        )
        for period_number in range(1, len(ldn_job.params["periods"])):
            reporting_nesting = LCLegendNesting.Schema().loads(
                ldn_job.params["periods"][period_number]["params"]["layer_lc_deg_band"][
                    "metadata"
                ].get("nesting")
            )
            assert baseline_nesting.nesting == reporting_nesting.nesting

        logger.debug(f"Computing reporting period {period_number} summary")

        # Get prod_mode and compute_bbs_from from the first period
        first_period_params = ldn_job.params["periods"][0]["params"]
        prod_mode = first_period_params["prod_mode"]

        # Determine compute_bbs_from based on productivity mode
        if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
            compute_bbs_from = first_period_params["layer_traj_path"]
        elif prod_mode in (
            ProductivityMode.JRC_5_CLASS_LPD.value,
            ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
        ):
            compute_bbs_from = first_period_params["layer_lpd_path"]
        else:
            raise Exception(f"Unknown productivity mode {prod_mode}")

        summary_table_status, summary_table_change, reporting_df = (
            compute_status_summary(
                temp_df,
                prod_mode,
                job_output_path,
                aoi,
                compute_bbs_from,
                ldn_job.params["periods"],
                baseline_nesting,
                n_cpus=effective_n_cpus,
            )
        )
        period_vrts.append(reporting_df.path)
        period_dfs.append(reporting_df)
    else:
        summary_table_status = None
        summary_table_change = None

    logger.info("Finalizing layers")

    # Add detailed logging and timeout for finalization steps
    FINALIZATION_TIMEOUT = 30 * 60  # 30 minutes timeout
    finalization_start_time = time.time()

    class FinalizationTimeoutError(Exception):
        pass

    # Cross-platform timeout using threading
    timeout_occurred = threading.Event()

    def timeout_handler():
        time.sleep(FINALIZATION_TIMEOUT)
        if not timeout_occurred.is_set():
            timeout_occurred.set()
            logger.error(
                "Finalization phase timed out after %s seconds", FINALIZATION_TIMEOUT
            )

    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()

    try:
        logger.info("Step 1: Starting VRT combination at %s", time.strftime("%H:%M:%S"))
        overall_vrt_path = job_output_path.parent / f"{job_output_path.stem}.vrt"
        logging.debug("Combining all period VRTs into %s", overall_vrt_path)
        logging.debug("Period VRTs are: %s", period_vrts)

        # Log VRT file sizes to check for issues
        for i, vrt_path in enumerate(period_vrts):
            if timeout_occurred.is_set():
                raise FinalizationTimeoutError("Finalization phase timed out")
            if os.path.exists(vrt_path):
                size_mb = os.path.getsize(vrt_path) / (1024 * 1024)
                logger.info("VRT file %d: %s (size: %.1f MB)", i + 1, vrt_path, size_mb)
            else:
                logger.warning("VRT file %d does not exist: %s", i + 1, vrt_path)

        if timeout_occurred.is_set():
            raise FinalizationTimeoutError("Finalization phase timed out")

        vrt_start = time.time()
        util.combine_all_bands_into_vrt(period_vrts, overall_vrt_path)
        vrt_time = time.time() - vrt_start
        logger.info("Step 1: VRT combination completed in %.1f seconds", vrt_time)

        if timeout_occurred.is_set():
            raise FinalizationTimeoutError("Finalization phase timed out")

        logger.info(
            "Step 2: Starting data file combination at %s", time.strftime("%H:%M:%S")
        )
        datafile_start = time.time()
        out_df = combine_data_files(overall_vrt_path, period_dfs)
        out_df.path = overall_vrt_path.name
        datafile_time = time.time() - datafile_start
        logger.info(
            "Step 2: Data file combination completed in %.1f seconds", datafile_time
        )

        if timeout_occurred.is_set():
            raise FinalizationTimeoutError("Finalization phase timed out")

        logger.info(
            "Step 3: Starting band key JSON creation at %s", time.strftime("%H:%M:%S")
        )
        json_start = time.time()
        # Also save bands to a key file for ease of use in PRAIS
        key_json = job_output_path.parent / f"{job_output_path.stem}_band_key.json"
        with open(key_json, "w") as f:
            json.dump(DataFile.Schema().dump(out_df), f, indent=4)
        json_time = time.time() - json_start
        logger.info(
            "Step 3: Band key JSON creation completed in %.1f seconds", json_time
        )

        if timeout_occurred.is_set():
            raise FinalizationTimeoutError("Finalization phase timed out")

        logger.info(
            "Step 4: Starting summary JSON report at %s", time.strftime("%H:%M:%S")
        )
        report_start = time.time()
        summary_json_output_path = (
            job_output_path.parent / f"{job_output_path.stem}_summary.json"
        )
        report_json = save_reporting_json(
            summary_json_output_path,
            summary_tables,
            summary_table_status,
            summary_table_change,
            ldn_job.params,
            ldn_job.task_name,
            aoi,
            summary_table_stable_kwargs,
        )
        report_time = time.time() - report_start
        logger.info(
            "Step 4: Summary JSON report completed in %.1f seconds", report_time
        )

        if timeout_occurred.is_set():
            raise FinalizationTimeoutError("Finalization phase timed out")

        logger.info(
            "Step 5: Creating final results object at %s", time.strftime("%H:%M:%S")
        )
        results_start = time.time()
        results = RasterResults(
            name="land_condition_summary",
            uri=URI(uri=overall_vrt_path),
            rasters={
                DataType.INT16.value: Raster(
                    uri=URI(uri=overall_vrt_path),
                    bands=out_df.bands,
                    datatype=DataType.INT16,
                    filetype=RasterFileType.COG,
                ),
            },
            data={"report": report_json},
        )
        results_time = time.time() - results_start
        logger.info(
            "Step 5: Results object creation completed in %.1f seconds", results_time
        )

        # Signal completion to stop timeout thread
        timeout_occurred.set()

        total_finalization_time = time.time() - finalization_start_time
        logger.info(
            "All finalization steps completed successfully in %.1f seconds total",
            total_finalization_time,
        )

    except FinalizationTimeoutError:
        logger.error(
            "Finalization phase timed out after %s seconds", FINALIZATION_TIMEOUT
        )
        raise
    except Exception as e:
        timeout_occurred.set()  # Stop timeout thread
        logger.error("Error during finalization: %s", e)
        logger.exception("Full exception details:")
        raise

    ldn_job.end_date = dt.datetime.now(dt.timezone.utc)
    ldn_job.progress = 100

    return results


def _process_block_summary(
    params: models.DegradationSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw,
) -> Tuple[models.SummaryTableLD, Dict]:
    # Create the key used to recode lc classes from raw codes to ordinal codes,
    # needed for transition calculations
    class_codes = sorted([c.code for c in params.nesting.child.key])
    class_positions = [*range(1, len(class_codes) + 1)]

    # Cache expensive lookups once at the start
    lc_indices = params.in_df.indices_for_name(config.LC_BAND_NAME)
    lc_band_years = params.in_df.metadata_for_name(config.LC_BAND_NAME, "year")
    soc_indices = params.in_df.indices_for_name(config.SOC_BAND_NAME)
    soc_band_years = params.in_df.metadata_for_name(config.SOC_BAND_NAME, "year")

    lc_bands = [(band, year) for band, year in zip(lc_indices, lc_band_years)]
    soc_bands = [(band, year) for band, year in zip(soc_indices, soc_band_years)]

    # Create lookup dictionaries for O(1) year-to-row mapping
    lc_year_to_row = {year: row for row, year in lc_bands}
    lc_years_set = set(lc_band_years)  # For faster membership testing

    # Cache all band index lookups at the start for efficiency
    traj_band_idx = (
        params.in_df.index_for_name(config.TRAJ_BAND_NAME)
        if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value
        else None
    )
    state_band_idx = (
        params.in_df.index_for_name(config.STATE_BAND_NAME)
        if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value
        else None
    )
    perf_band_idx = (
        params.in_df.index_for_name(config.PERF_BAND_NAME)
        if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value
        else None
    )
    soc_deg_band_idx = params.in_df.index_for_name(config.SOC_DEG_BAND_NAME)
    lc_deg_band_idx = params.in_df.index_for_name(config.LC_DEG_BAND_NAME)

    # Pre-compute population band indices
    pop_rows_total = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field="type", field_filter="total"
    )
    pop_rows_male = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field="type", field_filter="male"
    )
    pop_rows_female = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field="type", field_filter="female"
    )

    # Create container for output arrays (will write later in main thread)
    write_arrays = []

    # Calculate cell area for each horizontal line
    # logger.debug('y: {}'.format(y))
    # logger.debug('x: {}'.format(x))
    # logger.debug('rows: {}'.format(rows))

    # Make an array of the same size as the input arrays containing
    # the area of each cell (which is identical for all cells in a
    # given row - cell areas only vary among rows)
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1).astype(np.float64)

    if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        traj_array = in_array[traj_band_idx, :, :]
        traj_recode = recode_traj(traj_array)

        state_array = in_array[state_band_idx, :, :]
        state_recode = recode_state(state_array)

        perf_array = in_array[perf_band_idx, :, :]

        deg_prod5 = calc_prod5(traj_recode, state_recode, perf_array)

    elif params.prod_mode in (
        ProductivityMode.JRC_5_CLASS_LPD.value,
        ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value,
    ):
        if params.prod_mode == ProductivityMode.JRC_5_CLASS_LPD.value:
            band_name = config.JRC_LPD_BAND_NAME
        elif params.prod_mode == ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value:
            band_name = config.FAO_WOCAT_LPD_BAND_NAME

        band_idx = params.in_df.index_for_name(band_name)
        deg_prod5 = in_array[band_idx, :, :]
        # TODO: Below is temporary until missing data values are
        # fixed in LPD layer on GEE and missing data values are
        # fixed in LPD layer made by UNCCD for SIDS
        deg_prod5[(deg_prod5 == 0) | (deg_prod5 == 15)] = config.NODATA_VALUE
    else:
        raise Exception(f"Unknown productivity mode {params.prod_mode}")

    # Recode deg_prod5 as stable, degraded, improved (deg_prod3)
    deg_prod3 = prod5_to_prod3(deg_prod5)

    if "prod" in params.error_recode:
        prod_error_recode = in_array[
            params.in_df.index_for_name(config.PROD_DEG_ERROR_RECODE_BAND_NAME), :, :
        ]
        recode_indicator_errors(deg_prod3, prod_error_recode)

    ###########################################################
    # Calculate LC transition arrays
    lc_deg_band_period = params.periods["land_cover"]

    # Use cached lookup dictionaries for O(1) access
    lc_deg_initial_cover_row = lc_year_to_row[lc_deg_band_period["year_initial"]]
    lc_deg_final_cover_row = lc_year_to_row[lc_deg_band_period["year_final"]]

    # a_lc_trans_lc_deg is an array land cover transitions over the time period
    # used in the land cover degradation layer
    a_lc_trans_lc_deg = calc_lc_trans(
        in_array[lc_deg_initial_cover_row, :, :],
        in_array[lc_deg_final_cover_row, :, :],
        params.trans_matrix.legend.get_multiplier(),
        class_codes,
        class_positions,
    )
    lc_trans_arrays = [a_lc_trans_lc_deg]
    lc_trans_zonal_areas_periods = [lc_deg_band_period]

    # Productivity data might be calculated over a different period than the
    # land cover degradation data. If this is the case, and land cover layers
    # are available for the years actually used for productivity, then create
    # an array of land cover transition that can be used for productivity, and
    # call that a_lc_trans_prod_deg
    soc_deg_band_period = params.periods["soc"]
    prod_deg_band_period = params.periods["productivity"]

    if prod_deg_band_period == lc_deg_band_period:
        a_lc_trans_prod_deg = a_lc_trans_lc_deg
    elif (
        prod_deg_band_period["year_initial"] in lc_years_set
        and prod_deg_band_period["year_final"] in lc_years_set
    ):
        prod_deg_initial_cover_row = lc_year_to_row[
            prod_deg_band_period["year_initial"]
        ]
        prod_deg_final_cover_row = lc_year_to_row[prod_deg_band_period["year_final"]]
        a_lc_trans_prod_deg = calc_lc_trans(
            in_array[prod_deg_initial_cover_row, :, :],
            in_array[prod_deg_final_cover_row, :, :],
            params.trans_matrix.legend.get_multiplier(),
            class_codes,
            class_positions,
        )
        lc_trans_arrays.append(a_lc_trans_prod_deg)
        lc_trans_zonal_areas_periods.append(prod_deg_band_period)
    else:
        a_lc_trans_prod_deg = None
    # Soil organic carbon data also might be calculated over a different period
    # than the land cover degradation data. Similar to what was done for
    # productivity, if this is the case create an array of land cover
    # transition that can be used for SOC, and call a_lc_trans_soc_deg

    if soc_deg_band_period == lc_deg_band_period:
        a_lc_trans_soc_deg = a_lc_trans_lc_deg
    elif soc_deg_band_period == prod_deg_band_period:
        a_lc_trans_soc_deg = a_lc_trans_prod_deg
    elif (
        soc_deg_band_period["year_initial"] in lc_years_set
        and soc_deg_band_period["year_final"] in lc_years_set
    ):
        soc_deg_initial_cover_row = lc_year_to_row[soc_deg_band_period["year_initial"]]
        soc_deg_final_cover_row = lc_year_to_row[soc_deg_band_period["year_final"]]
        a_lc_trans_soc_deg = calc_lc_trans(
            in_array[soc_deg_initial_cover_row, :, :],
            in_array[soc_deg_final_cover_row, :, :],
            params.trans_matrix.legend.get_multiplier(),
        )
        lc_trans_arrays.append(a_lc_trans_soc_deg)
        lc_trans_zonal_areas_periods.append(soc_deg_band_period)
    else:
        a_lc_trans_soc_deg = None

    ###########################################################
    # Calculate SOC totals by year. Note final units of soc_totals
    # tables are tons C (summed over the total area of each class).

    # First filter the SOC years to only those years for which land cover is
    # available. Use cached lookup for efficiency.
    soc_bands_with_lc_avail = [
        (band, year) for band, year in soc_bands if year in lc_years_set
    ]

    # Pre-compute LC row indices for SOC years using cached lookup
    lc_rows_for_soc = [lc_year_to_row[year] for _, year in soc_bands_with_lc_avail]
    soc_by_lc_annual_totals = []

    for index, (soc_row, _) in enumerate(soc_bands_with_lc_avail):
        a_lc = in_array[lc_rows_for_soc[index], :, :]
        a_soc = in_array[soc_row, :, :]
        soc_by_lc_annual_totals.append(
            zonal_total_weighted(
                a_lc,
                a_soc,
                cell_areas * 100,
                mask,  # from sq km to hectares
            )
        )

    ###########################################################
    # Calculate crosstabs for productivity

    if a_lc_trans_prod_deg is not None:
        lc_trans_prod_bizonal = bizonal_total(
            a_lc_trans_prod_deg, deg_prod5, cell_areas, mask
        )
    else:
        # If no land cover data is available for first year of productivity
        # data, then can't do this bizonal total
        lc_trans_prod_bizonal = {}

    lc_annual_totals = []

    for lc_row, _ in lc_bands:
        a_lc = in_array[lc_row, :, :]
        lc_annual_totals.append(zonal_total(a_lc, cell_areas, mask))

    ###########################################################
    # Calculate crosstabs for land cover
    lc_trans_zonal_areas = []

    for lc_trans_array in lc_trans_arrays:
        lc_trans_zonal_areas.append(zonal_total(lc_trans_array, cell_areas, mask))

    ################
    # Calculate SDG

    # Derive a water mask from last lc year - handling fact that there could be
    # multiple water codes if this is a custom legend. 7 is the IPCC water code.
    water = np.isin(in_array[lc_deg_initial_cover_row, :, :], params.nesting.nesting[7])
    water = water.astype(bool, copy=False)

    deg_soc = in_array[soc_deg_band_idx, :, :]
    deg_soc = recode_deg_soc(deg_soc, water)

    if "soc" in params.error_recode:
        soc_error_recode = in_array[
            params.in_df.index_for_name(config.SOC_DEG_ERROR_RECODE_BAND_NAME), :, :
        ]
        deg_soc = recode_indicator_errors(deg_soc, soc_error_recode)

    deg_lc = in_array[lc_deg_band_idx, :, :]

    if "lc" in params.error_recode:
        lc_error_recode = in_array[
            params.in_df.index_for_name(config.LC_DEG_ERROR_RECODE_BAND_NAME), :, :
        ]
        deg_lc = recode_indicator_errors(deg_lc, lc_error_recode)

    deg_sdg = calc_deg_sdg(deg_prod3, deg_lc, deg_soc)

    if "sdg" in params.error_recode:
        sdg_error_recode = in_array[
            params.in_df.index_for_name(config.SDG_DEG_ERROR_RECODE_BAND_NAME), :, :
        ]
        deg_sdg = recode_indicator_errors(deg_sdg, sdg_error_recode)

    write_arrays.append({"array": deg_sdg, "xoff": xoff, "yoff": yoff})

    ###########################################################
    # Tabulate summaries
    sdg_summary = zonal_total(deg_sdg, cell_areas, mask)
    lc_summary = zonal_total(deg_lc, cell_areas, mask)

    # Make a tabulation of the productivity and SOC summaries without water for
    # usage on Prais4
    mask_plus_water = mask.copy()
    mask_plus_water[water] = True
    prod_summary = {
        "all_cover_types": zonal_total(deg_prod3, cell_areas, mask),
        "non_water": zonal_total(deg_prod3, cell_areas, mask_plus_water),
    }

    soc_summary = {
        "all_cover_types": zonal_total(deg_soc, cell_areas, mask),
        "non_water": zonal_total(deg_soc, cell_areas, mask_plus_water),
    }

    ###########################################################
    # Population affected by degradation

    if len(pop_rows_total) == 1:
        assert len(pop_rows_male) == 0 and len(pop_rows_female) == 0
        pop_by_sex = False

        pop_array_total = in_array[pop_rows_total[0], :, :].astype(np.float64)
        pop_array_total_masked = pop_array_total.copy()
        pop_array_total_masked[pop_array_total == config.NODATA_VALUE] = 0
        sdg_zonal_population_male = {}
        sdg_zonal_population_female = {}
    else:
        assert len(pop_rows_total) == 0
        assert len(pop_rows_male) == 1 and len(pop_rows_female) == 1
        pop_by_sex = True

        logger.debug(
            "pop_rows_male[0] %s, in_array.shape %s", pop_rows_male[0], in_array.shape
        )
        logger.debug(
            "pop_rows_female[0] %s, in_array.shape %s",
            pop_rows_female[0],
            in_array.shape,
        )

        pop_array_male = in_array[pop_rows_male[0], :, :].astype(np.float64)
        pop_array_male_masked = pop_array_male.copy()
        pop_array_male_masked[pop_array_male == config.NODATA_VALUE] = 0
        sdg_zonal_population_male = zonal_total(deg_sdg, pop_array_male_masked, mask)

        pop_array_female = in_array[pop_rows_female[0], :, :].astype(np.float64)
        pop_array_female_masked = pop_array_female.copy()
        pop_array_female_masked[pop_array_female == config.NODATA_VALUE] = 0
        sdg_zonal_population_female = zonal_total(
            deg_sdg, pop_array_female_masked, mask
        )

        pop_array_total = pop_array_male + pop_array_female
        pop_array_total_masked = pop_array_male_masked + pop_array_female_masked

    sdg_zonal_population_total = zonal_total(deg_sdg, pop_array_total_masked, mask)

    # Save SO3 array
    pop_array_total[deg_sdg == -1] = -pop_array_total[deg_sdg == -1]
    # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
    # means LC transitions that indicate to/from water as deg/imp will be
    # masked out
    pop_array_total[water] = config.NODATA_VALUE
    write_arrays.append({"array": pop_array_total, "xoff": xoff, "yoff": yoff})

    if pop_by_sex:
        # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
        # means LC transitions that indicate to/from water as deg/imp will be
        # masked out
        pop_array_female[water] = config.NODATA_VALUE
        pop_array_female[deg_sdg == -1] = -pop_array_female[deg_sdg == -1]
        write_arrays.append({"array": pop_array_female, "xoff": xoff, "yoff": yoff})
        # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
        # means LC transitions that indicate to/from water as deg/imp will be
        # masked out
        pop_array_male[water] = config.NODATA_VALUE
        pop_array_male[deg_sdg == -1] = -pop_array_male[deg_sdg == -1]
        write_arrays.append({"array": pop_array_male, "xoff": xoff, "yoff": yoff})

    if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        write_arrays.append({"array": deg_prod5, "xoff": xoff, "yoff": yoff})

    return (
        models.SummaryTableLD(
            soc_by_lc_annual_totals,
            lc_annual_totals,
            lc_trans_zonal_areas,
            lc_trans_zonal_areas_periods,
            lc_trans_prod_bizonal,
            sdg_zonal_population_total,
            sdg_zonal_population_male,
            sdg_zonal_population_female,
            sdg_summary,
            prod_summary,
            lc_summary,
            soc_summary,
        ),
        write_arrays,
    )


def _get_n_pop_band_for_type(dfs, pop_type):
    n_bands = 0

    for df in dfs:
        n_bands += len(
            df.indices_for_name(
                config.POPULATION_BAND_NAME, field="type", field_filter=pop_type
            )
        )

    return n_bands


def _have_pop_by_sex(pop_dfs):
    # Cache the function calls to avoid repeated lookups
    n_total_pop_bands = _get_n_pop_band_for_type(pop_dfs, "total")
    n_female_pop_bands = _get_n_pop_band_for_type(pop_dfs, "female")
    n_male_pop_bands = _get_n_pop_band_for_type(pop_dfs, "male")

    logger.debug(
        "n_total_pop_bands %s, n_female_pop_bands %s, n_male_pop_bands %s",
        n_total_pop_bands,
        n_female_pop_bands,
        n_male_pop_bands,
    )

    if n_total_pop_bands == 1:
        assert n_female_pop_bands == 0 and n_male_pop_bands == 0
        return False
    else:
        assert n_total_pop_bands == 0
        assert n_female_pop_bands == 1 and n_male_pop_bands == 1
        return True


def _get_so3_band_instance(population_type, prod_params):
    return Band(
        name=config.POP_AFFECTED_BAND_NAME,
        no_data_value=config.NODATA_VALUE.item(),  # write as python type
        metadata={
            "population_year": prod_params["year_final"],
            "deg_year_initial": prod_params["year_initial"],
            "deg_year_final": prod_params["year_final"],
            "type": population_type,
        },
        add_to_map=False,
    )


@dataclasses.dataclass()
class SummarizeTileInputs:
    in_file: Path
    wkt_aoi: str
    in_dfs: List[DataFile]
    prod_mode: str
    lc_legend_nesting: LCLegendNesting
    lc_trans_matrix: LCTransitionDefinitionDeg
    period_name: str
    periods: dict
    mask_worker_function: Optional[Callable] = None
    mask_worker_params: Optional[dict] = None
    deg_worker_function: Optional[Callable] = None
    deg_worker_params: Optional[dict] = None


def _summarize_tile(inputs: SummarizeTileInputs):
    error_message = None
    tile_name = inputs.in_file.name
    logger.debug(
        "Processing tile: %s", tile_name
    )  # Changed to debug to reduce verbosity

    # Compute a mask layer that will be used in the tabulation code to
    # mask out areas outside of the AOI. Do this instead of using
    # gdal.Clip to save having to clip and rewrite all of the layers in
    # the VRT
    mask_tif = tempfile.NamedTemporaryFile(
        suffix="_ld_summary_mask.tif", delete=False
    ).name
    logger.debug("Saving mask to %s", mask_tif)
    geojson = util.wkt_geom_to_geojson_file_string(inputs.wkt_aoi)

    if inputs.mask_worker_function:
        mask_worker_params = inputs.mask_worker_params or {}
        mask_result = inputs.mask_worker_function(
            mask_tif, geojson, str(inputs.in_file), **mask_worker_params
        )

    else:
        mask_worker = workers.Mask(mask_tif, geojson, str(inputs.in_file))
        mask_result = mask_worker.work()

    if not mask_result:
        error_message = f"Error creating mask for tile {tile_name}."
        result = None

    else:
        logger.debug(
            "Computing degradation summary for tile: %s", tile_name
        )  # Changed to debug to reduce verbosity
        in_df = combine_data_files(inputs.in_file, inputs.in_dfs)
        n_out_bands = 2  # 1 band for SDG, and 1 band for total pop affected

        if _have_pop_by_sex(inputs.in_dfs):
            logger.debug("Have population broken down by sex - adding 2 output bands")
            n_out_bands += 2

        if inputs.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
            model_band_number = in_df.index_for_name(config.TRAJ_BAND_NAME) + 1
            # Add a band for the combined productivity indicator
            n_out_bands += 1
        else:
            model_band_number = in_df.index_for_name(config.LC_DEG_BAND_NAME) + 1

        out_file = inputs.in_file.parent / (
            inputs.in_file.stem + "_sdg" + inputs.in_file.suffix
        )
        logger.debug("Calculating summary table and saving to %s", out_file)

        params = models.DegradationSummaryParams(
            in_df=in_df,
            prod_mode=inputs.prod_mode,
            in_file=in_df.path,
            out_file=out_file,
            model_band_number=model_band_number,
            n_out_bands=n_out_bands,
            mask_file=mask_tif,
            nesting=inputs.lc_legend_nesting,
            trans_matrix=inputs.lc_trans_matrix,
            period_name=inputs.period_name,
            periods=inputs.periods,
        )

        if inputs.deg_worker_function:
            deg_worker_params = inputs.deg_worker_params or {}
            result = inputs.deg_worker_function(params, **deg_worker_params)
        else:
            summarizer = worker.DegradationSummary(params, _process_block_summary)
            result = summarizer.work()

        if not result:
            if result.is_killed():
                error_message = (
                    f"Cancelled calculation of summary table for tile {tile_name}."
                )
            else:
                error_message = f"Error calculating summary table for tile {tile_name}."
                result = None
        else:
            logger.debug(
                "Completed processing tile: %s", tile_name
            )  # Changed to debug to reduce verbosity
            result = models.accumulate_summarytableld(result)
            result.cast_to_cpython()  # needed for multiprocessing

    return result, out_file, error_message


def _aoi_process_multiprocess(inputs, n_cpus):
    from .. import util

    # Use ThreadPoolExecutor for I/O bound tasks + ProcessPoolExecutor for CPU bound
    # This hybrid approach can improve throughput when tasks have mixed characteristics

    # Check if we're already in a ThreadPoolExecutor thread to avoid deadlocks
    current_thread = threading.current_thread()
    is_in_thread_pool = getattr(
        current_thread, "_is_in_thread_pool", False
    ) or current_thread.name.startswith("ThreadPoolExecutor")

    if is_in_thread_pool:
        logger.info(
            "Running in ThreadPoolExecutor thread - using multiprocessing.Pool instead of ProcessPoolExecutor to avoid deadlock"
        )
        # Use multiprocessing.Pool instead of ProcessPoolExecutor when in a thread
        # This avoids the deadlock while maintaining parallel processing
        chunksize = max(1, len(inputs) // (n_cpus * 2))

        with multiprocessing.get_context("spawn").Pool(n_cpus) as p:
            summary_tables = []
            out_files = []
            total_tiles = len(inputs)
            logger.info(
                f"Processing {total_tiles} tiles with {n_cpus} processes using multiprocessing.Pool"
            )

            try:
                for n, output in enumerate(
                    p.imap_unordered(_summarize_tile, inputs, chunksize=chunksize)
                ):
                    completed_tiles = n + 1
                    progress_percent = (completed_tiles / total_tiles) * 100

                    # Only log progress at significant milestones to reduce log spam
                    if (
                        completed_tiles % max(5, total_tiles // 20) == 0
                    ) or completed_tiles == total_tiles:
                        util.log_progress(
                            completed_tiles / total_tiles,
                            message=f"Land degradation processing: {completed_tiles}/{total_tiles} tiles completed ({progress_percent:.1f}%) - {total_tiles - completed_tiles} remaining",
                        )
                    error_message = output[2]

                    if error_message is not None:
                        logger.error("Error %s", error_message)
                        p.terminate()
                        return None

                    summary_tables.append(output[0])
                    out_files.append(output[1])

                logger.info(
                    f"Successfully completed {total_tiles}/{total_tiles} tiles with multiprocessing.Pool"
                )
            except Exception as e:
                logger.error(f"Error in multiprocessing.Pool: {e}")
                p.terminate()
                return None

    # For very large datasets, use more aggressive parallelization
    elif len(inputs) > n_cpus * 2:
        # Use process pool with larger chunksize for better load balancing
        chunksize = max(1, len(inputs) // (n_cpus * 4))

        with multiprocessing.get_context("spawn").Pool(n_cpus) as p:
            summary_tables = []
            out_files = []
            total_tiles = len(inputs)
            logger.info(
                f"Processing {total_tiles} tiles in parallel with {n_cpus} processes"
            )

            try:
                # Use map_async with timeout for better control
                async_result = p.map_async(_summarize_tile, inputs, chunksize=chunksize)

                # Wait for results with timeout
                results = async_result.get(timeout=TOTAL_PROCESSING_TIMEOUT)

                # Process results with progress tracking
                for n, output in enumerate(results):
                    completed_tiles = n + 1
                    progress_percent = (completed_tiles / total_tiles) * 100
                    util.log_progress(
                        completed_tiles / total_tiles,
                        message=f"Land degradation processing: {completed_tiles}/{total_tiles} tiles completed ({progress_percent:.1f}%) - {total_tiles - completed_tiles} remaining",
                    )
                    error_message = output[2]

                    if error_message is not None:
                        logger.error("Error %s", error_message)
                        p.terminate()
                        return None

                    summary_tables.append(output[0])
                    out_files.append(output[1])

                logger.info(
                    f"Successfully completed {len(results)}/{total_tiles} tiles with large dataset processing"
                )

            except multiprocessing.TimeoutError as e:
                logger.error(
                    f"Large dataset processing timed out after {TOTAL_PROCESSING_TIMEOUT // 3600} hours: {e}"
                )
                p.terminate()
                p.join()
                return None
            except Exception as e:
                logger.error(f"Error in large dataset processing: {e}")
                p.terminate()
                p.join()
                return None
    else:
        # For smaller datasets, prefer multiprocessing.Pool for reliability
        # ProcessPoolExecutor can have deadlock issues in complex processing workflows
        logger.info(
            f"Using multiprocessing.Pool with {n_cpus} workers for reliable processing"
        )
        chunksize = max(1, len(inputs) // (n_cpus * 2))

        with multiprocessing.get_context("spawn").Pool(n_cpus) as p:
            summary_tables = []
            out_files = []
            total_tiles = len(inputs)
            logger.info(
                f"Processing {total_tiles} tiles with {n_cpus} processes using multiprocessing.Pool"
            )

            try:
                # Use map_async with timeout for better control
                async_result = p.map_async(_summarize_tile, inputs, chunksize=chunksize)

                # Wait for results with timeout
                results = async_result.get(timeout=TOTAL_PROCESSING_TIMEOUT)

                # Process results
                for n, output in enumerate(results):
                    completed_tiles = n + 1
                    progress_percent = (completed_tiles / total_tiles) * 100
                    util.log_progress(
                        completed_tiles / total_tiles,
                        message=f"Land degradation processing: {completed_tiles}/{total_tiles} tiles completed ({progress_percent:.1f}%) - {total_tiles - completed_tiles} remaining",
                    )
                    error_message = output[2]

                    if error_message is not None:
                        logger.error("Error %s", error_message)
                        p.terminate()
                        return None

                    summary_tables.append(output[0])
                    out_files.append(output[1])

                logger.info(
                    f"Successfully completed {len(results)}/{total_tiles} tiles with multiprocessing.Pool"
                )

            except multiprocessing.TimeoutError as e:
                logger.error(
                    f"Tile processing timed out after {TOTAL_PROCESSING_TIMEOUT // 3600} hours: {e}"
                )
                p.terminate()
                p.join()
                return None
            except Exception as e:
                logger.error(f"Error in multiprocessing.Pool: {e}")
                p.terminate()
                p.join()
                return None

    summary_table = models.accumulate_summarytableld(summary_tables)
    return summary_table, out_files


def _aoi_process_sequential(inputs):
    summary_tables = []
    out_files = []
    total_tiles = len(inputs)
    logger.info(f"Processing {total_tiles} tiles sequentially")

    for n, item in enumerate(inputs):
        tile_num = n + 1
        logger.info(f"Starting tile {tile_num}/{total_tiles}: {item.in_file.name}")
        output = _summarize_tile(item)
        completed_tiles = tile_num
        progress_percent = (completed_tiles / total_tiles) * 100
        util.log_progress(
            completed_tiles / total_tiles,
            message=f"Land degradation processing: {completed_tiles}/{total_tiles} tiles completed ({progress_percent:.1f}%) - {total_tiles - completed_tiles} remaining",
        )
        error_message = output[2]

        if error_message is not None:
            logger.error("Error %s", error_message)
            break

        summary_tables.append(output[0])
        out_files.append(output[1])

    summary_table = models.accumulate_summarytableld(summary_tables)

    return summary_table, out_files


def _process_region(
    wkt_aoi,
    pixel_aligned_bbox,
    in_dfs: List[DataFile],
    output_layers_path: Path,
    prod_mode: str,
    lc_legend_nesting: LCLegendNesting,
    lc_trans_matrix: LCTransitionDefinitionDeg,
    period_name: str,
    periods: dict,
    n_cpus: int,
    translate_worker_function: Callable = None,
    translate_worker_params: dict = None,
    mask_worker_function: Callable = None,
    mask_worker_params: dict = None,
    deg_worker_function: Callable = None,
    deg_worker_params: dict = None,
) -> Tuple[Optional[models.SummaryTableLD], str]:
    """Runs summary statistics for a particular area"""

    # Combines SDG 15.3.1 input raster into a VRT and crop to the AOI
    indic_vrt = tempfile.NamedTemporaryFile(
        suffix="_ld_summary_inputs.vrt", delete=False
    ).name
    logger.info(f"Saving indicator VRT to: {indic_vrt}")
    gdal.BuildVRT(
        indic_vrt,
        [item.path for item in in_dfs],
        outputBounds=pixel_aligned_bbox,
        resolution="highest",
        resampleAlg=gdal.GRA_NearestNeighbour,
        separate=True,
    )

    error_message = ""
    logger.info(
        "Cutting input into tiles (and reprojecting) and saving to %s",
        output_layers_path,
    )

    if translate_worker_function:
        tiles = translate_worker_function(
            indic_vrt, str(output_layers_path), **translate_worker_params
        )
    else:
        # Intelligent tiling: adjust CPU count based on data size
        # Open the VRT to get dimensions
        temp_ds = gdal.Open(indic_vrt)
        width, height = temp_ds.RasterXSize, temp_ds.RasterYSize
        n_pixels = width * height
        del temp_ds

        # For very large datasets, reduce CPU count to avoid memory pressure
        # For small datasets, use fewer CPUs to reduce overhead
        if n_pixels > 50_000_000:  # 50M pixels
            effective_cpus = min(n_cpus, 8)  # Cap at 8  for very large data
            logger.info(
                f"Large dataset detected ({n_pixels} pixels), "
                f"using {effective_cpus} CPUs for tiling"
            )
        elif n_pixels < 1_000_000:  # 1M pixels
            effective_cpus = min(n_cpus, 2)  # Use fewer CPUs for small data
            logger.info(
                f"Small dataset detected ({n_pixels} pixels), "
                f"using {effective_cpus} CPUs for tiling"
            )
        else:
            effective_cpus = n_cpus

        translate_worker = workers.CutTiles(
            indic_vrt, effective_cpus, output_layers_path
        )
        tiles = translate_worker.work()

    logger.info("Calculating summaries for each tile")
    if tiles:
        total_tiles = len(tiles)
        processing_strategy = "parallel" if n_cpus > 1 else "sequential"
        logger.info(
            f"Created {total_tiles} tiles for processing using {processing_strategy} strategy"
        )

        inputs = [
            SummarizeTileInputs(
                in_file=tile,
                wkt_aoi=wkt_aoi,
                in_dfs=in_dfs,
                prod_mode=prod_mode,
                lc_legend_nesting=lc_legend_nesting,
                lc_trans_matrix=lc_trans_matrix,
                period_name=period_name,
                periods=periods,
                mask_worker_function=mask_worker_function,
                mask_worker_params=mask_worker_params,
                deg_worker_function=deg_worker_function,
                deg_worker_params=deg_worker_params,
            )
            for tile in tiles
        ]
        if n_cpus > 1:
            summary_table, output_paths = _aoi_process_multiprocess(inputs, n_cpus)
        else:
            summary_table, output_paths = _aoi_process_sequential(inputs)
    else:
        error_message = "Error reprojecting layers."
        summary_table = None
        tiles = None
        output_paths = None

    return summary_table, tiles, output_paths, error_message


def _compute_ld_summary_table(
    aoi,
    in_dfs,
    compute_bbs_from,
    prod_mode,
    output_job_path: Path,
    lc_legend_nesting: LCLegendNesting,
    lc_trans_matrix: LCTransitionDefinitionDeg,
    period_name: str,
    periods: dict,
    n_cpus: int,
) -> Tuple[models.SummaryTableLD, Path, Path]:
    """Computes summary table and the output tif file(s)"""

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    bbs = aoi.get_aligned_output_bounds(compute_bbs_from)
    assert len(wkt_aois) == len(bbs)

    if len(wkt_aois) > 1:
        output_name_pattern = f"{output_job_path.stem}" + "_{index}.tif"
    else:
        output_name_pattern = f"{output_job_path.stem}" + ".tif"

    stable_kwargs = {
        "in_dfs": in_dfs,
        "prod_mode": prod_mode,
        "lc_legend_nesting": lc_legend_nesting,
        "lc_trans_matrix": lc_trans_matrix,
        "period_name": period_name,
        "periods": periods,
        "n_cpus": n_cpus,
    }

    summary_tables = []
    reproj_paths = []
    output_paths = []

    for index, (wkt_aoi, pixel_aligned_bbox) in enumerate(zip(wkt_aois, bbs), start=1):
        base_output_path = output_job_path.parent / output_name_pattern.format(
            index=index
        )

        (
            this_summary_table,
            these_reproj_paths,
            these_output_paths,
            error_message,
        ) = _process_region(
            wkt_aoi=wkt_aoi,
            pixel_aligned_bbox=pixel_aligned_bbox,
            output_layers_path=base_output_path,
            **stable_kwargs,
        )

        if this_summary_table is None:
            raise RuntimeError(error_message)
        else:
            summary_tables.append(this_summary_table)
            reproj_paths.extend(these_reproj_paths)
            output_paths.extend(these_output_paths)

    summary_table = models.accumulate_summarytableld(summary_tables)

    if len(reproj_paths) > 1:
        reproj_path = output_job_path.parent / f"{output_job_path.stem}_tiles.vrt"
        gdal.BuildVRT(str(reproj_path), [str(p) for p in reproj_paths])
    else:
        reproj_path = reproj_paths[0]

    if len(output_paths) > 1:
        output_path = output_job_path.parent / f"{output_job_path.stem}_tiles_sdg.vrt"
        gdal.BuildVRT(str(output_path), [str(p) for p in output_paths])
    else:
        output_path = output_paths[0]

    return summary_table, output_path, reproj_path
