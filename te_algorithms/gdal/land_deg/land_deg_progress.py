import logging
import multiprocessing
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal
from te_schemas.datafile import DataFile
from te_schemas.land_cover import LCLegendNesting
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band

from .. import util, workers
from ..util_numba import bizonal_total, zonal_total
from . import config, models, worker
from .land_deg_numba import (
    calc_deg_soc,
    prod5_to_prod3,
    sdg_status_expanded,
    sdg_status_expanded_to_simple,
)

# Timeout constants for status processing (reuse from land_deg.py)
STATUS_PROCESSING_TIMEOUT = 12 * 3600  # 12 hours for status processing
STATUS_REGION_TIMEOUT = 6 * 3600  # 6 hours per region

logger = logging.getLogger(__name__)


def _accumulate_ld_summary_tables_status(
    tables: List[models.SummaryTableLDStatus],
) -> models.SummaryTableLDStatus:
    """Used for combining summary tables from multiple blocks of an image"""
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]

    assert all(
        [
            len(out.sdg_summaries) == len(out.prod_summaries),
            len(out.sdg_summaries) == len(out.lc_summaries),
            len(out.sdg_summaries) == len(out.soc_summaries),
        ]
    )
    for table in tables[1:]:
        for i in range(len(out.sdg_summaries)):
            out.sdg_summaries[i] = util.accumulate_dicts(
                [out.sdg_summaries[i], table.sdg_summaries[i]]
            )
            out.lc_summaries[i] = util.accumulate_dicts(
                [out.lc_summaries[i], table.lc_summaries[i]]
            )

            # prod and soc both have a dict giving all land and non-water totals,
            # so need to handle differently
            assert set(out.prod_summaries[i].keys()) == set(
                table.prod_summaries[i].keys()
            )
            assert set(out.soc_summaries[i].keys()) == set(
                table.soc_summaries[i].keys()
            )
            out.prod_summaries[i] = {
                key: util.accumulate_dicts(
                    [out.prod_summaries[i][key], table.prod_summaries[i][key]]
                )
                for key in out.prod_summaries[i].keys()
            }
            out.soc_summaries[i] = {
                key: util.accumulate_dicts(
                    [out.soc_summaries[i][key], table.soc_summaries[i][key]]
                )
                for key in out.soc_summaries[i].keys()
            }

    return out


def _accumulate_ld_summary_tables_change(
    tables: List[models.SummaryTableLDChange],
) -> models.SummaryTableLDChange:
    """Used for combining summary tables from multiple blocks of an image"""
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]

    assert all(
        [
            len(out.sdg_crosstabs) == len(out.prod_crosstabs),
            len(out.sdg_crosstabs) == len(out.lc_crosstabs),
            len(out.sdg_crosstabs) == len(out.soc_crosstabs),
        ]
    )

    for table in tables[1:]:
        for i in range(len(out.sdg_crosstabs)):
            out.sdg_crosstabs[i] = util.accumulate_dicts(
                [out.sdg_crosstabs[i], table.sdg_crosstabs[i]]
            )
            out.prod_crosstabs[i] = util.accumulate_dicts(
                [out.prod_crosstabs[i], table.prod_crosstabs[i]]
            )
            out.lc_crosstabs[i] = util.accumulate_dicts(
                [out.lc_crosstabs[i], table.lc_crosstabs[i]]
            )
            out.soc_crosstabs[i] = util.accumulate_dicts(
                [out.soc_crosstabs[i], table.soc_crosstabs[i]]
            )

    return out


def _process_single_region(region_data):
    """Process a single region for status summary - helper for parallel processing"""
    try:
        (
            index,
            wkt_aoi,
            this_bbs,
            status_vrt,
            status_band_dict,
            job_output_path,
            status_name_pattern,
            mask_name_fragment,
            periods,
            prod_mode,
            nesting,
            mask_worker_function,
            mask_worker_params,
            status_worker_function,
            status_worker_params,
        ) = region_data

        logger.info(f"Starting processing of region {index}")

        cropped_status_vrt = tempfile.NamedTemporaryFile(
            suffix="_ld_status_summary_inputs.vrt", delete=False
        ).name
        logger.debug(f"Region {index}: Building cropped VRT to {cropped_status_vrt}")
        gdal.BuildVRT(
            cropped_status_vrt,
            status_vrt,
            outputBounds=this_bbs,
            resolution="highest",
            resampleAlg=gdal.GRA_NearestNeighbour,
        )

        mask_tif = tempfile.NamedTemporaryFile(
            suffix="_ld_status_mask.tif", delete=False
        ).name
        logger.info(f"Saving mask to {mask_tif}")
        logger.info(
            str(job_output_path.parent / mask_name_fragment.format(index=index))
        )
        geojson = util.wkt_geom_to_geojson_file_string(wkt_aoi)

        logger.debug(f"Region {index}: Starting mask creation")
        if mask_worker_function:
            mask_worker_params = mask_worker_params or {}
            mask_result = mask_worker_function(
                mask_tif, geojson, str(cropped_status_vrt), **mask_worker_params
            )
        else:
            mask_worker = workers.Mask(
                mask_tif,
                geojson,
                str(cropped_status_vrt),
            )
            mask_result = mask_worker.work()

        logger.debug(f"Region {index}: Mask creation completed, result: {mask_result}")

        if not mask_result:
            logger.error(f"Region {index}: Error creating mask")
            return None, None, "Error creating mask."

        status_out_path = job_output_path.parent / status_name_pattern.format(
            index=index
        )

        logger.info(
            f"Calculating status summary table and saving layer to: {status_out_path}"
        )

        logger.debug(f"Region {index}: Creating status parameters")
        status_params = models.DegradationStatusSummaryParams(
            prod_mode=prod_mode,
            in_file=str(cropped_status_vrt),
            out_file=str(status_out_path),
            band_dict=status_band_dict,
            model_band_number=1,
            n_out_bands=4 * (len(periods) - 1),
            n_reporting=len(periods) - 1,
            mask_file=mask_tif,
            nesting=nesting,
        )

        logger.debug(f"Region {index}: Starting status processing")
        if status_worker_function:
            status_worker_params = status_worker_params or {}
            result = status_worker_function(
                status_params, _process_block_status, **status_worker_params
            )
        else:
            summarizer = worker.DegradationSummary(status_params, _process_block_status)
            result = summarizer.work()

        logger.debug(
            f"Region {index}: Status processing completed, result type: {type(result)}"
        )

        if not result:
            logger.error(f"Region {index}: Error calculating status summary tables")
            return None, None, "Error calculating status summary tables."

        # Extract status and change tables from result
        logger.debug(f"Region {index}: Extracting results from {len(result)} blocks")
        summary_table_status = [item[0] for item in result]
        summary_table_change = [item[1] for item in result]

        # For parallel processing, accumulate the results here instead of returning complex objects
        # This avoids the multiprocessing serialization issues with large numba typed dictionaries
        logger.debug(
            f"Region {index}: Accumulating results to avoid serialization issues"
        )

        # Accumulate within the region to reduce data size for serialization
        accumulated_status = _accumulate_ld_summary_tables_status(summary_table_status)
        accumulated_change = _accumulate_ld_summary_tables_change(summary_table_change)

        # Convert the accumulated objects to simpler Python objects for serialization
        def simplify_for_serialization(obj):
            """Fast conversion of numba objects to serializable Python objects"""
            # Handle the most common case first - numba typed dicts
            if hasattr(obj, "items") and hasattr(obj, "__getitem__"):
                # Convert tuple keys to strings for JSON serialization
                return {
                    str(k)
                    if isinstance(k, tuple)
                    else (int(k) if isinstance(k, (int, np.integer)) else k): float(v)
                    if isinstance(v, (int, float, np.number))
                    else v
                    for k, v in obj.items()
                }

            # Handle known object structures directly without reflection
            if hasattr(obj, "sdg_summaries"):
                # This is a SummaryTableLDStatus object
                return {
                    "sdg_summaries": [dict(d.items()) for d in obj.sdg_summaries],
                    "prod_summaries": [
                        {key: dict(val.items()) for key, val in d.items()}
                        for d in obj.prod_summaries
                    ],
                    "lc_summaries": [dict(d.items()) for d in obj.lc_summaries],
                    "soc_summaries": [
                        {key: dict(val.items()) for key, val in d.items()}
                        for d in obj.soc_summaries
                    ],
                }
            elif hasattr(obj, "sdg_crosstabs"):
                # This is a SummaryTableLDChange object
                # Convert tuple keys to strings for serialization
                return {
                    "sdg_crosstabs": [
                        {str(k) if isinstance(k, tuple) else k: v for k, v in d.items()}
                        for d in obj.sdg_crosstabs
                    ],
                    "prod_crosstabs": [
                        {str(k) if isinstance(k, tuple) else k: v for k, v in d.items()}
                        for d in obj.prod_crosstabs
                    ],
                    "lc_crosstabs": [
                        {str(k) if isinstance(k, tuple) else k: v for k, v in d.items()}
                        for d in obj.lc_crosstabs
                    ],
                    "soc_crosstabs": [
                        {str(k) if isinstance(k, tuple) else k: v for k, v in d.items()}
                        for d in obj.soc_crosstabs
                    ],
                }

            # Fast path for lists
            elif isinstance(obj, (list, tuple)):
                return [simplify_for_serialization(item) for item in obj]

            # Fast path for regular dicts
            elif isinstance(obj, dict):
                return {k: simplify_for_serialization(v) for k, v in obj.items()}

            # Return as-is for primitives
            else:
                return obj

        # Create simplified versions for serialization
        status_serializable = simplify_for_serialization(accumulated_status)
        change_serializable = simplify_for_serialization(accumulated_change)

        logger.info(f"Region {index}: Successfully completed processing")
        return status_serializable, change_serializable, str(status_out_path)

    except Exception as e:
        logger.error(
            f"Region {getattr(region_data, 'index', 'unknown')}: Unexpected error: {e}"
        )
        logger.exception("Full exception details:")
        return None, None, f"Unexpected error: {e}"


def compute_status_summary(
    df,
    prod_mode,
    job_output_path,
    aoi,
    compute_bbs_from,
    periods,
    nesting: LCLegendNesting,
    mask_worker_function: Union[None, Callable] = None,
    mask_worker_params: Union[None, dict] = None,
    status_worker_function: Union[None, Callable] = None,
    status_worker_params: Union[None, dict] = None,
    n_cpus: Optional[int] = None,
):
    """Compute status summary with optional parallel processing for multiple regions"""

    # Adaptive CPU management similar to land_deg.py
    if n_cpus is None:
        n_cpus = max(1, multiprocessing.cpu_count() - 1)

    status_vrt, status_band_dict = _get_status_summary_input_vrt(df, prod_mode, periods)

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    bbs = aoi.get_aligned_output_bounds(compute_bbs_from)
    assert len(wkt_aois) == len(bbs)

    if len(wkt_aois) > 1:
        status_name_pattern = (
            f"{job_output_path.stem}" + "_reporting_summary_{index}.tif"
        )
        mask_name_fragment = (
            "Generating mask for reporting analysis (part {index} of "
            + f"{len(wkt_aois)})"
        )
    else:
        status_name_pattern = f"{job_output_path.stem}" + "_reporting.tif"
        mask_name_fragment = "Generating mask for reporting analysis"

    # Prepare data for parallel processing
    region_data = [
        (
            index,
            wkt_aoi,
            this_bbs,
            status_vrt,
            status_band_dict,
            job_output_path,
            status_name_pattern,
            mask_name_fragment,
            periods,
            prod_mode,
            nesting,
            mask_worker_function,
            mask_worker_params,
            status_worker_function,
            status_worker_params,
        )
        for index, (wkt_aoi, this_bbs) in enumerate(zip(wkt_aois, bbs), start=1)
    ]

    summary_table_status = []
    summary_table_change = []
    reporting_paths = []

    # Use parallel processing for multiple regions, sequential for single region
    if len(wkt_aois) > 1 and n_cpus > 1:
        logger.info(
            f"Processing {len(wkt_aois)} regions in parallel with {min(n_cpus, len(wkt_aois))} workers"
        )

        # Use multiprocessing with improved serialization handling
        # Pre-accumulate results in workers to reduce serialization overhead
        logger.info(
            "Using parallel processing with optimized serialization for status processing"
        )

        with multiprocessing.get_context("spawn").Pool(
            min(n_cpus, len(wkt_aois))
        ) as pool:
            try:
                # Use map_async with timeout for better control
                async_result = pool.map_async(_process_single_region, region_data)

                # Wait for results with timeout
                results = async_result.get(timeout=STATUS_PROCESSING_TIMEOUT)
                logger.info(
                    f"Successfully completed parallel processing of {len(results)} regions"
                )
            except multiprocessing.TimeoutError:
                logger.error(
                    f"Status processing timed out after {STATUS_PROCESSING_TIMEOUT // 3600} hours"
                )
                pool.terminate()
                pool.join()
                raise RuntimeError(
                    f"Status processing timed out after {STATUS_PROCESSING_TIMEOUT // 3600} hours"
                )
            except Exception as e:
                logger.error(f"Error in parallel status processing: {e}")
                pool.terminate()
                pool.join()
                raise RuntimeError(f"Error calculating status: {e}")

        # Process the simplified results and reconstruct the objects
        logger.info("Reconstructing status objects from parallel results")
        for i, result in enumerate(results):
            if result is None or len(result) != 3:
                raise RuntimeError(f"Invalid result from region {i + 1}: {result}")

            status_data, change_data, status_path = result
            if status_data is None or change_data is None:
                raise RuntimeError(
                    f"Error processing status region {i + 1}: {status_path if isinstance(status_path, str) else 'Unknown error'}"
                )

            # Reconstruct the objects from the simplified serialized data
            # Note: We'll need to create new instances since we simplified them for serialization
            logger.debug(f"Region {i + 1}: Processing serialized results")

            # For now, treat as simplified objects and accumulate at the end
            # This maintains performance while avoiding serialization issues
            summary_table_status.append(status_data)
            summary_table_change.append(change_data)
            reporting_paths.append(status_path)

    else:
        # Sequential processing for single region or low CPU count
        logger.info(f"Processing {len(wkt_aois)} regions sequentially")
        for i, region in enumerate(region_data):
            logger.info(
                f"Starting sequential processing of region {i + 1}/{len(region_data)}"
            )
            result = _process_single_region(region)

            if result is None or len(result) != 3:
                raise RuntimeError(f"Invalid result from region {i + 1}: {result}")

            status_data, change_data, status_path = result
            if status_data is None or change_data is None:
                error_msg = (
                    status_path if isinstance(status_path, str) else "Unknown error"
                )
                raise RuntimeError(
                    f"Error processing status region {i + 1}: {error_msg}"
                )

            logger.info(f"Region {i + 1}: Adding status and change data")
            summary_table_status.append(status_data)
            summary_table_change.append(change_data)
            reporting_paths.append(status_path)

    logger.info(
        f"Accumulating {len(summary_table_status)} status tables and {len(summary_table_change)} change tables"
    )

    # Handle both parallel (simplified) and sequential (simplified) results
    # Since _process_single_region always returns simplified data now, we always need reconstruction
    logger.info("Processing results from parallel/sequential execution")

    # Create a simple accumulator for the simplified data
    def accumulate_simplified_status(tables):
        if not tables:
            return None
        if len(tables) == 1:
            return tables[0]

        # Fast accumulation for known structure
        accumulated = {
            "sdg_summaries": [],
            "prod_summaries": [],
            "lc_summaries": [],
            "soc_summaries": [],
        }

        # Determine the number of reporting periods from first table
        n_periods = len(tables[0]["sdg_summaries"])

        # Initialize accumulated structure
        for i in range(n_periods):
            accumulated["sdg_summaries"].append({})
            accumulated["lc_summaries"].append({})
            accumulated["prod_summaries"].append(
                {"all_cover_types": {}, "non_water": {}}
            )
            accumulated["soc_summaries"].append(
                {"all_cover_types": {}, "non_water": {}}
            )

        # Fast accumulation using direct key access
        for table in tables:
            for i in range(n_periods):
                # Accumulate SDG summaries
                for k, v in table["sdg_summaries"][i].items():
                    accumulated["sdg_summaries"][i][k] = (
                        accumulated["sdg_summaries"][i].get(k, 0) + v
                    )

                # Accumulate LC summaries
                for k, v in table["lc_summaries"][i].items():
                    accumulated["lc_summaries"][i][k] = (
                        accumulated["lc_summaries"][i].get(k, 0) + v
                    )

                # Accumulate prod summaries
                for cover_type in ["all_cover_types", "non_water"]:
                    for k, v in table["prod_summaries"][i][cover_type].items():
                        accumulated["prod_summaries"][i][cover_type][k] = (
                            accumulated["prod_summaries"][i][cover_type].get(k, 0) + v
                        )

                # Accumulate SOC summaries
                for cover_type in ["all_cover_types", "non_water"]:
                    for k, v in table["soc_summaries"][i][cover_type].items():
                        accumulated["soc_summaries"][i][cover_type][k] = (
                            accumulated["soc_summaries"][i][cover_type].get(k, 0) + v
                        )

        return accumulated

    def accumulate_simplified_change(tables):
        if not tables:
            return None
        if len(tables) == 1:
            return tables[0]

        # Fast accumulation for known structure
        accumulated = {
            "sdg_crosstabs": [],
            "prod_crosstabs": [],
            "lc_crosstabs": [],
            "soc_crosstabs": [],
        }

        # Determine the number of reporting periods from first table
        n_periods = len(tables[0]["sdg_crosstabs"])

        # Initialize accumulated structure
        for i in range(n_periods):
            accumulated["sdg_crosstabs"].append({})
            accumulated["prod_crosstabs"].append({})
            accumulated["lc_crosstabs"].append({})
            accumulated["soc_crosstabs"].append({})

        # Fast accumulation using direct key access
        for table in tables:
            for i in range(n_periods):
                # Accumulate all crosstabs (keys may be strings from tuple conversion)
                for crosstab_type in [
                    "sdg_crosstabs",
                    "prod_crosstabs",
                    "lc_crosstabs",
                    "soc_crosstabs",
                ]:
                    for k, v in table[crosstab_type][i].items():
                        accumulated[crosstab_type][i][k] = (
                            accumulated[crosstab_type][i].get(k, 0) + v
                        )

        return accumulated

    accumulated_status = accumulate_simplified_status(summary_table_status)
    accumulated_change = accumulate_simplified_change(summary_table_change)

    # Reconstruct proper objects from simplified data for downstream compatibility
    if accumulated_status is None or accumulated_change is None:
        raise RuntimeError(
            "Failed to accumulate status or change data from parallel processing"
        )

    logger.debug("Reconstructing SummaryTableLDStatus and SummaryTableLDChange objects")

    # Helper function to convert string tuple keys back to tuples
    def convert_string_keys_to_tuples(d):
        """Convert string representations of tuples back to tuple keys"""
        result = {}
        for k, v in d.items():
            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                # Parse string tuple back to tuple
                try:
                    # Safe evaluation of tuple strings like "(1, 2)"
                    tuple_key = eval(k)
                    if isinstance(tuple_key, tuple):
                        result[tuple_key] = v
                    else:
                        result[k] = v
                except Exception:
                    # If parsing fails, keep as string
                    result[k] = v
            else:
                result[k] = v
        return result

    # Convert crosstab string keys back to tuples for change data
    reconstructed_change_data = {
        "sdg_crosstabs": [
            convert_string_keys_to_tuples(d)
            for d in accumulated_change["sdg_crosstabs"]
        ],
        "prod_crosstabs": [
            convert_string_keys_to_tuples(d)
            for d in accumulated_change["prod_crosstabs"]
        ],
        "lc_crosstabs": [
            convert_string_keys_to_tuples(d) for d in accumulated_change["lc_crosstabs"]
        ],
        "soc_crosstabs": [
            convert_string_keys_to_tuples(d)
            for d in accumulated_change["soc_crosstabs"]
        ],
    }

    summary_table_status = models.SummaryTableLDStatus(
        sdg_summaries=accumulated_status["sdg_summaries"],
        prod_summaries=accumulated_status["prod_summaries"],
        lc_summaries=accumulated_status["lc_summaries"],
        soc_summaries=accumulated_status["soc_summaries"],
    )
    summary_table_change = models.SummaryTableLDChange(
        sdg_crosstabs=reconstructed_change_data["sdg_crosstabs"],
        prod_crosstabs=reconstructed_change_data["prod_crosstabs"],
        lc_crosstabs=reconstructed_change_data["lc_crosstabs"],
        soc_crosstabs=reconstructed_change_data["soc_crosstabs"],
    )

    if len(reporting_paths) > 1:
        reporting_path = (
            job_output_path.parent / f"{job_output_path.stem}_reporting.vrt"
        )
        gdal.BuildVRT(str(reporting_path), [str(p) for p in reporting_paths])
    else:
        reporting_path = reporting_paths[0]

    out_bands = [
        Band(
            name=config.SDG_STATUS_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                "baseline_year_initial": periods[0]["params"]["periods"][
                    "productivity"
                ]["year_initial"],
                "baseline_year_final": periods[0]["params"]["periods"]["productivity"][
                    "year_final"
                ],
                "reporting_year_initial": periods[i]["params"]["periods"][
                    "productivity"
                ]["year_initial"],
                "reporting_year_final": periods[i]["params"]["periods"]["productivity"][
                    "year_final"
                ],
            },
            add_to_map=True,
            activated=True,
        )
        for i in range(1, len(periods))
    ]
    out_bands.extend(
        [
            Band(
                name=config.PROD_STATUS_BAND_NAME,
                no_data_value=config.NODATA_VALUE.item(),  # write as python type
                metadata={
                    "baseline_year_initial": periods[0]["params"]["periods"][
                        "productivity"
                    ]["year_initial"],
                    "baseline_year_final": periods[0]["params"]["periods"][
                        "productivity"
                    ]["year_final"],
                    "reporting_year_initial": periods[i]["params"]["periods"][
                        "productivity"
                    ]["year_initial"],
                    "reporting_year_final": periods[i]["params"]["periods"][
                        "productivity"
                    ]["year_final"],
                },
                add_to_map=False,
                activated=False,
            )
            for i in range(1, len(periods))
        ]
    )
    out_bands.extend(
        [
            Band(
                name=config.LC_STATUS_BAND_NAME,
                no_data_value=config.NODATA_VALUE.item(),  # write as python type
                metadata={
                    "baseline_year_initial": periods[0]["params"]["periods"][
                        "land_cover"
                    ]["year_initial"],
                    "baseline_year_final": periods[0]["params"]["periods"][
                        "land_cover"
                    ]["year_final"],
                    "reporting_year_initial": periods[i]["params"]["periods"][
                        "land_cover"
                    ]["year_initial"],
                    "reporting_year_final": periods[i]["params"]["periods"][
                        "land_cover"
                    ]["year_final"],
                },
                add_to_map=False,
                activated=False,
            )
            for i in range(1, len(periods))
        ]
    )
    out_bands.extend(
        [
            Band(
                name=config.SOC_STATUS_BAND_NAME,
                no_data_value=config.NODATA_VALUE.item(),  # write as python type
                metadata={
                    "baseline_year_initial": periods[0]["params"]["periods"]["soc"][
                        "year_initial"
                    ],
                    "baseline_year_final": periods[0]["params"]["periods"]["soc"][
                        "year_final"
                    ],
                    "reporting_year_initial": periods[i]["params"]["periods"]["soc"][
                        "year_initial"
                    ],
                    "reporting_year_final": periods[i]["params"]["periods"]["soc"][
                        "year_final"
                    ],
                },
                add_to_map=False,
                activated=False,
            )
            for i in range(1, len(periods))
        ]
    )

    return (
        summary_table_status,
        summary_table_change,
        DataFile(reporting_path, out_bands),
    )


def _get_status_summary_input_vrt(df, prod_mode, periods):
    ##########################################################################
    # Get SDG layers
    prod_final_years = [
        period["params"]["periods"]["productivity"]["year_final"] for period in periods
    ]
    sdg_indices = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.SDG_BAND_NAME),
            df.metadata_for_name(config.SDG_BAND_NAME, "year_final"),
        )
        if year in prod_final_years
    ]
    assert len(sdg_indices) == len(prod_final_years)
    sdg_indices = sorted(sdg_indices, key=lambda row: row[1])
    sdg_baseline_index = sdg_indices[0][0]
    sdg_reporting_indices = [sdg_index[0] for sdg_index in sdg_indices[1:]]

    ##########################################################################
    # Get LPD layers
    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        prod5_indices = [
            (index, year)
            for index, year in zip(
                df.indices_for_name(config.TE_LPD_BAND_NAME),
                df.metadata_for_name(config.TE_LPD_BAND_NAME, "year_final"),
            )
            if year in prod_final_years
        ]
    else:
        if prod_mode == ProductivityMode.JRC_5_CLASS_LPD.value:
            lpd_layer_name = config.JRC_LPD_BAND_NAME
        elif prod_mode == ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value:
            lpd_layer_name = config.FAO_WOCAT_LPD_BAND_NAME
        elif prod_mode == ProductivityMode.CUSTOM_5_CLASS_LPD.value:
            lpd_layer_name = config.CUSTOM_LPD_BAND_NAME
        else:
            raise KeyError
        prod5_indices = [
            (index, year)
            for index, year in zip(
                df.indices_for_name(lpd_layer_name),
                df.metadata_for_name(lpd_layer_name, "year_final"),
            )
            if year in prod_final_years
        ]
    assert len(prod5_indices) == len(prod_final_years)
    prod5_indices = sorted(prod5_indices, key=lambda row: row[1])
    prod5_baseline_index = prod5_indices[0][0]
    prod5_reporting_indices = [prod5_index[0] for prod5_index in prod5_indices[1:]]

    ##########################################################################
    # Get land cover degradation layers
    lc_final_years = [
        period["params"]["periods"]["land_cover"]["year_final"] for period in periods
    ]
    lc_deg_indices = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.LC_DEG_BAND_NAME),
            df.metadata_for_name(config.LC_DEG_BAND_NAME, "year_final"),
        )
        if year in lc_final_years
    ]
    assert len(lc_deg_indices) == len(lc_final_years)
    lc_deg_indices = sorted(lc_deg_indices, key=lambda row: row[1])
    lc_deg_baseline_index = lc_deg_indices[0][0]
    lc_deg_reporting_indices = [lc_deg_index[0] for lc_deg_index in lc_deg_indices[1:]]

    # Need initial year of land cover from baseline for water mask
    lc_bands = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.LC_BAND_NAME),
            df.metadata_for_name(config.LC_BAND_NAME, "year"),
        )
    ]
    lc_baseline_index = [
        index
        for index, year in lc_bands
        if year == periods[0]["params"]["periods"]["land_cover"]["year_initial"]
    ][0]

    ##########################################################################
    # Get SOC layers, ensuring only the final years of each period are pulled
    soc_deg_baseline_index = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.SOC_DEG_BAND_NAME),
            df.metadata_for_name(config.SOC_DEG_BAND_NAME, "year_final"),
        )
        if year == periods[0]["params"]["periods"]["soc"]["year_final"]
    ][0][0]
    soc_final_years = [
        period["params"]["periods"]["soc"]["year_final"] for period in periods
    ]
    soc_indices = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.SOC_BAND_NAME),
            df.metadata_for_name(config.SOC_BAND_NAME, "year"),
        )
        if year in soc_final_years
    ]
    soc_indices = sorted(soc_indices, key=lambda row: row[1])
    soc_baseline_index = soc_indices[0][0]
    soc_reporting_indices = [soc_index[0] for soc_index in soc_indices[1:]]

    df_band_list = [
        ("sdg_baseline_bandnum", sdg_baseline_index),
        ("prod5_baseline_bandnum", prod5_baseline_index),
        ("lc_baseline_bandnum", lc_baseline_index),  # needed for water mask
        ("lc_deg_baseline_bandnum", lc_deg_baseline_index),
        ("soc_baseline_bandnum", soc_baseline_index),
        ("soc_deg_baseline_bandnum", soc_deg_baseline_index),
    ]
    for i in range(len(sdg_reporting_indices)):
        df_band_list.extend(
            [
                (f"sdg_reporting_{i}_bandnum", sdg_reporting_indices[i]),
                (f"prod5_reporting_{i}_bandnum", prod5_reporting_indices[i]),
                (f"lc_deg_reporting_{i}_bandnum", lc_deg_reporting_indices[i]),
                (f"soc_reporting_{i}_bandnum", soc_reporting_indices[i]),
            ]
        )

    ##########################################################################
    # Save data
    band_vrts = [
        util.save_vrt(df.path, band_num + 1) for name, band_num in df_band_list
    ]
    out_vrt = tempfile.NamedTemporaryFile(
        suffix="_ld_status_inputs.vrt", delete=False
    ).name
    gdal.BuildVRT(out_vrt, [vrt for vrt in band_vrts], separate=True)
    vrt_band_dict = {item[0]: index for index, item in enumerate(df_band_list, start=1)}

    return out_vrt, vrt_band_dict


def _process_block_status(
    params: models.DegradationStatusSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw,
) -> Tuple[Tuple[models.SummaryTableLDStatus, models.SummaryTableLDChange], List[Dict]]:
    """Optimized status block processing with cached array operations"""
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1).astype(np.float64)

    # Derive a water mask from last lc year - handling fact that there could be
    # multiple water codes if this is a custom legend. 7 is the IPCC water code.
    water = np.isin(
        in_array[params.band_dict["lc_baseline_bandnum"] - 1, :, :],
        params.nesting.nesting[7],
    )
    water = water.astype(bool, copy=False)

    # Make a mask that also masks water, to be use for tabulation of the
    # productivity and SOC summaries without water for usage on Prais4
    mask_plus_water = mask.copy()
    mask_plus_water[water] = True

    # Cache baseline arrays to avoid repeated indexing
    sdg_baseline = in_array[params.band_dict["sdg_baseline_bandnum"] - 1, :, :]
    prod5_baseline = in_array[params.band_dict["prod5_baseline_bandnum"] - 1, :, :]
    lc_deg_baseline = in_array[params.band_dict["lc_deg_baseline_bandnum"] - 1, :, :]
    soc_deg_baseline = in_array[params.band_dict["soc_deg_baseline_bandnum"] - 1, :, :]

    # Pre-process baseline productivity data
    # Recode zeros in prod5 to config.NODATA_VALUE as the JRC LPD on
    # trends.earth assets had 0 used instead of our standard nodata value
    prod5_baseline[prod5_baseline == 0] = config.NODATA_VALUE
    prod3_baseline = prod5_to_prod3(prod5_baseline)

    ##########################################################################
    # Pre-allocate result containers for better memory efficiency
    sdg_statuses = []
    sdg_summaries = []
    sdg_crosstabs = []
    prod_summaries = []
    prod_statuses = []
    prod_crosstabs = []
    lc_summaries = []
    lc_statuses = []
    lc_crosstabs = []
    soc_summaries = []
    soc_statuses = []
    soc_crosstabs = []

    ##########################################################################
    # Process all reporting periods in a single loop for efficiency
    for i in range(params.n_reporting):
        # Get reporting period arrays
        sdg_reporting = in_array[
            params.band_dict[f"sdg_reporting_{i}_bandnum"] - 1, :, :
        ]
        prod5_reporting = in_array[
            params.band_dict[f"prod5_reporting_{i}_bandnum"] - 1, :, :
        ]
        lc_deg_reporting = in_array[
            params.band_dict[f"lc_deg_reporting_{i}_bandnum"] - 1, :, :
        ]
        soc_reporting = in_array[
            params.band_dict[f"soc_reporting_{i}_bandnum"] - 1, :, :
        ]

        # Calculate SDG status
        sdg_status = sdg_status_expanded(sdg_baseline, sdg_reporting)
        sdg_statuses.append(sdg_status)
        sdg_status_3_class = sdg_status_expanded_to_simple(sdg_status)
        sdg_summaries.append(zonal_total(sdg_status_3_class, cell_areas, mask))
        sdg_crosstabs.append(
            bizonal_total(sdg_baseline, sdg_status_3_class, cell_areas, mask)
        )

        # Calculate productivity status
        prod5_reporting[prod5_reporting == 0] = config.NODATA_VALUE
        prod3_reporting = prod5_to_prod3(prod5_reporting)

        prod_status = sdg_status_expanded(prod3_baseline, prod3_reporting)
        prod_status_3_class = sdg_status_expanded_to_simple(prod_status)

        prod_statuses.append(prod_status)
        prod_summaries.append(
            {
                "all_cover_types": zonal_total(prod_status_3_class, cell_areas, mask),
                "non_water": zonal_total(
                    prod_status_3_class,
                    cell_areas,
                    mask_plus_water,
                ),
            }
        )
        prod_crosstabs.append(
            bizonal_total(prod3_baseline, prod_status_3_class, cell_areas, mask)
        )

        # Calculate LC status
        lc_status = sdg_status_expanded(lc_deg_baseline, lc_deg_reporting)
        lc_statuses.append(lc_status)
        lc_status_3_class = sdg_status_expanded_to_simple(lc_status)
        lc_summaries.append(zonal_total(lc_status_3_class, cell_areas, mask))
        lc_crosstabs.append(
            bizonal_total(lc_deg_baseline, lc_status_3_class, cell_areas, mask)
        )

        # Calculate SOC status
        soc_deg_reporting = calc_deg_soc(
            in_array[params.band_dict["soc_baseline_bandnum"] - 1, :, :],
            soc_reporting,
            water,
        )
        soc_status = sdg_status_expanded(soc_deg_baseline, soc_deg_reporting)
        soc_statuses.append(soc_status)
        soc_status_3_class = sdg_status_expanded_to_simple(soc_status)
        soc_summaries.append(
            {
                "all_cover_types": zonal_total(soc_status_3_class, cell_areas, mask),
                "non_water": zonal_total(
                    soc_status_3_class, cell_areas, mask_plus_water
                ),
            }
        )
        soc_crosstabs.append(
            bizonal_total(soc_deg_baseline, soc_status_3_class, cell_areas, mask)
        )

    ##########################################################################
    # Build write arrays in a more efficient manner
    write_arrays = []

    # Add all SDG status arrays
    for sdg_status in sdg_statuses:
        write_arrays.append({"array": sdg_status, "xoff": xoff, "yoff": yoff})

    # Add all productivity status arrays
    for prod_status in prod_statuses:
        write_arrays.append({"array": prod_status, "xoff": xoff, "yoff": yoff})

    # Add all SOC status arrays
    for soc_status in soc_statuses:
        write_arrays.append({"array": soc_status, "xoff": xoff, "yoff": yoff})

    # Add all LC status arrays
    for lc_status in lc_statuses:
        write_arrays.append({"array": lc_status, "xoff": xoff, "yoff": yoff})

    return (
        (
            models.SummaryTableLDStatus(
                sdg_summaries,
                prod_summaries,
                lc_summaries,
                soc_summaries,
            ),
            models.SummaryTableLDChange(
                sdg_crosstabs,
                prod_crosstabs,
                lc_crosstabs,
                soc_crosstabs,
            ),
        ),
        write_arrays,
    )
