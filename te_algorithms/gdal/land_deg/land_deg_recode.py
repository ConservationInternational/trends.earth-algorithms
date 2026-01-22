import dataclasses
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Union

import numpy as np
from osgeo import gdal
from te_schemas import reporting
from te_schemas.aoi import AOI
from te_schemas.datafile import DataFile
from te_schemas.error_recode import ErrorRecodePolygons
from te_schemas.jobs import Job
from te_schemas.results import (
    URI,
    Band,
    DataType,
    JsonResults,
    Raster,
    RasterFileType,
    RasterResults,
    TiledRaster,
)

from .. import workers
from ..util import accumulate_dicts, save_vrt, wkt_geom_to_geojson_file_string
from ..util_numba import zonal_total
from . import config
from .land_deg_numba import recode_indicator_errors, sdg_status_expanded
from .models import DegradationErrorRecodeSummaryParams, SummaryTableLDErrorRecode
from .worker import DegradationSummary

logger = logging.getLogger(__name__)


def rasterize_error_recode(
    out_file: Path,
    model_file: Path,
    geojson: ErrorRecodePolygons,
) -> None:
    # Convert layer into an integer code so that all three recode_deg_to
    # options can be encoded within a single tiff

    recode_to_trans_code = geojson.recode_to_trans_code_dict
    error_recode_dict = ErrorRecodePolygons.Schema().dump(geojson)

    for feat in error_recode_dict["features"]:
        feat["properties"]["error_recode"] = recode_to_trans_code[
            (
                feat["properties"]["recode_deg_to"],
                feat["properties"]["recode_stable_to"],
                feat["properties"]["recode_imp_to"],
            )
        ]

        # Convert periods_affected to a bitmask for rasterization
        # bit 1 = baseline, bit 2 = report_1, bit 4 = report_2
        periods_affected = feat["properties"].get("periods_affected")
        if not periods_affected:
            raise ValueError(
                f"periods_affected is required and cannot be empty for feature with "
                f"recode_deg_to={feat['properties'].get('recode_deg_to')}, "
                f"recode_stable_to={feat['properties'].get('recode_stable_to')}, "
                f"recode_imp_to={feat['properties'].get('recode_imp_to')}"
            )

        periods_mask = 0
        if "baseline" in periods_affected:
            periods_mask |= 1
        if "report_1" in periods_affected:
            periods_mask |= 2
        if "report_2" in periods_affected:
            periods_mask |= 4
        feat["properties"]["periods_mask"] = periods_mask

    # TODO: Assumes WGS84 for now
    # Create temporary files for each band
    temp_error_recode = tempfile.NamedTemporaryFile(
        suffix="_temp_error_recode.tif", delete=False
    ).name
    temp_periods_mask = tempfile.NamedTemporaryFile(
        suffix="_temp_periods_mask.tif", delete=False
    ).name

    try:
        rasterize_worker_error = workers.Rasterize(
            Path(temp_error_recode),
            model_file,
            error_recode_dict,
            "error_recode",
        )
        rasterize_worker_error.work()

        rasterize_worker_periods = workers.Rasterize(
            Path(temp_periods_mask),
            model_file,
            error_recode_dict,
            "periods_mask",
        )
        rasterize_worker_periods.work()

        # Combine both single-band rasters into one multi-band VRT
        # Note: Do NOT delete the temp files here as the VRT references them
        gdal.BuildVRT(
            str(out_file), [temp_error_recode, temp_periods_mask], separate=True
        )

    except Exception:
        # Only clean up temporary files if there was an error
        if os.path.exists(temp_error_recode):
            os.remove(temp_error_recode)
        if os.path.exists(temp_periods_mask):
            os.remove(temp_periods_mask)
        raise


def _process_block(
    params: DegradationErrorRecodeSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw,
) -> Tuple[SummaryTableLDErrorRecode, List[Dict]]:
    """
    Process a block of data to recode multiple SDG bands and compute zonal statistics.

    Returns summary statistics for baseline and reporting periods, plus arrays to write.
    """
    # Extract band arrays
    baseline_array = None
    reporting_1_array = None
    reporting_2_array = None
    recode_array = in_array[params.band_dict["recode_bandnum"] - 1, :, :]
    periods_mask_array = in_array[params.band_dict["periods_mask_bandnum"] - 1, :, :]

    # Get the available bands based on parameters
    if params.baseline_band_num is not None:
        baseline_array = in_array[params.band_dict["baseline_bandnum"] - 1, :, :]
    if params.report_1_band_num is not None:
        reporting_1_array = in_array[params.band_dict["reporting_1_bandnum"] - 1, :, :]
    if params.report_2_band_num is not None:
        reporting_2_array = in_array[params.band_dict["reporting_2_bandnum"] - 1, :, :]

    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1)

    # Get the recode parameters
    codes, deg_to, stable_to, imp_to = params.trans_code_lists

    write_arrays = []
    summaries = {}

    # Process each band that exists
    if baseline_array is not None:
        # Call the error recoding function with actual or None arrays
        baseline_recoded, reporting_1_recoded, reporting_2_recoded = (
            recode_indicator_errors(
                baseline_array,
                reporting_1_array,  # Can be None
                reporting_2_array,  # Can be None
                recode_array,
                periods_mask_array,
                np.array(codes, dtype=np.int16),
                np.array(deg_to, dtype=np.int16),
                np.array(stable_to, dtype=np.int16),
                np.array(imp_to, dtype=np.int16),
            )
        )

        # Compute zonal statistics for baseline
        baseline_summary = zonal_total(baseline_recoded, cell_areas, mask)
        summaries["baseline_summary"] = baseline_summary

        # Add baseline to write arrays
        write_arrays.append(
            {
                "array": baseline_recoded,
                "xoff": xoff,
                "yoff": yoff,
                "band_name": "baseline",
            }
        )

        # Process reporting periods if they exist
        if params.report_1_band_num is not None and reporting_1_recoded is not None:
            # Calculate 7-class status map for reporting period 1
            reporting_1_status = sdg_status_expanded(
                baseline_recoded, reporting_1_recoded
            )

            report_1_summary = zonal_total(reporting_1_status, cell_areas, mask)
            summaries["report_1_summary"] = report_1_summary

            # Default output: 7-class status map
            write_arrays.append(
                {
                    "array": reporting_1_status,
                    "xoff": xoff,
                    "yoff": yoff,
                    "band_name": "reporting_1_status",
                }
            )

            # Optionally write out the raw reporting period layer
            if params.write_reporting_sdg_tifs:
                write_arrays.append(
                    {
                        "array": reporting_1_recoded,
                        "xoff": xoff,
                        "yoff": yoff,
                        "band_name": "reporting_1_raw",
                    }
                )

        if params.report_2_band_num is not None and reporting_2_recoded is not None:
            # Calculate 7-class status map for reporting period 2
            reporting_2_status = sdg_status_expanded(
                baseline_recoded, reporting_2_recoded
            )

            report_2_summary = zonal_total(reporting_2_status, cell_areas, mask)
            summaries["report_2_summary"] = report_2_summary

            # Default output: 7-class status map
            write_arrays.append(
                {
                    "array": reporting_2_status,
                    "xoff": xoff,
                    "yoff": yoff,
                    "band_name": "reporting_2_status",
                }
            )

            # Optionally write out the raw reporting period layer
            if params.write_reporting_sdg_tifs:
                write_arrays.append(
                    {
                        "array": reporting_2_recoded,
                        "xoff": xoff,
                        "yoff": yoff,
                        "band_name": "reporting_2_raw",
                    }
                )

    # Always include the recode array in output
    write_arrays.append(
        {"array": recode_array, "xoff": xoff, "yoff": yoff, "band_name": "recode"}
    )

    # Create summary table
    summary_table = SummaryTableLDErrorRecode(
        baseline_summary=summaries.get("baseline_summary", {}),
        report_1_summary=summaries.get("report_1_summary"),
        report_2_summary=summaries.get("report_2_summary"),
    )

    return summary_table, write_arrays


def _accumulate_summary_tables(
    tables: List[SummaryTableLDErrorRecode],
) -> SummaryTableLDErrorRecode:
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]

        for table in tables[1:]:
            out.baseline_summary = accumulate_dicts(
                [out.baseline_summary, table.baseline_summary]
            )

            if out.report_1_summary is not None and table.report_1_summary is not None:
                out.report_1_summary = accumulate_dicts(
                    [out.report_1_summary, table.report_1_summary]
                )
            elif table.report_1_summary is not None:
                out.report_1_summary = table.report_1_summary

            if out.report_2_summary is not None and table.report_2_summary is not None:
                out.report_2_summary = accumulate_dicts(
                    [out.report_2_summary, table.report_2_summary]
                )
            elif table.report_2_summary is not None:
                out.report_2_summary = table.report_2_summary

        return out


def _get_error_recode_input_vrt(
    baseline_df=None, reporting_1_df=None, reporting_2_df=None, error_df=None
):
    """
    Create a VRT with multiple input bands for error recoding.

    Parameters:
    - baseline_df: DataFile for baseline SDG layer (optional)
    - reporting_1_df: DataFile for reporting period 1 SDG layer (optional)
    - reporting_2_df: DataFile for reporting period 2 SDG layer (optional)
    - error_df: DataFile containing error recode and periods_affected bands (required)

    Returns:
    - out_vrt: Path to the created VRT file
    - vrt_band_dict: Dictionary mapping band names to band numbers in the VRT
    """
    band_vrts = []
    df_band_list = []
    band_counter = 1

    # Add baseline if available
    if baseline_df is not None:
        band_vrts.append(save_vrt(baseline_df.path, 1))
        df_band_list.append(("baseline_bandnum", band_counter))
        band_counter += 1

    # Add reporting period 1 if available
    if reporting_1_df is not None:
        band_vrts.append(save_vrt(reporting_1_df.path, 1))
        df_band_list.append(("reporting_1_bandnum", band_counter))
        band_counter += 1

    # Add reporting period 2 if available
    if reporting_2_df is not None:
        band_vrts.append(save_vrt(reporting_2_df.path, 1))
        df_band_list.append(("reporting_2_bandnum", band_counter))
        band_counter += 1

    # Add error recode band (required)
    if error_df is not None:
        band_vrts.append(
            save_vrt(
                error_df.path,
                error_df.index_for_name(config.ERROR_RECODE_BAND_NAME) + 1,
            )
        )
        df_band_list.append(("recode_bandnum", band_counter))
        band_counter += 1

        # Add periods_affected band if it exists
        try:
            periods_band_index = error_df.index_for_name("periods_affected")
            band_vrts.append(save_vrt(error_df.path, periods_band_index + 1))
            df_band_list.append(("periods_mask_bandnum", band_counter))
            band_counter += 1
        except (ValueError, AttributeError):
            # periods_affected band doesn't exist, create a dummy band
            # This ensures backward compatibility
            pass

    out_vrt = tempfile.NamedTemporaryFile(
        suffix="_error_recode_inputs.vrt", delete=False
    ).name
    gdal.BuildVRT(out_vrt, [vrt for vrt in band_vrts], separate=True)

    vrt_band_dict = {item[0]: item[1] for item in df_band_list}

    return out_vrt, vrt_band_dict


def _prepare_df(path, band_str, band_index) -> DataFile:
    band = Band(**band_str)

    return DataFile(path=Path(save_vrt(path, band_index)), bands=[band])


def _prepare_error_recode_df(path, band_str) -> DataFile:
    """Special version of _prepare_df for error recode files with multiple bands."""
    # Create the primary band (Error recode) - always band 1
    primary_band = Band(**band_str)

    # Open the raster to check for additional bands
    ds = gdal.Open(str(path))
    bands = [primary_band]

    if ds and ds.RasterCount >= 2:
        # Assume the second band is the periods_affected band
        # (since we create it that way in rasterize_error_recode)
        periods_band = Band(
            name="periods_affected",
            metadata={},
            no_data_value=primary_band.no_data_value,
            activated=True,
        )
        bands.append(periods_band)

    if ds:
        ds = None  # Close the dataset

    return DataFile(path=Path(path), bands=bands)


def get_serialized_results(st, layer_name, periods_to_output=None):
    """Get serialized results for specified periods.

    Args:
        st: Summary table containing baseline_summary, report_1_summary, report_2_summary
        layer_name: Name for the output layer
        periods_to_output: List of periods to include in output. If None, includes all.
                          Valid values: 'baseline', 'report_1', 'report_2'
    """
    # Default to all periods if not specified
    if periods_to_output is None:
        periods_to_output = ["baseline", "report_1", "report_2"]

    def create_area_list(summary_dict, title_suffix=""):
        return reporting.AreaList(
            f"SDG Indicator 15.3.1{title_suffix}",
            "sq km",
            [
                reporting.Area("Improved", summary_dict.get(1, 0.0)),
                reporting.Area("Stable", summary_dict.get(0, 0.0)),
                reporting.Area("Degraded", summary_dict.get(-1, 0.0)),
                reporting.Area(
                    "No data", summary_dict.get(int(config.NODATA_VALUE), 0)
                ),
            ],
        )

    def create_status_area_list(status_summary_dict, title_suffix=""):
        """Create area list from 7-class status summary."""
        return reporting.AreaList(
            f"SDG Indicator 15.3.1{title_suffix}",
            "sq km",
            [
                # Degraded classes (1, 2, 3)
                reporting.Area(
                    "Persistent degradation", status_summary_dict.get(1, 0.0)
                ),
                reporting.Area("Recent degradation", status_summary_dict.get(2, 0.0)),
                reporting.Area("Baseline degradation", status_summary_dict.get(3, 0.0)),
                # Stable class (4)
                reporting.Area("Stability", status_summary_dict.get(4, 0.0)),
                # Improved classes (5, 6, 7)
                reporting.Area("Baseline improvement", status_summary_dict.get(5, 0.0)),
                reporting.Area("Recent improvement", status_summary_dict.get(6, 0.0)),
                reporting.Area(
                    "Persistent improvement", status_summary_dict.get(7, 0.0)
                ),
                # No data
                reporting.Area(
                    "No data", status_summary_dict.get(int(config.NODATA_VALUE), 0)
                ),
            ],
        )

    reports = []
    baseline_summary = None

    # Add baseline summary only if baseline is in periods_to_output
    if "baseline" in periods_to_output:
        baseline_summary = create_area_list(st.baseline_summary, " - Baseline")
        reports.append(baseline_summary)

    # Add reporting periods if they exist and are in periods_to_output
    if "report_1" in periods_to_output and st.report_1_summary is not None:
        report_1_summary = create_status_area_list(
            st.report_1_summary, " - Reporting Period 1 Status"
        )
        reports.append(report_1_summary)

    if "report_2" in periods_to_output and st.report_2_summary is not None:
        report_2_summary = create_status_area_list(
            st.report_2_summary, " - Reporting Period 2 Status"
        )
        reports.append(report_2_summary)

    # Create the main report - use baseline if available, otherwise first available summary
    if baseline_summary is not None:
        sdg_report = reporting.SDG15Report(summary=baseline_summary)
        additional_reports = reports[1:] if len(reports) > 1 else []
    elif len(reports) > 0:
        # No baseline in output, use first available report as primary
        sdg_report = reporting.SDG15Report(summary=reports[0])
        additional_reports = reports[1:] if len(reports) > 1 else []
    else:
        # No reports at all - create empty summary
        empty_summary = create_area_list({}, " - No Data")
        sdg_report = reporting.SDG15Report(summary=empty_summary)
        additional_reports = []

    # Add additional summaries to the serialized output
    result_dict = dataclasses.asdict(sdg_report)
    if len(additional_reports) > 0:
        result_dict["additional_summaries"] = [
            dataclasses.asdict(report) for report in additional_reports
        ]

    return result_dict


def recode_errors(params) -> Job:
    logger.debug("Entering recode_errors")
    aoi = AOI(params["aoi"])
    job_output_path = Path(params["output_path"])
    baseline_df = _prepare_df(
        params["layer_baseline_band_path"],
        params["layer_baseline_band"],
        params["layer_baseline_band_index"],
    )

    # Prepare reporting period 1 data if provided
    reporting_1_df = None
    if "layer_reporting_1_band_path" in params:
        reporting_1_df = _prepare_df(
            params["layer_reporting_1_band_path"],
            params["layer_reporting_1_band"],
            params["layer_reporting_1_band_index"],
        )

    # Prepare reporting period 2 data if provided
    reporting_2_df = None
    if "layer_reporting_2_band_path" in params:
        reporting_2_df = _prepare_df(
            params["layer_reporting_2_band_path"],
            params["layer_reporting_2_band"],
            params["layer_reporting_2_band_index"],
        )

    error_recode_df = _prepare_error_recode_df(
        params["layer_error_recode_path"],
        params["layer_error_recode_band"],
    )
    logger.debug("Loading error polygons")
    error_polygons_data = params["error_polygons"]
    error_polygons = ErrorRecodePolygons(
        features=error_polygons_data.get("features", []),
        name=error_polygons_data.get("name"),
        crs=error_polygons_data.get("crs"),
        type=error_polygons_data.get("type", "FeatureCollection"),
    )

    baseline_band_data = params["layer_baseline_band"]
    baseline_band = Band(
        name=baseline_band_data["name"],
        no_data_value=baseline_band_data.get("no_data_value"),
        metadata=baseline_band_data.get("metadata", {}),
        add_to_map=baseline_band_data.get("add_to_map", False),
        activated=baseline_band_data.get("activated", False),
    )

    # Get periods to include in output (defaults to all if not specified)
    periods_to_output = params.get(
        "periods_to_output", ["baseline", "report_1", "report_2"]
    )

    logger.debug("Running _compute_error_recode")
    summary_table, error_recode_paths = _compute_error_recode(
        baseline_df=baseline_df,
        reporting_1_df=reporting_1_df,
        reporting_2_df=reporting_2_df,
        error_recode_df=error_recode_df,
        error_polygons=error_polygons,
        job_output_path=job_output_path.parent / f"{job_output_path.stem}.json",
        aoi=aoi,
    )

    if params["write_tifs"]:
        # Create bands dynamically based on what was processed and what should be in output
        out_bands = []

        # Include baseline band only if baseline is in periods_to_output
        if "baseline" in periods_to_output:
            out_bands.append(
                Band(
                    name=f"{baseline_band.name} - Baseline",
                    no_data_value=int(config.NODATA_VALUE),
                    metadata=params["metadata"],  # copy metadata from input job
                    add_to_map=True,
                    activated=True,
                )
            )

        # Add reporting period 1 status band if it exists and is in periods_to_output
        if reporting_1_df is not None and "report_1" in periods_to_output:
            out_bands.append(
                Band(
                    name=f"{baseline_band.name} - Reporting Period 1 Status",
                    no_data_value=int(config.NODATA_VALUE),
                    metadata=params["metadata"],
                    add_to_map=True,
                    activated=True,
                )
            )

            # Add raw reporting period 1 band if write_reporting_sdg_tifs is enabled
            if params.get("write_reporting_sdg_tifs", False):
                out_bands.append(
                    Band(
                        name=f"{baseline_band.name} - Reporting Period 1 Raw",
                        no_data_value=int(config.NODATA_VALUE),
                        metadata=params["metadata"],
                        add_to_map=False,
                        activated=False,
                    )
                )

        # Add reporting period 2 status band if it exists and is in periods_to_output
        if reporting_2_df is not None and "report_2" in periods_to_output:
            out_bands.append(
                Band(
                    name=f"{baseline_band.name} - Reporting Period 2 Status",
                    no_data_value=int(config.NODATA_VALUE),
                    metadata=params["metadata"],
                    add_to_map=True,
                    activated=True,
                )
            )

            # Add raw reporting period 2 band if write_reporting_sdg_tifs is enabled
            if params.get("write_reporting_sdg_tifs", False):
                out_bands.append(
                    Band(
                        name=f"{baseline_band.name} - Reporting Period 2 Raw",
                        no_data_value=int(config.NODATA_VALUE),
                        metadata=params["metadata"],
                        add_to_map=False,
                        activated=False,
                    )
                )

        # Always include the error recode band
        out_bands.append(
            Band(
                name=config.ERROR_RECODE_BAND_NAME,
                no_data_value=int(config.NODATA_VALUE),
                metadata=params["metadata"],  # copy metadata from input job
                add_to_map=False,
                activated=False,
            )
        )

        if len(error_recode_paths) > 1:
            error_recode_vrt = (
                job_output_path.parent / f"{job_output_path.stem}_error_recode.vrt"
            )
            gdal.BuildVRT(str(error_recode_vrt), [str(p) for p in error_recode_paths])
            rasters: Dict[str, Union[Raster, TiledRaster]] = {
                DataType.INT16.value: TiledRaster(
                    tile_uris=[URI(uri=p) for p in error_recode_paths],
                    uri=URI(uri=error_recode_vrt),
                    bands=out_bands,
                    datatype=DataType.INT16,
                    filetype=RasterFileType.COG,
                )
            }
            main_uri = URI(uri=error_recode_vrt)
        else:
            rasters: Dict[str, Union[Raster, TiledRaster]] = {
                DataType.INT16.value: Raster(
                    uri=URI(uri=error_recode_paths[0]),
                    bands=out_bands,
                    datatype=DataType.INT16,
                    filetype=RasterFileType.COG,
                )
            }
            main_uri = URI(uri=error_recode_paths[0])

        results = RasterResults(
            name=params["layer_baseline_band"]["name"],
            uri=main_uri,
            rasters=rasters,
            data=get_serialized_results(
                summary_table,
                params["layer_baseline_band"]["name"] + " recode",
                periods_to_output=periods_to_output,
            ),
        )

    else:
        results = JsonResults(
            name=params["layer_baseline_band"]["name"],
            data=get_serialized_results(
                summary_table,
                params["layer_baseline_band"]["name"] + " recode",
                periods_to_output=periods_to_output,
            ),
        )

    return results


def _compute_error_recode(
    baseline_df=None,
    reporting_1_df=None,
    reporting_2_df=None,
    error_recode_df=None,
    error_polygons=None,
    job_output_path=None,
    aoi=None,
    write_reporting_sdg_tifs=False,
    mask_worker_function: Optional[Callable] = None,
    mask_worker_params: Optional[dict] = None,
    error_recode_worker_function: Optional[Callable] = None,
    error_recode_worker_params: Optional[dict] = None,
):
    in_vrt, band_dict = _get_error_recode_input_vrt(
        baseline_df, reporting_1_df, reporting_2_df, error_recode_df
    )

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    # Use the first available datafile for getting bounds
    model_df = baseline_df or reporting_1_df or reporting_2_df or error_recode_df
    bbs = aoi.get_aligned_output_bounds(model_df.path)

    error_recode_name_pattern = {
        1: f"{job_output_path.stem}" + "_error_recode.tif",
        2: f"{job_output_path.stem}" + "_error_recode_{index}.tif",
    }[len(bbs)]
    mask_name_fragment = {
        1: "Generating mask for error recoding",
        2: "Generating mask for error recoding (part {index} of 2)",
    }[len(wkt_aois)]

    error_recode_tables = []
    error_recode_paths = []
    error_message = None

    for index, (wkt_aoi, this_bbs) in enumerate(zip(wkt_aois, bbs), start=1):
        cropped_in_vrt = tempfile.NamedTemporaryFile(
            suffix="_ld_error_recode_inputs.vrt", delete=False
        ).name
        gdal.BuildVRT(cropped_in_vrt, in_vrt, outputBounds=this_bbs)

        mask_tif = tempfile.NamedTemporaryFile(
            suffix="_ld_error_recode_mask.tif", delete=False
        ).name
        logger.info(f"Saving mask to {mask_tif}")
        logger.info(
            str(job_output_path.parent / mask_name_fragment.format(index=index))
        )
        geojson = wkt_geom_to_geojson_file_string(wkt_aoi)

        if mask_worker_function:
            mask_result = mask_worker_function(
                Path(mask_tif),
                geojson,
                Path(cropped_in_vrt),
                **(mask_worker_params or {}),
            )
        else:
            mask_worker = workers.Mask(
                Path(mask_tif),
                geojson,
                Path(cropped_in_vrt),
            )
            mask_result = mask_worker.work()

        if mask_result:
            out_path = job_output_path.parent / error_recode_name_pattern.format(
                index=index
            )
            error_recode_paths.append(out_path)

            logger.info(f"Calculating error recode and saving layer to: {out_path}")

            # Determine which band numbers are available
            baseline_band_num = band_dict.get("baseline_bandnum")
            reporting_1_band_num = band_dict.get("reporting_1_bandnum")
            reporting_2_band_num = band_dict.get("reporting_2_bandnum")

            # Calculate the number of output bands accurately
            n_out_bands = 1  # Always include recode band

            # Add baseline band if available
            if baseline_df is not None:
                n_out_bands += 1

            # Add reporting period 1 bands if available
            if reporting_1_df is not None:
                n_out_bands += 1  # Status map
                if write_reporting_sdg_tifs:
                    n_out_bands += 1  # Raw recoded band

            # Add reporting period 2 bands if available
            if reporting_2_df is not None:
                n_out_bands += 1  # Status map
                if write_reporting_sdg_tifs:
                    n_out_bands += 1  # Raw recoded band

            error_recode_params = DegradationErrorRecodeSummaryParams(
                in_file=str(cropped_in_vrt),
                out_file=str(out_path),
                band_dict=band_dict,
                model_band_number=1,
                n_out_bands=n_out_bands,
                mask_file=mask_tif,
                trans_code_lists=error_polygons.trans_code_lists,
                write_reporting_sdg_tifs=write_reporting_sdg_tifs,
                baseline_band_num=baseline_band_num,
                report_1_band_num=reporting_1_band_num,
                report_2_band_num=reporting_2_band_num,
            )

            if error_recode_worker_function:
                result = error_recode_worker_function(
                    error_recode_params,
                    _process_block,
                    **(error_recode_worker_params or {}),
                )
            else:
                summarizer = DegradationSummary(error_recode_params, _process_block)
                result = summarizer.work()

            if not result:
                if hasattr(result, "is_killed") and result.is_killed():
                    error_message = "Cancelled calculation of error recoding."
                else:
                    error_message = "Failed while performing error recode."
                    result = None
            else:
                error_recode_tables.append(_accumulate_summary_tables(result))

        else:
            error_message = "Error creating mask."

    if error_message:
        logger.error(error_message)
        raise RuntimeError(f"Error during error recode: {error_message}")

    error_recode_table = _accumulate_summary_tables(error_recode_tables)

    return error_recode_table, error_recode_paths
