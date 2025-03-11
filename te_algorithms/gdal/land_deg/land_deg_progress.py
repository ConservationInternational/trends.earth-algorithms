import logging
import tempfile
from typing import Callable, Dict, List, Tuple

import numpy as np
from osgeo import gdal
from te_schemas.datafile import DataFile
from te_schemas.land_cover import LCLegendNesting
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band

from .. import util, workers
from ..util_numba import zonal_total
from . import config, models, worker
from .land_deg_numba import (
    calc_soc_pch,
    prod5_to_prod3,
    recode_deg_soc,
    sdg_status_expanded,
    sdg_status_expanded_to_simple,
)

logger = logging.getLogger(__name__)


def _accumulate_ld_progress_summary_tables(
    tables: List[models.SummaryTableLDProgress],
) -> models.SummaryTableLDProgress:
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


def compute_progress_summary(
    df,
    prod_mode,
    job_output_path,
    aoi,
    compute_bbs_from,
    periods,
    nesting: LCLegendNesting,
    mask_worker_function: Callable = None,
    mask_worker_params: dict = None,
    progress_worker_function: Callable = None,
    progress_worker_params: dict = None,
):
    # Calculate progress summary
    progress_vrt, progress_band_dict = _get_progress_summary_input_vrt(
        df, prod_mode, periods
    )

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    bbs = aoi.get_aligned_output_bounds(compute_bbs_from)
    assert len(wkt_aois) == len(bbs)

    if len(wkt_aois) > 1:
        progress_name_pattern = f"{job_output_path.stem}" + "_reporting_{index}.tif"
        mask_name_fragment = (
            "Generating mask for reporting analysis (part {index} of "
            + f"{len(wkt_aois)})"
        )
    else:
        progress_name_pattern = f"{job_output_path.stem}" + "_reporting.tif"
        mask_name_fragment = "Generating mask for reporting analysis"

    progress_summary_tables = []
    reporting_paths = []
    error_message = None

    for index, (wkt_aoi, this_bbs) in enumerate(zip(wkt_aois, bbs), start=1):
        cropped_progress_vrt = tempfile.NamedTemporaryFile(
            suffix="_ld_progress_summary_inputs.vrt", delete=False
        ).name
        gdal.BuildVRT(
            cropped_progress_vrt,
            progress_vrt,
            outputBounds=this_bbs,
            resolution="highest",
            resampleAlg=gdal.GRA_NearestNeighbour,
        )

        mask_tif = tempfile.NamedTemporaryFile(
            suffix="_ld_progress_mask.tif", delete=False
        ).name
        logger.info(f"Saving mask to {mask_tif}")
        logger.info(
            str(job_output_path.parent / mask_name_fragment.format(index=index))
        )
        geojson = util.wkt_geom_to_geojson_file_string(wkt_aoi)

        if mask_worker_function:
            mask_result = mask_worker_function(
                mask_tif, geojson, str(cropped_progress_vrt), **mask_worker_params
            )
        else:
            mask_worker = workers.Mask(
                mask_tif,
                geojson,
                str(cropped_progress_vrt),
            )
            mask_result = mask_worker.work()

        if mask_result:
            progress_out_path = job_output_path.parent / progress_name_pattern.format(
                index=index
            )
            reporting_paths.append(progress_out_path)

            logger.info(
                f"Calculating progress summary table and saving layer to: {progress_out_path}"
            )
            progress_params = models.DegradationProgressSummaryParams(
                prod_mode=prod_mode,
                in_file=str(cropped_progress_vrt),
                out_file=str(progress_out_path),
                band_dict=progress_band_dict,
                model_band_number=1,
                n_out_bands=4 * (len(periods) - 1),
                n_reporting=len(periods) - 1,
                mask_file=mask_tif,
                nesting=nesting,
            )

            if progress_worker_function:
                result = progress_worker_function(
                    progress_params, _process_block_progress, **progress_worker_params
                )
            else:
                summarizer = worker.DegradationSummary(
                    progress_params, _process_block_progress
                )
                result = summarizer.work()

            if not result:
                if result.is_killed():
                    error_message = "Cancelled calculation of progress summary table."
                else:
                    error_message = "Error calculating progress summary table."
                    result = None
            else:
                progress_summary_tables.append(
                    _accumulate_ld_progress_summary_tables(result)
                )

        else:
            error_message = "Error creating mask."

    if error_message:
        logger.error(error_message)
        raise RuntimeError(f"Error calculating progress: {error_message}")

    progress_summary_table = _accumulate_ld_progress_summary_tables(
        progress_summary_tables
    )

    if len(reporting_paths) > 1:
        reporting_path = job_output_path.parent / f"{job_output_path.stem}_progress.vrt"
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

    return progress_summary_table, DataFile(reporting_path, out_bands)


def _get_progress_summary_input_vrt(df, prod_mode, periods):
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
        suffix="_ld_progress_inputs.vrt", delete=False
    ).name
    gdal.BuildVRT(out_vrt, [vrt for vrt in band_vrts], separate=True)
    vrt_band_dict = {item[0]: index for index, item in enumerate(df_band_list, start=1)}

    return out_vrt, vrt_band_dict


def _process_block_progress(
    params: models.DegradationProgressSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw,
) -> Tuple[models.SummaryTableLDProgress, Dict]:
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1)

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

    ##########################################################################
    # Calculate SDG status layers
    sdg_baseline = in_array[params.band_dict["sdg_baseline_bandnum"] - 1, :, :]
    sdg_statuses = []
    sdg_summaries = []
    for i in range(params.n_reporting):
        sdg_reporting = in_array[
            params.band_dict[f"sdg_reporting_{i}_bandnum"] - 1, :, :
        ]
        sdg_status = sdg_status_expanded(sdg_baseline, sdg_reporting)
        sdg_statuses.append(sdg_status)
        sdg_summaries.append(
            zonal_total(sdg_status_expanded_to_simple(sdg_status), cell_areas, mask)
        )

    ##########################################################################
    # Calculate change in productivity degradation relative to baseline for each reporting period
    prod5_baseline = in_array[params.band_dict["prod5_baseline_bandnum"] - 1, :, :]
    prod_summaries = []
    prod_statuses = []
    for i in range(params.n_reporting):
        prod5_reporting = in_array[
            params.band_dict[f"prod5_reporting_{i}_bandnum"] - 1, :, :
        ]
        # Recode zeros in prod5 to config.NODATA_VALUE as the JRC LPD on
        # trends.earth assets had 0 used instead of our standard nodata value
        prod5_baseline[prod5_baseline == 0] = config.NODATA_VALUE
        prod5_reporting[prod5_reporting == 0] = config.NODATA_VALUE

        prod_status = sdg_status_expanded(
            prod5_to_prod3(prod5_baseline), prod5_to_prod3(prod5_reporting)
        )

        prod_statuses.append(prod_status)
        prod_summaries.append(
            {
                "all_cover_types": zonal_total(
                    sdg_status_expanded_to_simple(prod_status), cell_areas, mask
                ),
                "non_water": zonal_total(
                    sdg_status_expanded_to_simple(prod_status),
                    cell_areas,
                    mask_plus_water,
                ),
            }
        )

    ##########################################################################
    # LC
    lc_summaries = []
    lc_statuses = []
    lc_deg_baseline = in_array[params.band_dict["lc_deg_baseline_bandnum"] - 1, :, :]
    for i in range(params.n_reporting):
        lc_deg_reporting = in_array[
            params.band_dict[f"lc_deg_reporting_{i}_bandnum"] - 1, :, :
        ]

        lc_status = sdg_status_expanded(lc_deg_baseline, lc_deg_reporting)

        lc_statuses.append(lc_status)
        lc_summaries.append(
            zonal_total(sdg_status_expanded_to_simple(lc_status), cell_areas, mask)
        )

    ##########################################################################
    # SOC
    soc_summaries = []
    soc_statuses = []
    soc_deg_baseline = in_array[params.band_dict["soc_deg_baseline_bandnum"] - 1, :, :]
    for i in range(params.n_reporting):
        soc_pch = calc_soc_pch(
            in_array[params.band_dict["soc_baseline_bandnum"] - 1, :, :],
            in_array[params.band_dict[f"soc_reporting_{i}_bandnum"] - 1, :, :],
        )
        soc_deg_reporting = recode_deg_soc(soc_pch, water)
        soc_status = sdg_status_expanded(soc_deg_baseline, soc_deg_reporting)
        soc_statuses.append(soc_status)
        soc_summaries.append(
            {
                "all_cover_types": zonal_total(soc_status, cell_areas, mask),
                "non_water": zonal_total(soc_status, cell_areas, mask_plus_water),
            }
        )

    ##########################################################################
    # Write results
    write_arrays = [
        {"array": sdg_status, "xoff": xoff, "yoff": yoff} for sdg_status in sdg_statuses
    ]

    for prod_status in prod_statuses:
        write_arrays.append({"array": prod_status, "xoff": xoff, "yoff": yoff})
    for soc_status in soc_statuses:
        write_arrays.append({"array": soc_status, "xoff": xoff, "yoff": yoff})
    for lc_status in lc_statuses:
        write_arrays.append({"array": lc_status, "xoff": xoff, "yoff": yoff})

    return (
        models.SummaryTableLDProgress(
            sdg_summaries, prod_summaries, soc_summaries, lc_summaries
        ),
        write_arrays,
    )
