import logging
import tempfile
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from osgeo import gdal
from te_schemas.datafile import DataFile
from te_schemas.land_cover import LCLegendNesting
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band

from . import config
from . import models
from . import worker
from .. import util
from .. import workers
from ..util_numba import zonal_total
from .land_deg_numba import calc_deg_lc
from .land_deg_numba import calc_deg_sdg
from .land_deg_numba import calc_progress_lc_deg
from .land_deg_numba import calc_soc_pch
from .land_deg_numba import recode_deg_soc

logger = logging.getLogger(__name__)


def _accumulate_ld_progress_summary_tables(
    tables: List[models.SummaryTableLDProgress],
) -> models.SummaryTableLDProgress:
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]

    for table in tables[1:]:
        out.sdg_summary = util.accumulate_dicts([out.sdg_summary, table.sdg_summary])
        assert set(out.prod_summary.keys()) == set(table.prod_summary.keys())
        out.prod_summary = {
            key: util.accumulate_dicts([out.prod_summary[key], table.prod_summary[key]])
            for key in out.prod_summary.keys()
        }
        assert set(out.soc_summary.keys()) == set(table.soc_summary.keys())
        out.soc_summary = {
            key: util.accumulate_dicts([out.soc_summary[key], table.soc_summary[key]])
            for key in out.soc_summary.keys()
        }
        out.lc_summary = util.accumulate_dicts([out.lc_summary, table.lc_summary])

    return out


def compute_progress_summary(
    df,
    prod_mode,
    job_output_path,
    aoi,
    compute_bbs_from,
    baseline_period,
    progress_period,
    nesting: LCLegendNesting,
    mask_worker_function: Callable = None,
    mask_worker_params: dict = None,
    progress_worker_function: Callable = None,
    progress_worker_params: dict = None,
):
    # Calculate progress summary
    progress_vrt, progress_band_dict = _get_progress_summary_input_vrt(df, prod_mode)

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    bbs = aoi.get_aligned_output_bounds(compute_bbs_from)
    assert len(wkt_aois) == len(bbs)

    progress_name_pattern = {
        1: f"{job_output_path.stem}" + "_progress.tif",
        2: f"{job_output_path.stem}" + "_progress_{index}.tif",
    }[len(wkt_aois)]
    mask_name_fragment = {
        1: "Generating mask for progress analysis",
        2: "Generating mask for progress analysis (part {index} of 2)",
    }[len(wkt_aois)]

    progress_summary_tables = []
    progress_paths = []
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
            progress_paths.append(progress_out_path)

            logger.info(
                f"Calculating progress summary table and saving layer to: {progress_out_path}"
            )
            progress_params = models.DegradationProgressSummaryParams(
                prod_mode=prod_mode,
                in_file=str(cropped_progress_vrt),
                out_file=str(progress_out_path),
                band_dict=progress_band_dict,
                model_band_number=1,
                n_out_bands=4,
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

    if len(progress_paths) > 1:
        progress_path = job_output_path.parent / f"{job_output_path.stem}_progress.vrt"
        gdal.BuildVRT(str(progress_path), [str(p) for p in progress_paths])
    else:
        progress_path = progress_paths[0]

    out_bands = [
        Band(
            name=config.SDG_STATUS_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                "baseline_year_initial": baseline_period["year_initial"],
                "baseline_year_final": baseline_period["year_final"],
                "progress_year_initial": progress_period["year_initial"],
                "progress_year_final": progress_period["year_final"],
            },
            add_to_map=True,
            activated=True,
        ),
        Band(
            name=config.PROD_DEG_COMPARISON_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                "baseline_year_initial": baseline_period["year_initial"],
                "baseline_year_final": baseline_period["year_final"],
                "progress_year_initial": progress_period["year_initial"],
                "progress_year_final": progress_period["year_final"],
            },
            add_to_map=True,
            activated=True,
        ),
        Band(
            name=config.SOC_DEG_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                "year_initial": baseline_period["year_initial"],
                "year_final": progress_period["year_final"],
            },
            add_to_map=True,
            activated=True,
        ),
        Band(
            name=config.LC_DEG_COMPARISON_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                "baseline_year_initial": baseline_period["year_initial"],
                "baseline_year_final": baseline_period["year_final"],
                "progress_year_initial": progress_period["year_initial"],
                "progress_year_final": progress_period["year_final"],
            },
            add_to_map=True,
            activated=True,
        ),
    ]

    return progress_summary_table, DataFile(progress_path, out_bands)


def _get_progress_summary_input_vrt(df, prod_mode):
    if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        prod5_indices = [
            (index, year)
            for index, year in zip(
                df.indices_for_name(config.TE_LPD_BAND_NAME),
                df.metadata_for_name(config.TE_LPD_BAND_NAME, "year_initial"),
            )
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
                df.metadata_for_name(lpd_layer_name, "year_initial"),
            )
        ]
    assert len(prod5_indices) == 2
    prod5_indices = sorted(prod5_indices, key=lambda row: row[1])
    prod5_baseline_index = prod5_indices[0][0]
    prod5_progress_index = prod5_indices[1][0]

    lc_deg_indices = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.LC_DEG_BAND_NAME),
            df.metadata_for_name(config.LC_DEG_BAND_NAME, "year_initial"),
        )
    ]
    assert len(lc_deg_indices) == 2
    lc_deg_indices = sorted(lc_deg_indices, key=lambda row: row[1])
    lc_deg_baseline_index = lc_deg_indices[0][0]
    lc_deg_progress_index = lc_deg_indices[1][0]

    lc_bands = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.LC_BAND_NAME),
            df.metadata_for_name(config.LC_BAND_NAME, "year"),
        )
    ]
    lc_baseline_index = [
        index for index, year in lc_bands if year == lc_deg_indices[0][1]
    ][0]

    soc_indices = [
        (index, year)
        for index, year in zip(
            df.indices_for_name(config.SOC_BAND_NAME),
            df.metadata_for_name(config.SOC_BAND_NAME, "year"),
        )
    ]
    soc_indices = sorted(soc_indices, key=lambda row: row[1])
    soc_initial_index = soc_indices[0][0]
    soc_final_index = soc_indices[-1][0]

    df_band_list = [
        ("prod5_baseline_bandnum", prod5_baseline_index),
        ("prod5_progress_bandnum", prod5_progress_index),
        ("lc_deg_baseline_bandnum", lc_deg_baseline_index),
        ("lc_deg_progress_bandnum", lc_deg_progress_index),
        ("lc_baseline_bandnum", lc_baseline_index),
        ("soc_initial_bandnum", soc_initial_index),
        ("soc_final_bandnum", soc_final_index),
    ]

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

    if params.prod_mode == ProductivityMode.FAO_WOCAT_5_CLASS_LPD.value:
        trans_code = [
            11, 12, 13, 14, 15,
            21, 22, 23, 24, 25,
            31, 32, 33, 34, 35,
            41, 42, 43, 44, 45,
            51, 52, 53, 54, 55,
        ]  # fmt:skip

        trans_meaning_sdg = [
            -1, -1, -1, 0, 1,
            -1, -1, -1, 0, 1,
            -1, -1, 0, 0, 1,
            -1, -1, 0, 0, 1,
            -1, -1, 0, 0, 1,
        ]  # fmt:skip
    else:
        trans_code = [
            11, 12, 13, 14, 15,
            21, 22, 23, 24, 25,
            31, 32, 33, 34, 35,
            41, 42, 43, 44, 45,
            51, 52, 53, 54, 55,
        ]  # fmt:skip

        trans_meaning_sdg = [
            -1, -1, -1, 1, 1,
            -1, -1, -1, 0, 1,
            -1, -1, 0, 0, 1,
            -1, -1, 0, 0, 1,
            -1, -1, 0, 0, 1,
        ]  # fmt:skip

    # Derive a water mask from last lc year - handling fact that there could be
    # multiple water codes if this is a custom legend. 7 is the IPCC water code.
    water = np.isin(
        in_array[params.band_dict["lc_baseline_bandnum"] - 1, :, :],
        params.nesting.nesting[7],
    )
    water = water.astype(bool, copy=False)

    prod5_baseline = in_array[params.band_dict["prod5_baseline_bandnum"] - 1, :, :]
    prod5_progress = in_array[params.band_dict["prod5_progress_bandnum"] - 1, :, :]
    # TODO: recode zeros in prod5 to config.NODATA_VALUE as the JRC LPD on
    # trends.earth assets had 0 used instead of our standard nodata value
    prod5_baseline[prod5_baseline == 0] = config.NODATA_VALUE
    prod5_progress[prod5_progress == 0] = config.NODATA_VALUE

    # Productivity - can use the productivity degradation calculation function
    # to do the recoding, as it calculates transitions and recodes them
    # according to a matrix
    deg_prod_progress = calc_deg_lc(
        prod5_baseline, prod5_progress, trans_code, trans_meaning_sdg, 10
    )

    # Make a tabulation of the productivity and SOC summaries without water for
    # usage on Prais4
    mask_plus_water = mask.copy()
    mask_plus_water[water] = True

    prod_summary = {
        "all_cover_types": zonal_total(deg_prod_progress, cell_areas, mask),
        "non_water": zonal_total(deg_prod_progress, cell_areas, mask_plus_water),
    }

    # LC
    deg_lc = calc_progress_lc_deg(
        in_array[params.band_dict["lc_deg_baseline_bandnum"] - 1, :, :],
        in_array[params.band_dict["lc_deg_progress_bandnum"] - 1, :, :],
    )
    lc_summary = zonal_total(deg_lc, cell_areas, mask)

    # SOC
    soc_pch = calc_soc_pch(
        in_array[params.band_dict["soc_initial_bandnum"] - 1, :, :],
        in_array[params.band_dict["soc_final_bandnum"] - 1, :, :],
    )
    deg_soc = recode_deg_soc(soc_pch, water)
    soc_summary = {
        "all_cover_types": zonal_total(deg_soc, cell_areas, mask),
        "non_water": zonal_total(deg_soc, cell_areas, mask_plus_water),
    }

    # Summarize results
    deg_sdg = calc_deg_sdg(deg_prod_progress, deg_lc, deg_soc)

    sdg_summary = zonal_total(deg_sdg, cell_areas, mask)

    write_arrays = [
        {"array": deg_sdg, "xoff": xoff, "yoff": yoff},
        {"array": deg_prod_progress, "xoff": xoff, "yoff": yoff},
        {"array": soc_pch, "xoff": xoff, "yoff": yoff},
        {"array": deg_lc, "xoff": xoff, "yoff": yoff},
    ]

    return (
        models.SummaryTableLDProgress(
            sdg_summary, prod_summary, soc_summary, lc_summary
        ),
        write_arrays,
    )
