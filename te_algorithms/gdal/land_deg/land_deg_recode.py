import logging
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np
from osgeo import gdal
from te_schemas.datafile import DataFile
from te_schemas.error_recode import ErrorRecodeGeoJSON
from te_schemas.jobs import JobBand

from . import config
from .. import workers

logger = logging.getLogger(__name__)


def rasterize_error_recode(
    out_file: Path,
    model_file: Path,
    geojson: ErrorRecodeGeoJSON,
) -> None:
    # Key for how recoding works
    #   First digit indicates from:
    #     1 is deg
    #     2 is imp
    #     3 is stable
    #     0 is unchanged
    #
    #   Second digit indicates to:
    #     1 is deg
    #     2 is imp
    #     3 is stable
    #     0 is unchanged
    #
    #   So keys are:
    #     recode_deg_to: unchanged 10, stable 12, improved 13
    #     recode_stable_to: unchanged, 20 deg 21,improved 23
    #     recode_imp_to: unchanged 30, deg 31, stable 32

    # Convert layer into an integer code so that all three recode_deg_to
    # options can be encoded within a single tiff

    recode_deg_to_options = [None, -32768, 0, 1]
    recode_stable_to_options = [None, -32768, -1, 1]
    recode_imp_to_options = [None, -32768, -1, 0]

    trans_code_to_recode = {}
    recode_to_trans_code = {}
    n = 0

    for i in range(len(recode_deg_to_options)):
        for j in range(len(recode_stable_to_options)):
            for k in range(len(recode_imp_to_options)):
                trans_code_to_recode[n] = (
                    recode_deg_to_options[i], recode_stable_to_options[j],
                    recode_imp_to_options[k]
                )
                recode_to_trans_code[(
                    recode_deg_to_options[i], recode_stable_to_options[j],
                    recode_imp_to_options[k]
                )] = n
                n += 1

    error_recode_dict = ErrorRecodeGeoJSON.Schema().dump(geojson)

    for feat in error_recode_dict['features']:
        feat['properties']['error_recode'] = recode_to_trans_code[(
            feat['properties']['recode_deg_to'],
            feat['properties']['recode_stable_to'],
            feat['properties']['recode_imp_to']
        )]

    # TODO: Assumes WGS84 for now
    rasterize_worker = workers.Rasterize(
        str(out_file), str(model_file), error_recode_dict, 'error_recode'
    )
    rasterize_worker.work()


def _process_block_error_recode_sdg(
    params: DegradationErrorRecodeParams, in_array, mask, xoff: int, yoff: int,
    cell_areas_raw
) -> Tuple[SummaryTableLDErrorRecode, Dict]:

    sdg = in_array[
        params.in_df.index_for_name(config.SDG_STATUS_BAND_NAME), :, :]
    error = in_array[
        params.in_df.index_for_name(config.ERROR_RECODE_BAND_NAME), :, :]
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1)

    # below works on data in place
    recode_errors(sdg, error, params.error_recode_dict)

    sdg_summary = zonal_total(sdg, cell_areas, mask)

    write_arrays = [{'array': sdg, 'xoff': xoff, 'yoff': yoff}]

    return (SummaryTableLDErrorRecode(sdg_summary), write_arrays)


def compute_error_recode_summary(
    df,
    job_output_path,
    aoi,
    error_recode_worker_function: Callable = None,
    error_recode_worker_params: dict = None
):
    # Calculate progress summary
    progress_vrt, progress_band_dict = _get_progress_summary_input_vrt(
        df, prod_mode
    )

    bbs = aoi.get_aligned_output_bounds(df.path)

    progress_name_pattern = {
        1: f"{job_output_path.stem}" + "_progress.tif",
        2: f"{job_output_path.stem}" + "_progress_{index}.tif"
    }[len(bbs)]

    progress_summary_tables = []
    progress_paths = []
    error_message = None

    for index, this_bbs in enumerate(bbs, start=1):
        progress_out_path = job_output_path.parent / progress_name_pattern.format(
            index=index
        )
        progress_paths.append(progress_out_path)

        # Need to crop error_recode to this bbs

        logger.info(
            f'Calculating progress summary table and saving layer to: {progress_out_path}'
        )
        progress_params = DegradationProgressSummaryParams(
            prod_mode=prod_mode,
            in_file=str(progress_vrt),
            out_file=str(progress_out_path),
            band_dict=progress_band_dict,
            model_band_number=1,
            n_out_bands=4,
            mask_file=mask_tif
        )

        if progress_worker_function:
            result = progress_worker_function(
                progress_params, _process_block_progress,
                **progress_worker_params
            )
        else:
            summarizer = DegradationSummary(
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
        JobBand(
            name=config.SDG_STATUS_BAND_NAME,
            no_data_value=NODATA_VALUE,
            metadata={
                'baseline_year_initial': baseline_period['year_initial'],
                'baseline_year_final': baseline_period['year_final'],
                'progress_year_initial': progress_period['year_initial'],
                'progress_year_final': progress_period['year_final']
            },
            add_to_map=True,
            activated=True
        ),
        JobBand(
            name=config.PROD_DEG_COMPARISON_BAND_NAME,
            no_data_value=NODATA_VALUE,
            metadata={
                'baseline_year_initial': baseline_period['year_initial'],
                'baseline_year_final': baseline_period['year_final'],
                'progress_year_initial': progress_period['year_initial'],
                'progress_year_final': progress_period['year_final']
            },
            add_to_map=True,
            activated=True
        ),
        JobBand(
            name=config.SOC_DEG_BAND_NAME,
            no_data_value=NODATA_VALUE,
            metadata={
                'year_initial': baseline_period['year_initial'],
                'year_final': progress_period['year_final']
            },
            add_to_map=True,
            activated=True
        ),
        JobBand(
            name=config.LC_DEG_COMPARISON_BAND_NAME,
            no_data_value=NODATA_VALUE,
            metadata={
                'year_initial': baseline_period['year_initial'],
                'year_final': progress_period['year_final']
            },
            add_to_map=True,
            activated=True
        )
    ]

    return progress_summary_table, DataFile(progress_path, out_bands)
