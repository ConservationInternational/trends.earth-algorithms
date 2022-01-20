import datetime as dt
import json
import logging
import tempfile
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from osgeo import gdal
from te_schemas import land_cover
from te_schemas.aoi import AOI
from te_schemas.datafile import combine_data_files
from te_schemas.datafile import DataFile
from te_schemas.jobs import Job
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band
from te_schemas.results import DataType
from te_schemas.results import Raster
from te_schemas.results import RasterFileType
from te_schemas.results import RasterResults
from te_schemas.results import URI

from . import config
from . import models
from . import worker
from .. import util
from .. import workers
from .. import xl
from ... import __release_date__
from ... import __version__
from ..util_numba import bizonal_total
from ..util_numba import zonal_total
from ..util_numba import zonal_total_weighted
from .land_deg_numba import calc_deg_sdg
from .land_deg_numba import calc_lc_trans
from .land_deg_numba import calc_prod5
from .land_deg_numba import prod5_to_prod3
from .land_deg_numba import recode_deg_soc
from .land_deg_numba import recode_indicator_errors
from .land_deg_numba import recode_state
from .land_deg_numba import recode_traj
from .land_deg_progress import compute_progress_summary
from .land_deg_report import save_reporting_json
from .land_deg_report import save_summary_table_excel

logger = logging.getLogger(__name__)


def _accumulate_ld_summary_tables(
    tables: List[models.SummaryTableLD]
) -> models.SummaryTableLD:

    if len(tables) == 1:
        return tables[0]

    out = tables[0]

    for table in tables[1:]:
        out.soc_by_lc_annual_totals = [
            util.accumulate_dicts([a, b]) for a, b in
            zip(out.soc_by_lc_annual_totals, table.soc_by_lc_annual_totals)
        ]
        out.lc_annual_totals = [
            util.accumulate_dicts([a, b])
            for a, b in zip(out.lc_annual_totals, table.lc_annual_totals)
        ]
        out.lc_trans_zonal_areas = [
            util.accumulate_dicts([a, b]) for a, b in
            zip(out.lc_trans_zonal_areas, table.lc_trans_zonal_areas)
        ]
        # A period should be listed for each object in lc_trans_zonal_areas
        assert (
            len(out.lc_trans_zonal_areas
                ) == len(table.lc_trans_zonal_areas_periods)
        )
        # Periods for lc_trans_zonal_areas must be the same in both objects
        assert (
            out.lc_trans_zonal_areas_periods ==
            table.lc_trans_zonal_areas_periods
        )
        out.lc_trans_prod_bizonal = util.accumulate_dicts(
            [out.lc_trans_prod_bizonal, table.lc_trans_prod_bizonal]
        )
        out.lc_trans_zonal_soc_initial = util.accumulate_dicts(
            [out.lc_trans_zonal_soc_initial, table.lc_trans_zonal_soc_initial]
        )
        out.lc_trans_zonal_soc_final = util.accumulate_dicts(
            [out.lc_trans_zonal_soc_final, table.lc_trans_zonal_soc_final]
        )
        out.sdg_zonal_population_total = util.accumulate_dicts(
            [out.sdg_zonal_population_total, table.sdg_zonal_population_total]
        )
        out.sdg_zonal_population_male = util.accumulate_dicts(
            [out.sdg_zonal_population_male, table.sdg_zonal_population_male]
        )
        out.sdg_zonal_population_female = util.accumulate_dicts(
            [
                out.sdg_zonal_population_female,
                table.sdg_zonal_population_female
            ]
        )
        out.sdg_summary = util.accumulate_dicts(
            [out.sdg_summary, table.sdg_summary]
        )
        assert set(out.prod_summary.keys()) == set(table.prod_summary.keys())
        out.prod_summary = {
            key: util.accumulate_dicts(
                [out.prod_summary[key], table.prod_summary[key]]
            )
            for key in out.prod_summary.keys()
        }
        assert set(out.soc_summary.keys()) == set(table.soc_summary.keys())
        out.soc_summary = {
            key: util.accumulate_dicts(
                [out.soc_summary[key], table.soc_summary[key]]
            )
            for key in out.soc_summary.keys()
        }
        out.lc_summary = util.accumulate_dicts(
            [out.lc_summary, table.lc_summary]
        )

    return out


def _prepare_land_cover_dfs(params: Dict) -> List[DataFile]:
    lc_path = params["layer_lc_path"]
    lc_dfs = [
        DataFile(
            path=util.save_vrt(lc_path, params["layer_lc_deg_band_index"]),
            bands=[Band(**params["layer_lc_deg_band"])]
        )
    ]

    for lc_aux_band, lc_aux_band_index, in zip(
        params["layer_lc_aux_bands"], params["layer_lc_aux_band_indexes"]
    ):
        lc_dfs.append(
            DataFile(
                path=util.save_vrt(lc_path, lc_aux_band_index),
                bands=[Band(**lc_aux_band)]
            )
        )
    lc_dfs.append(
        DataFile(
            path=util.save_vrt(
                params["layer_lc_trans_path"],
                params["layer_lc_trans_band_index"],
            ),
            bands=[Band(**params["layer_lc_trans_band"])]
        )
    )

    return lc_dfs


def _prepare_population_dfs(params: Dict) -> DataFile:
    population_dfs = []

    for population_band, population_band_index, in zip(
        params["layer_population_bands"],
        params["layer_population_band_indexes"]
    ):
        population_dfs.append(
            DataFile(
                path=util.save_vrt(
                    params["layer_population_path"], population_band_index
                ),
                bands=[Band(**population_band)]
            )
        )

    return population_dfs


def _prepare_soil_organic_carbon_dfs(params: Dict) -> List[DataFile]:
    soc_path = params["layer_soc_path"]
    soc_dfs = [
        DataFile(
            path=util.save_vrt(soc_path, params["layer_soc_deg_band_index"]),
            bands=[Band(**params["layer_soc_deg_band"])]
        )
    ]

    for soc_aux_band, soc_aux_band_index, in zip(
        params["layer_soc_aux_bands"], params["layer_soc_aux_band_indexes"]
    ):
        soc_dfs.append(
            DataFile(
                path=util.save_vrt(soc_path, soc_aux_band_index),
                bands=[Band(**soc_aux_band)]
            )
        )

    return soc_dfs


def _prepare_trends_earth_mode_dfs(
    params: Dict
) -> Tuple[DataFile, DataFile, DataFile]:
    traj_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_traj_path"],
            params["layer_traj_band_index"],
        ),
        bands=[Band(**params["layer_traj_band"])]
    )
    perf_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_perf_path"],
            params["layer_perf_band_index"],
        ),
        bands=[Band(**params["layer_perf_band"])]
    )
    state_vrt_df = DataFile(
        path=util.save_vrt(
            params["layer_state_path"],
            params["layer_state_band_index"],
        ),
        bands=[Band(**params["layer_state_band"])]
    )

    return traj_vrt_df, perf_vrt_df, state_vrt_df


def _prepare_jrc_lpd_mode_df(params: Dict) -> DataFile:
    return DataFile(
        path=util.save_vrt(
            params["layer_lpd_path"], params["layer_lpd_band_index"]
        ),
        bands=[Band(**params["layer_lpd_band"])]
    )


def summarise_land_degradation(
    ldn_job: Job, aoi: AOI, job_output_path: Path
) -> Job:
    """Calculate final SDG 15.3.1 indicator and save to disk"""
    logger.debug('at top of compute_ldn')

    summary_tables = {}
    summary_table_stable_kwargs = {}

    period_dfs = []
    period_vrts = []

    for period_name, period_params in ldn_job.params.items():
        lc_dfs = _prepare_land_cover_dfs(period_params)
        soc_dfs = _prepare_soil_organic_carbon_dfs(period_params)
        population_dfs = _prepare_population_dfs(period_params)
        sub_job_output_path = job_output_path.parent / f"{job_output_path.stem}_{period_name}.json"
        prod_mode = period_params["prod_mode"]

        period_params['periods'] = {
            'land_cover': period_params["layer_lc_deg_years"],
            'soc': period_params["layer_soc_deg_years"]
        }

        if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
            period_params['periods']['productivity'] = period_params[
                "layer_traj_years"]
        elif prod_mode == ProductivityMode.JRC_5_CLASS_LPD.value:
            period_params['periods']['productivity'] = period_params[
                "layer_lpd_years"]
        else:
            raise Exception(f"Unknown productivity mode {prod_mode}")

        # Add in period start/end if it isn't already in the parameters
        # (wouldn't be if these layers were all run individually and not
        # with the all-in-one tool)

        if 'period' not in period_params:
            period_params["period"] = {
                "name":
                period_name,
                "year_initial":
                period_params['periods']['productivity']['year_initial'],
                "year_final":
                period_params['periods']['productivity']['year_final']
            }
        nesting = period_params["layer_lc_deg_band"]["metadata"].get('nesting')

        if nesting:
            # Nesting is included only to ensure it goes into output, so if
            # missing (as it might be for local data), it will be set to None
            nesting = land_cover.LCLegendNesting.Schema().loads(nesting)
        summary_table_stable_kwargs[period_name] = {
            "aoi":
            aoi,
            "lc_legend_nesting":
            nesting,
            "lc_trans_matrix":
            land_cover.LCTransitionDefinitionDeg.Schema().loads(
                period_params["layer_lc_deg_band"]["metadata"]['trans_matrix'],
            ),
            # "soc_legend_nesting":
            # land_cover.LCLegendNesting.Schema().loads(
            #     period_params["layer_soc_deg_band"]["metadata"]['nesting'], ),
            # "soc_trans_matrix":
            # land_cover.LCTransitionDefinitionDeg.Schema().loads(
            #     period_params["layer_soc_deg_band"]["metadata"]
            #     ['trans_matrix'], ),
            "output_job_path":
            sub_job_output_path,
            "period_name":
            period_name,
            "periods":
            period_params['periods'],
        }

        if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
            traj, perf, state = _prepare_trends_earth_mode_dfs(period_params)
            in_dfs = lc_dfs + soc_dfs + [traj, perf, state] + population_dfs
            summary_table, sdg_path, reproj_path = _compute_ld_summary_table(
                in_dfs=in_dfs,
                prod_mode=ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value,
                compute_bbs_from=traj.path,
                **summary_table_stable_kwargs[period_name]
            )
        elif prod_mode == ProductivityMode.JRC_5_CLASS_LPD.value:
            lpd_df = _prepare_jrc_lpd_mode_df(period_params)
            in_dfs = lc_dfs + soc_dfs + [lpd_df] + population_dfs
            summary_table, sdg_path, reproj_path = _compute_ld_summary_table(
                in_dfs=in_dfs,
                prod_mode=ProductivityMode.JRC_5_CLASS_LPD.value,
                compute_bbs_from=lpd_df.path,
                **summary_table_stable_kwargs[period_name],
            )
        else:
            raise RuntimeError(f"Invalid prod_mode: {prod_mode!r}")

        summary_tables[period_name] = summary_table

        sdg_band = Band(
            name=config.SDG_BAND_NAME,
            no_data_value=config.NODATA_VALUE.item(),  # write as python type
            metadata={
                'year_initial': period_params['period']['year_initial'],
                'year_final': period_params['period']['year_final'],
            },
            activated=True
        )
        sdg_df = DataFile(sdg_path, [sdg_band])

        so3_band_total = _get_so3_band_instance(
            'total', period_params['periods']['productivity']
        )
        sdg_df.bands.append(so3_band_total)

        if _have_pop_by_sex(population_dfs):
            so3_band_female = _get_so3_band_instance(
                'female', period_params['periods']['productivity']
            )
            sdg_df.bands.append(so3_band_female)
            so3_band_male = _get_so3_band_instance(
                'male', period_params['periods']['productivity']
            )
            sdg_df.bands.append(so3_band_male)

        if prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
            prod_band = Band(
                name=config.TE_LPD_BAND_NAME,
                no_data_value=config.NODATA_VALUE.item(
                ),  # write as python type
                metadata={
                    'year_initial':
                    period_params['periods']['productivity']['year_initial'],
                    'year_final':
                    period_params['periods']['productivity']['year_final']
                },
                activated=True
            )
            sdg_df.bands.append(prod_band)

        reproj_df = combine_data_files(reproj_path, in_dfs)
        # Don't add any of the input layers to the map by default - only SDG,
        # prod, and SO3, which are already marked add_to_map=True

        for band in reproj_df.bands:
            band.add_to_map = False

        period_vrt = job_output_path.parent / f"{sub_job_output_path.stem}_rasterdata.vrt"
        util.combine_all_bands_into_vrt([sdg_path, reproj_path], period_vrt)

        # Now that there is a single VRT with all files in it, combine the DFs
        # into it so that it can be used to source band names/metadata for the
        # job bands list
        period_df = combine_data_files(period_vrt, [sdg_df, reproj_df])

        for band in period_df.bands:
            band.metadata['period'] = period_name
        period_dfs.append(period_df)

        period_vrts.append(period_vrt)

        summary_table_output_path = sub_job_output_path.parent / f"{sub_job_output_path.stem}.xlsx"
        save_summary_table_excel(
            summary_table_output_path, summary_table, period_params["periods"],
            period_params["layer_lc_years"], period_params["layer_soc_years"],
            summary_table_stable_kwargs[period_name]['lc_legend_nesting'],
            summary_table_stable_kwargs[period_name]['lc_trans_matrix'],
            period_name
        )

    if len(ldn_job.params.items()) == 2:
        # Make temporary combined VRT and DataFile just for the progress
        # calculations. Don't save these in the output folder as at end of this
        # process all the DFs will be combined and referenced to a VRT in that
        # folder
        temp_overall_vrt = Path(
            tempfile.NamedTemporaryFile(suffix='.vrt', delete=False).name
        )
        util.combine_all_bands_into_vrt(period_vrts, temp_overall_vrt)
        temp_df = combine_data_files(temp_overall_vrt, period_dfs)

        progress_summary_table, progress_df = compute_progress_summary(
            temp_df, prod_mode, job_output_path, aoi,
            ldn_job.params['baseline']['period'],
            ldn_job.params['progress']['period']
        )
        period_vrts.append(progress_df.path)
        period_dfs.append(progress_df)
    else:
        progress_summary_table = None

    overall_vrt_path = job_output_path.parent / f"{job_output_path.stem}.vrt"
    util.combine_all_bands_into_vrt(period_vrts, overall_vrt_path)
    out_df = combine_data_files(overall_vrt_path, period_dfs)
    out_df.path = overall_vrt_path.name

    # Also save bands to a key file for ease of use in PRAIS
    key_json = job_output_path.parent / f"{job_output_path.stem}_band_key.json"
    with open(key_json, 'w') as f:
        json.dump(DataFile.Schema().dump(out_df), f, indent=4)

    summary_json_output_path = job_output_path.parent / f"{job_output_path.stem}_summary.json"
    report_json = save_reporting_json(
        summary_json_output_path, summary_tables, progress_summary_table,
        ldn_job.params, ldn_job.task_name, aoi, summary_table_stable_kwargs
    )

    ldn_job.results = RasterResults(
        name='land_condition_summary',
        uri=URI(uri=overall_vrt_path, type='local'),
        rasters={
            DataType.INT16.value:
            Raster(
                uri=URI(uri=overall_vrt_path, type='local'),
                bands=out_df.bands,
                datatype=DataType.INT16,
                filetype=RasterFileType.COG,
            ),
        },
        data={'report': report_json}
    )

    ldn_job.end_date = dt.datetime.now(dt.timezone.utc)
    ldn_job.progress = 100

    return ldn_job


def _process_block_summary(
    params: models.DegradationSummaryParams, in_array, mask, xoff: int,
    yoff: int, cell_areas_raw
) -> Tuple[models.SummaryTableLD, Dict]:

    lc_band_years = params.in_df.metadata_for_name(config.LC_BAND_NAME, 'year')
    lc_bands = [
        (band, year) for band, year in
        zip(params.in_df.indices_for_name(config.LC_BAND_NAME), lc_band_years)
    ]
    soc_bands = [
        (band, year) for band, year in zip(
            params.in_df.indices_for_name(config.SOC_BAND_NAME),
            params.in_df.metadata_for_name(config.SOC_BAND_NAME, 'year')
        )
    ]

    # Create container for output arrays (will write later in main thread)
    write_arrays = []

    # Calculate cell area for each horizontal line
    # logger.debug('y: {}'.format(y))
    # logger.debug('x: {}'.format(x))
    # logger.debug('rows: {}'.format(rows))

    # Make an array of the same size as the input arrays containing
    # the area of each cell (which is identical for all cells in a
    # given row - cell areas only vary among rows)
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1],
                           axis=1).astype(np.float64)

    if params.prod_mode == ProductivityMode.TRENDS_EARTH_5_CLASS_LPD.value:
        traj_array = in_array[
            params.in_df.index_for_name(config.TRAJ_BAND_NAME), :, :]
        traj_recode = recode_traj(traj_array)

        state_array = in_array[
            params.in_df.index_for_name(config.STATE_BAND_NAME), :, :]
        state_recode = recode_state(state_array)

        perf_array = in_array[
            params.in_df.index_for_name(config.PERF_BAND_NAME), :, :]

        deg_prod5 = calc_prod5(traj_recode, state_recode, perf_array)

    elif params.prod_mode == ProductivityMode.JRC_5_CLASS_LPD.value:
        deg_prod5 = in_array[
            params.in_df.index_for_name(config.JRC_LPD_BAND_NAME), :, :]
        # TODO: Below is temporary until missing data values are
        # fixed in LPD layer on GEE and missing data values are
        # fixed in LPD layer made by UNCCD for SIDS
        deg_prod5[(deg_prod5 == 0) | (deg_prod5 == 15)] = config.NODATA_VALUE
    else:
        raise Exception(f"Unknown productivity mode {prod_mode}")

    # Recode deg_prod5 as stable, degraded, improved (deg_prod3)
    deg_prod3 = prod5_to_prod3(deg_prod5)

    if 'prod' in params.error_recode:
        prod_error_recode = in_array[
            params.in_df.
            index_for_name(config.PROD_DEG_ERROR_RECODE_BAND_NAME), :, :]
        recode_indicator_errors(deg_prod3, prod_error_recode)

    ###########################################################
    # Calculate LC transition arrays
    lc_deg_band_period = params.periods['land_cover']
    lc_deg_initial_cover_row = [
        row for row, year in lc_bands
        if year == lc_deg_band_period['year_initial']
    ][0]
    lc_deg_final_cover_row = [
        row for row, year in lc_bands
        if year == lc_deg_band_period['year_final']
    ][0]
    # a_lc_trans_lc_deg is an array land cover transitions over the time period
    # used in the land cover degradation layer
    a_lc_trans_lc_deg = calc_lc_trans(
        in_array[lc_deg_initial_cover_row, :, :],
        in_array[lc_deg_final_cover_row, :, :],
        params.trans_matrix.get_multiplier()
    )
    lc_trans_arrays = [a_lc_trans_lc_deg]
    lc_trans_zonal_areas_periods = [lc_deg_band_period]

    # Productivity data might be calculated over a different period than the
    # land cover degradation data. If this is the case, and land cover layers
    # are available for the years actually used for productivity, then create
    # an array of land cover transition that can be used for productivity, and
    # call that a_lc_trans_prod_deg
    soc_deg_band_period = params.periods['soc']
    prod_deg_band_period = params.periods['productivity']

    if prod_deg_band_period == lc_deg_band_period:
        a_lc_trans_prod_deg = a_lc_trans_lc_deg
    elif (
        prod_deg_band_period['year_initial'] in lc_band_years
        and prod_deg_band_period['year_final'] in lc_band_years
    ):
        prod_deg_initial_cover_row = [
            row for row, year in lc_bands
            if year == prod_deg_band_period['year_initial']
        ][0]
        prod_deg_final_cover_row = [
            row for row, year in lc_bands
            if year == prod_deg_band_period['year_final']
        ][0]
        a_lc_trans_prod_deg = calc_lc_trans(
            in_array[prod_deg_initial_cover_row, :, :],
            in_array[prod_deg_final_cover_row, :, :],
            params.trans_matrix.get_multiplier()
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
        soc_deg_band_period['year_initial'] in lc_band_years
        and soc_deg_band_period['year_final'] in lc_band_years
    ):
        soc_deg_initial_cover_row = [
            row for row, year in lc_bands
            if year == soc_deg_band_period['year_initial']
        ][0]
        soc_deg_final_cover_row = [
            row for row, year in lc_bands
            if year == soc_deg_band_period['year_final']
        ][0]
        a_lc_trans_soc_deg = calc_lc_trans(
            in_array[soc_deg_initial_cover_row, :, :],
            in_array[soc_deg_final_cover_row, :, :],
            params.trans_matrix.get_multiplier()
        )
        lc_trans_arrays.append(a_lc_trans_soc_deg)
        lc_trans_zonal_areas_periods.append(soc_deg_band_period)
    else:
        a_lc_trans_soc_deg = None

    ###########################################################
    # Calculate SOC totals by year. Note final units of soc_totals
    # tables are tons C (summed over the total area of each class).

    # First filter the SOC years to only those years for which land cover is
    # available.
    lc_years = [year for band, year in lc_bands]
    soc_bands_with_lc_avail = [
        (band, year) for band, year in soc_bands if year in lc_years
    ]
    lc_rows_for_soc = [
        params.in_df.index_for_name(config.LC_BAND_NAME, 'year', year)
        for band, year in soc_bands_with_lc_avail
    ]
    soc_by_lc_annual_totals = []

    for index, (soc_row, soc_year) in enumerate(soc_bands_with_lc_avail):
        a_lc = in_array[lc_rows_for_soc[index], :, :]
        a_soc = in_array[soc_row, :, :]
        soc_by_lc_annual_totals.append(
            zonal_total_weighted(
                a_lc,
                a_soc,
                cell_areas * 100,  # from sq km to hectares
                mask
            )
        )

    if (
        (soc_deg_band_period['year_initial'] in lc_years)
        and (soc_deg_band_period['year_final'] in lc_years)
    ):
        logger.debug(
            'year_initial %s, year_final %s, lc_years %s',
            soc_deg_band_period['year_initial'],
            soc_deg_band_period['year_final'], lc_years
        )
        a_soc_bl = in_array[params.in_df.indices_for_name(
            config.SOC_BAND_NAME,
            field='year',
            field_filter=soc_deg_band_period['year_initial']
        ), :, :]
        a_soc_final = in_array[params.in_df.indices_for_name(
            config.SOC_BAND_NAME,
            field='year',
            field_filter=soc_deg_band_period['year_final']
        ), :, :]

        lc_trans_zonal_soc_initial = zonal_total_weighted(
            a_lc_trans_soc_deg,
            a_soc_bl,
            cell_areas * 100,  # from sq km to hectares
            mask
        )
        lc_trans_zonal_soc_final = zonal_total_weighted(
            a_lc_trans_soc_deg,
            a_soc_final,
            cell_areas * 100,  # from sq km to hectares
            mask
        )
    else:
        lc_trans_zonal_soc_initial = {}
        lc_trans_zonal_soc_final = {}

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

    for lc_row, lc_year in lc_bands:
        a_lc = in_array[lc_row, :, :]
        lc_annual_totals.append(zonal_total(a_lc, cell_areas, mask))

    ###########################################################
    # Calculate crosstabs for land cover
    lc_trans_zonal_areas = []

    for lc_trans_array in lc_trans_arrays:
        lc_trans_zonal_areas.append(
            zonal_total(lc_trans_array, cell_areas, mask)
        )

    ################
    # Calculate SDG
    # Derive a water mask from last lc year
    water = in_array[lc_deg_initial_cover_row, :, :] == 7
    water = water.astype(bool, copy=False)

    deg_soc = in_array[
        params.in_df.index_for_name(config.SOC_DEG_BAND_NAME), :, :]
    deg_soc = recode_deg_soc(deg_soc, water)

    if 'soc' in params.error_recode:
        soc_error_recode = in_array[
            params.in_df.index_for_name(config.
                                        SOC_DEG_ERROR_RECODE_BAND_NAME), :, :]
        recode_indicator_errors(deg_soc, soc_error_recode)

    deg_lc = in_array[
        params.in_df.index_for_name(config.LC_DEG_BAND_NAME), :, :]

    if 'lc' in params.error_recode:
        lc_error_recode = in_array[
            params.in_df.index_for_name(config.
                                        LC_DEG_ERROR_RECODE_BAND_NAME), :, :]
        recode_indicator_errors(deg_lc, lc_error_recode)

    deg_sdg = calc_deg_sdg(deg_prod3, deg_lc, deg_soc)

    if 'sdg' in params.error_recode:
        sdg_error_recode = in_array[
            params.in_df.index_for_name(config.
                                        SDG_DEG_ERROR_RECODE_BAND_NAME), :, :]
        recode_indicator_errors(deg_sdg, sdg_error_recode)

    write_arrays.append({'array': deg_sdg, 'xoff': xoff, 'yoff': yoff})

    ###########################################################
    # Tabulate summaries
    sdg_summary = zonal_total(deg_sdg, cell_areas, mask)
    lc_summary = zonal_total(deg_lc, cell_areas, mask)

    # Make a tabulation of the productivity and SOC summaries without water for
    # usage on Prais4
    mask_plus_water = mask.copy()
    mask_plus_water[water] = True
    prod_summary = {
        'all_cover_types': zonal_total(deg_prod3, cell_areas, mask),
        'non_water': zonal_total(deg_prod3, cell_areas, mask_plus_water)
    }

    soc_summary = {
        'all_cover_types': zonal_total(deg_soc, cell_areas, mask),
        'non_water': zonal_total(deg_soc, cell_areas, mask_plus_water)
    }

    ###########################################################
    # Population affected by degradation

    pop_rows_total = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field='type', field_filter='total'
    )
    pop_rows_male = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field='type', field_filter='male'
    )
    pop_rows_female = params.in_df.indices_for_name(
        config.POPULATION_BAND_NAME, field='type', field_filter='female'
    )

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

        pop_array_male = in_array[pop_rows_male[0], :, :].astype(np.float64)
        pop_array_male_masked = pop_array_male.copy()
        pop_array_male_masked[pop_array_male == config.NODATA_VALUE] = 0
        sdg_zonal_population_male = zonal_total(
            deg_sdg, pop_array_male_masked, mask
        )

        pop_array_female = in_array[pop_rows_female[0], :, :].astype(
            np.float64
        )
        pop_array_female_masked = pop_array_female.copy()
        pop_array_female_masked[pop_array_female == config.NODATA_VALUE] = 0
        sdg_zonal_population_female = zonal_total(
            deg_sdg, pop_array_female_masked, mask
        )

        pop_array_total = pop_array_male + pop_array_female
        pop_array_total_masked = pop_array_male_masked + pop_array_female_masked

    sdg_zonal_population_total = zonal_total(
        deg_sdg, pop_array_total_masked, mask
    )

    # Save SO3 array
    pop_array_total[deg_sdg == -1] = -pop_array_total[deg_sdg == -1]
    # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
    # means LC transitions that indicate to/from water as deg/imp will be
    # masked out
    pop_array_total[water] = config.NODATA_VALUE
    write_arrays.append({'array': pop_array_total, 'xoff': xoff, 'yoff': yoff})

    if pop_by_sex:
        # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
        # means LC transitions that indicate to/from water as deg/imp will be
        # masked out
        pop_array_female[water] = config.NODATA_VALUE
        pop_array_female[deg_sdg == -1] = -pop_array_female[deg_sdg == -1]
        write_arrays.append(
            {
                'array': pop_array_female,
                'xoff': xoff,
                'yoff': yoff
            }
        )
        # Set water to NODATA_VALUE as requested by UNCCD for Prais. Note this
        # means LC transitions that indicate to/from water as deg/imp will be
        # masked out
        pop_array_male[water] = config.NODATA_VALUE
        pop_array_male[deg_sdg == -1] = -pop_array_male[deg_sdg == -1]
        write_arrays.append(
            {
                'array': pop_array_male,
                'xoff': xoff,
                'yoff': yoff
            }
        )

    if params.prod_mode == 'Trends.Earth productivity':
        write_arrays.append({'array': deg_prod5, 'xoff': xoff, 'yoff': yoff})

    return (
        models.SummaryTableLD(
            soc_by_lc_annual_totals, lc_annual_totals, lc_trans_zonal_areas,
            lc_trans_zonal_areas_periods, lc_trans_prod_bizonal,
            lc_trans_zonal_soc_initial, lc_trans_zonal_soc_final,
            sdg_zonal_population_total, sdg_zonal_population_male,
            sdg_zonal_population_female, sdg_summary, prod_summary,
            soc_summary, lc_summary
        ), write_arrays
    )


def _get_n_pop_band_for_type(dfs, pop_type):
    n_bands = 0

    for df in dfs:
        n_bands += len(
            df.indices_for_name(
                config.POPULATION_BAND_NAME,
                field='type',
                field_filter=pop_type
            )
        )

    return n_bands


def _have_pop_by_sex(pop_dfs):
    n_total_pop_bands = _get_n_pop_band_for_type(pop_dfs, 'total')
    n_female_pop_bands = _get_n_pop_band_for_type(pop_dfs, 'female')
    n_male_pop_bands = _get_n_pop_band_for_type(pop_dfs, 'male')

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
            'population_year': prod_params['year_final'],
            'deg_year_initial': prod_params['year_initial'],
            'deg_year_final': prod_params['year_final'],
            'type': population_type
        },
        add_to_map=False
    )


def _process_region(
    wkt_aoi,
    pixel_aligned_bbox,
    in_dfs: List[DataFile],
    output_sdg_path: Path,
    output_layers_path: Path,
    prod_mode: str,
    lc_legend_nesting: land_cover.LCLegendNesting,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg,
    reproject_worker_process_name,
    mask_worker_process_name,
    deg_worker_process_name,
    period_name: str,
    periods: dict,
    reproject_worker_function: Callable = None,
    reproject_worker_params: dict = None,
    mask_worker_function: Callable = None,
    mask_worker_params: dict = None,
    deg_worker_function: Callable = None,
    deg_worker_params: dict = None
) -> Tuple[Optional[models.SummaryTableLD], str]:
    '''Runs summary statistics for a particular area'''
    # build vrt
    # Combines SDG 15.3.1 input raster into a VRT and crop to the AOI
    indic_vrt = tempfile.NamedTemporaryFile(
        suffix='_ld_summary_inputs.vrt', delete=False
    ).name
    logger.info(f'Saving indicator VRT to: {indic_vrt}')
    gdal.BuildVRT(
        indic_vrt, [item.path for item in in_dfs],
        outputBounds=pixel_aligned_bbox,
        resolution='highest',
        resampleAlg=gdal.GRA_NearestNeighbour,
        separate=True
    )

    error_message = ""
    logger.info(f'Reprojecting inputs and saving to {output_layers_path}')

    if reproject_worker_function:
        reproject_result = reproject_worker_function(
            indic_vrt, str(output_layers_path), **reproject_worker_params
        )

    else:
        reproject_worker = workers.Warp(indic_vrt, str(output_layers_path))
        reproject_result = reproject_worker.work()

    if reproject_result:
        # Compute a mask layer that will be used in the tabulation code to
        # mask out areas outside of the AOI. Do this instead of using
        # gdal.Clip to save having to clip and rewrite all of the layers in
        # the VRT
        mask_tif = tempfile.NamedTemporaryFile(
            suffix='_ld_summary_mask.tif', delete=False
        ).name
        logger.info(f'Saving mask to {mask_tif}')
        geojson = util.wkt_geom_to_geojson_file_string(wkt_aoi)

        if mask_worker_function:
            mask_result = mask_worker_function(
                mask_tif, geojson, str(output_layers_path),
                **mask_worker_params
            )

        else:
            mask_worker = workers.Mask(
                mask_tif, geojson, str(output_layers_path)
            )
            mask_result = mask_worker.work()

        if mask_result:
            in_df = combine_data_files(output_layers_path, in_dfs)

            n_out_bands = 2

            if _have_pop_by_sex(in_dfs):
                n_out_bands += 2

            if prod_mode == 'Trends.Earth productivity':
                model_band_number = in_df.index_for_name(
                    config.TRAJ_BAND_NAME
                ) + 1
                # Save the combined productivity indicator as well, in the
                # second layer in the deg file
                n_out_bands += 1
            else:
                model_band_number = in_df.index_for_name(
                    config.LC_DEG_BAND_NAME
                ) + 1

            logger.info(
                f'Calculating summary table and saving to {output_sdg_path}'
            )

            params = models.DegradationSummaryParams(
                in_df=in_df,
                prod_mode=prod_mode,
                in_file=in_df.path,
                out_file=str(output_sdg_path),
                model_band_number=model_band_number,
                n_out_bands=n_out_bands,
                mask_file=mask_tif,
                nesting=lc_legend_nesting,
                trans_matrix=lc_trans_matrix,
                period_name=period_name,
                periods=periods
            )

            if deg_worker_function:
                result = deg_worker_function(params, **deg_worker_params)
            else:
                summarizer = worker.DegradationSummary(
                    params, _process_block_summary
                )
                result = summarizer.work()

            if not result:
                if result.is_killed():
                    error_message = "Cancelled calculation of summary table."
                else:
                    error_message = "Error calculating summary table."
                    result = None
            else:
                result = _accumulate_ld_summary_tables(result)

        else:
            error_message = "Error creating mask."
            result = None

    else:
        error_message = "Error reprojecting layers."
        result = None

    return result, error_message


def _compute_ld_summary_table(
    aoi, in_dfs, compute_bbs_from, prod_mode, output_job_path: Path,
    lc_legend_nesting: land_cover.LCLegendNesting,
    lc_trans_matrix: land_cover.LCTransitionDefinitionDeg, period_name: str,
    periods: dict
) -> Tuple[models.SummaryTableLD, Path, Path]:
    """Computes summary table and the output tif file(s)"""

    wkt_aois = aoi.meridian_split(as_extent=False, out_format='wkt')
    bbs = aoi.get_aligned_output_bounds(compute_bbs_from)
    assert len(wkt_aois) == len(bbs)

    output_name_pattern = {
        1: f"{output_job_path.stem}" + "_{layer}.tif",
        2: f"{output_job_path.stem}" + "{layer}_{index}.tif"
    }[len(wkt_aois)]
    reproject_name_fragment = {
        1: "Reprojecting layers",
        2: "Reprojecting layers (part {index} of 2)",
    }[len(wkt_aois)]
    mask_name_fragment = {
        1: "Generating mask",
        2: "Generating mask (part {index} of 2)",
    }[len(wkt_aois)]
    summary_name_fragment = {
        1: "Calculating summary table",
        2: "Calculating summary table (part {index} of 2)",
    }[len(wkt_aois)]
    stable_kwargs = {
        "in_dfs": in_dfs,
        "prod_mode": prod_mode,
        "lc_legend_nesting": lc_legend_nesting,
        "lc_trans_matrix": lc_trans_matrix,
        "period_name": period_name,
        "periods": periods,
    }

    summary_tables = []
    reproj_paths = []
    sdg_paths = []

    for index, (wkt_aoi,
                pixel_aligned_bbox) in enumerate(zip(wkt_aois, bbs), start=1):
        sdg_path = output_job_path.parent / output_name_pattern.format(
            layer="sdg", index=index
        )
        sdg_paths.append(sdg_path)
        reproj_path = output_job_path.parent / output_name_pattern.format(
            layer="inputs", index=index
        )
        reproj_paths.append(reproj_path)

        result, error_message = _process_region(
            wkt_aoi=wkt_aoi,
            pixel_aligned_bbox=pixel_aligned_bbox,
            output_sdg_path=sdg_path,
            output_layers_path=reproj_path,
            reproject_worker_process_name=reproject_name_fragment.format(
                index=index
            ),
            mask_worker_process_name=mask_name_fragment.format(index=index),
            deg_worker_process_name=summary_name_fragment.format(index=index),
            **stable_kwargs
        )

        if result is None:
            raise RuntimeError(error_message)
        else:
            summary_tables.append(result)

    summary_table = _accumulate_ld_summary_tables(summary_tables)

    if len(reproj_paths) > 1:
        reproj_path = output_job_path.parent / f"{output_job_path.stem}_inputs.vrt"
        gdal.BuildVRT(str(reproj_path), [str(p) for p in reproj_paths])
    else:
        reproj_path = reproj_paths[0]

    if len(sdg_paths) > 1:
        sdg_path = output_job_path.parent / f"{output_job_path.stem}_sdg.vrt"
        gdal.BuildVRT(str(sdg_path), [str(p) for p in sdg_paths])
    else:
        sdg_path = sdg_paths[0]

    return summary_table, sdg_path, reproj_path
