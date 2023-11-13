import logging
import tempfile
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from osgeo import gdal
from te_schemas import reporting
from te_schemas.aoi import AOI
from te_schemas.datafile import DataFile
from te_schemas.error_recode import ErrorRecodePolygons
from te_schemas.jobs import Job
from te_schemas.results import Band
from te_schemas.results import CloudResults
from te_schemas.results import DataType
from te_schemas.results import JsonResults
from te_schemas.results import Raster
from te_schemas.results import RasterFileType
from te_schemas.results import RasterResults
from te_schemas.results import TiledRaster
from te_schemas.results import URI

from . import config
from .. import workers
from ..util import accumulate_dicts
from ..util import save_vrt
from ..util import wkt_geom_to_geojson_file_string
from ..util_numba import zonal_total
from .land_deg_numba import recode_indicator_errors
from .models import DegradationErrorRecodeSummaryParams
from .models import SummaryTableLDErrorRecode
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

    # TODO: Assumes WGS84 for now
    rasterize_worker = workers.Rasterize(
        str(out_file), str(model_file), error_recode_dict, "error_recode"
    )
    rasterize_worker.work()


def _process_block(
    params: DegradationErrorRecodeSummaryParams,
    in_array,
    mask,
    xoff: int,
    yoff: int,
    cell_areas_raw,
) -> Tuple[SummaryTableLDErrorRecode, Dict]:
    sdg_array = in_array[params.band_dict["sdg_bandnum"] - 1, :, :]
    recode_array = in_array[params.band_dict["recode_bandnum"] - 1, :, :]
    cell_areas = np.repeat(cell_areas_raw, mask.shape[1], axis=1)

    # below works on data in place
    sdg_array = recode_indicator_errors(
        sdg_array, recode_array, *params.trans_code_lists
    )

    sdg_summary = zonal_total(sdg_array, cell_areas, mask)

    write_arrays = [
        {"array": sdg_array, "xoff": xoff, "yoff": yoff},
        {"array": recode_array, "xoff": xoff, "yoff": yoff},
    ]

    return (SummaryTableLDErrorRecode(sdg_summary), write_arrays)


def _accumulate_summary_tables(
    tables: List[SummaryTableLDErrorRecode],
) -> SummaryTableLDErrorRecode:
    if len(tables) == 1:
        return tables[0]
    else:
        out = tables[0]

        for table in tables[1:]:
            out.sdg_summary = accumulate_dicts([out.sdg_summary, table.sdg_summary])

        return out


def _get_error_recode_input_vrt(sdg_df, error_df):
    df_band_list = [
        ("sdg_bandnum", 1),  # there is only 1 band in sdg_df
        ("recode_bandnum", error_df.index_for_name(config.ERROR_RECODE_BAND_NAME)),
    ]

    band_vrts = [
        save_vrt(sdg_df.path, 1),  # there is only 1 band in sdg_df
        save_vrt(
            error_df.path, error_df.index_for_name(config.ERROR_RECODE_BAND_NAME) + 1
        ),
    ]

    out_vrt = tempfile.NamedTemporaryFile(
        suffix="_error_recode_inputs.vrt", delete=False
    ).name
    gdal.BuildVRT(out_vrt, [vrt for vrt in band_vrts], separate=True)

    vrt_band_dict = {item[0]: index for index, item in enumerate(df_band_list, start=1)}

    return out_vrt, vrt_band_dict


def _prepare_df(path, band_str, band_index) -> List[DataFile]:
    band = Band(**band_str)

    return DataFile(path=save_vrt(path, band_index), bands=[band])


def get_serialized_results(st, layer_name):
    sdg_summary = reporting.AreaList(
        "SDG Indicator 15.3.1",
        "sq km",
        [
            reporting.Area("Improved", st.sdg_summary.get(1, 0.0)),
            reporting.Area("Stable", st.sdg_summary.get(0, 0.0)),
            reporting.Area("Degraded", st.sdg_summary.get(-1, 0.0)),
            reporting.Area("No data", st.sdg_summary.get(int(config.NODATA_VALUE), 0)),
        ],
    )
    land_condition_report = reporting.LandConditionReport(
        sdg=reporting.SDG15Report(summary=sdg_summary)
    )

    return reporting.LandConditionReport.Schema().dump(land_condition_report)


def recode_errors(params) -> Job:
    aoi = AOI.Schema().load(params["aoi"])
    job_output_path = Path(params["output_path"])
    sdg_df = _prepare_df(
        params["layer_input_band_path"],
        params["layer_input_band"],
        params["layer_input_band_index"],
    )

    error_recode_df = _prepare_df(
        params["layer_error_recode_path"],
        params["layer_error_recode_band"],
        params["layer_error_recode_band_index"],
    )
    error_polygons = ErrorRecodePolygons.Schema().load(params["error_polygons"])

    input_band = Band.Schema().load(params["layer_input_band"])

    summary_table, error_recode_paths = _compute_error_recode(
        sdg_df=sdg_df,
        error_recode_df=error_recode_df,
        aoi=aoi,
        error_polygons=error_polygons,
        job_output_path=job_output_path.parent / f"{job_output_path.stem}.json",
    )

    if params["write_tifs"]:
        out_bands = [
            Band(
                name=input_band.name,
                no_data_value=int(config.NODATA_VALUE),
                metadata=params["metadata"],  # copy metadata from input job
                add_to_map=True,
                activated=True,
            ),
            Band(
                name=config.ERROR_RECODE_BAND_NAME,
                no_data_value=int(config.NODATA_VALUE),
                metadata=params["metadata"],  # copy metadata from input job
                add_to_map=False,
                activated=False,
            ),
        ]

        if len(error_recode_paths) > 1:
            error_recode_vrt = (
                job_output_path.parent / f"{job_output_path.stem}_error_recode.vrt"
            )
            gdal.BuildVRT(str(error_recode_vrt), [str(p) for p in error_recode_paths])
            rasters = {
                DataType.INT16.value: TiledRaster(
                    tile_uris=[URI(uri=p) for p in error_recode_paths],
                    uri=URI(uri=error_recode_vrt),
                    bands=out_bands,
                    datatype=DataType.INT16,
                    filetype=RasterFileType.COG,
                )
            }
            main_uri = (URI(uri=error_recode_vrt),)
        else:
            rasters = {
                DataType.INT16.value: Raster(
                    uri=URI(uri=error_recode_paths[0]),
                    bands=out_bands,
                    datatype=DataType.INT16,
                    filetype=RasterFileType.COG,
                )
            }
            main_uri = URI(uri=error_recode_paths[0])

        results = RasterResults(
            name=params["layer_input_band"]["name"],
            uri=main_uri,
            rasters=rasters,
            data=get_serialized_results(
                summary_table, params["layer_input_band"]["name"] + " recode"
            ),
        )

    else:
        results = JsonResults(
            name=params["layer_input_band"]["name"],
            data=get_serialized_results(
                summary_table, params["layer_input_band"]["name"] + " recode"
            ),
        )

    return results


def _compute_error_recode(
    sdg_df,
    error_recode_df,
    error_polygons,
    job_output_path,
    aoi,
    mask_worker_function: Callable = None,
    mask_worker_params: dict = None,
    error_recode_worker_function: Callable = None,
    error_recode_worker_params: dict = None,
):
    in_vrt, band_dict = _get_error_recode_input_vrt(sdg_df, error_recode_df)

    wkt_aois = aoi.meridian_split(as_extent=False, out_format="wkt")
    bbs = aoi.get_aligned_output_bounds(sdg_df.path)

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
                mask_tif, geojson, str(cropped_in_vrt), **mask_worker_params
            )
        else:
            mask_worker = workers.Mask(
                mask_tif,
                geojson,
                str(cropped_in_vrt),
            )
            mask_result = mask_worker.work()

        if mask_result:
            out_path = job_output_path.parent / error_recode_name_pattern.format(
                index=index
            )
            error_recode_paths.append(out_path)

            logger.info(f"Calculating error recode and saving layer to: {out_path}")
            error_recode_params = DegradationErrorRecodeSummaryParams(
                in_file=str(cropped_in_vrt),
                out_file=str(out_path),
                band_dict=band_dict,
                model_band_number=1,
                n_out_bands=2,
                mask_file=mask_tif,
                trans_code_lists=error_polygons.trans_code_lists,
            )

            if error_recode_worker_function:
                result = error_recode_worker_function(
                    error_recode_params, _process_block, **error_recode_worker_params
                )
            else:
                summarizer = DegradationSummary(error_recode_params, _process_block)
                result = summarizer.work()

            if not result:
                if result.is_killed():
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
