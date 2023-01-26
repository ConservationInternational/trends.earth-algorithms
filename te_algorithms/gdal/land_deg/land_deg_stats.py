import json
import logging
from typing import Dict

import numpy as np
from osgeo import gdal
from osgeo import ogr
from te_schemas.jobs import Job
from te_schemas.results import JsonResults

from . import config
from ..util_numba import calc_cell_area

logger = logging.getLogger(__name__)


def _get_raster_bounds(rds):
    ul_x, x_res, _, ul_y, _, y_res = rds.GetGeoTransform()
    lr_x = ul_x + x_res * rds.RasterXSize
    lr_y = ul_y + y_res * rds.RasterYSize

    return ogr.CreateGeometryFromWkt(
        f"""
        POLYGON((
            {ul_x} {ul_y},
            {lr_x} {ul_y},
            {lr_x} {lr_y},
            {ul_x} {lr_y},
            {ul_x} {ul_y}
        ))"""
    )


def _get_stats_for_band(band_name, masked, cell_areas, nodata):
    this_out = {"area_ha": np.sum(np.logical_not(masked.mask) * cell_areas)}
    if band_name in [
        config.SDG_BAND_NAME,
        config.SDG_STATUS_BAND_NAME,
        config.LC_DEG_BAND_NAME,
        config.LC_DEG_COMPARISON_BAND_NAME,
    ]:
        this_out["degraded_pct"] = (
            np.sum((masked == -1) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["stable_pct"] = (
            np.sum((masked == 0) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["improved_pct"] = (
            np.sum((masked == 1) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["nodata_pct"] = (
            np.sum((masked == nodata) * cell_areas) / this_out["area_ha"] * 100
        )
    elif band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        this_out["degraded_pct"] = (
            np.sum(np.logical_or(masked == 1, masked == 2) * cell_areas)
            / this_out["area_ha"]
            * 100
        )
        this_out["stable_pct"] = (
            np.sum(np.logical_or(masked == 3, masked == 4) * cell_areas)
            / this_out["area_ha"]
            * 100
        )
        this_out["improved_pct"] = (
            np.sum((masked == 5) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["nodata_pct"] = (
            np.sum(np.logical_or(masked == nodata, masked == 0) * cell_areas)
            / this_out["area_ha"]
            * 100
        )
    elif band_name == config.SOC_DEG_BAND_NAME:
        this_out["degraded_pct"] = (
            np.sum(np.logical_and(masked <= -10, masked >= -101) * cell_areas)
            / this_out["area_ha"]
            * 100
        )
        this_out["stable_pct"] = (
            np.sum((masked == 0) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["improved_pct"] = (
            np.sum((masked >= 10) * cell_areas) / this_out["area_ha"] * 100
        )
        this_out["nodata_pct"] = (
            np.sum((masked == nodata) * cell_areas) / this_out["area_ha"] * 100
        )

    # Convert from numpy types so they can be serialized
    for key, value in this_out.items():
        this_out[key] = float(value)

    logger.debug("Got stats")
    return this_out


def get_stats_for_geom(raster_path, band_name, band, geom, nodata_value=None):
    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if not rds:
        raise Exception("Failed to open raster.")
    rb = rds.GetRasterBand(band)
    if not rb:
        raise Exception("Band {} not found.".format(rb))
    ul_x, x_res, _, ul_y, _, y_res = rds.GetGeoTransform()

    raster_bounds = _get_raster_bounds(rds)

    # Ignore any areas of polygon that fall outside of the raster
    geom = geom.Intersection(raster_bounds)
    if geom.GetArea() == 0:
        raise Exception("Geom does not overlap raster")

    x_min, x_max, y_min, y_max = geom.GetEnvelope()

    ul_col = int((x_min - ul_x) / x_res)
    lr_col = int((x_max - ul_x) / x_res)
    ul_row = int((y_max - ul_y) / y_res)
    lr_row = int((y_min - ul_y) / y_res)
    width = lr_col - ul_col
    height = abs(ul_row - lr_row)

    src_offset = (ul_col, ul_row, width, height)

    src_array = rb.ReadAsArray(*src_offset)

    new_gt = (
        (ul_x + (src_offset[0] * x_res)),
        x_res,
        0.0,
        (ul_y + (src_offset[1] * y_res)),
        0.0,
        y_res,
    )

    logger.debug("Creating vector layer for geom")
    mem_drv = ogr.GetDriverByName("Memory")
    mem_ds = mem_drv.CreateDataSource("out")
    geom_layer = mem_ds.CreateLayer("poly", None, ogr.wkbPolygon)
    ogr_geom = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
    ft = ogr.Feature(geom_layer.GetLayerDefn())
    ft.SetGeometry(ogr_geom)
    geom_layer.CreateFeature(ft)
    ft.Destroy()

    logger.debug("Rasterizing geom")
    driver = gdal.GetDriverByName("MEM")
    rvds = driver.Create("", width, height, 1, gdal.GDT_Byte)
    rvds.SetGeoTransform(new_gt)
    gdal.RasterizeLayer(rvds, [1], geom_layer, burn_values=[1])
    rv_array = rvds.ReadAsArray()
    src_array = np.nan_to_num(src_array)
    masked = np.ma.MaskedArray(
        src_array,
        mask=np.logical_not(rv_array),
    )

    # Convert areas to hectares
    logger.debug("Calculating cell areas")
    cell_areas_raw = (
        np.array(
            [
                calc_cell_area(
                    y_min + y_res * n,
                    y_min + y_res * (n + 1),
                    x_res,
                )
                for n in range(height)
            ]
        )
        * 1e-4
    )  # 1e-4 is to convert from meters to hectares
    cell_areas_raw.shape = (cell_areas_raw.size, 1)
    cell_areas = np.repeat(cell_areas_raw, masked.shape[1], axis=1)

    logger.debug("Getting stats")
    if nodata_value is None:
        nodata_value = rb.GetNoDataValue()
    return _get_stats_for_band(band_name, masked, cell_areas, nodata_value)


def _calc_features_stats(geojson, raster_path, band_name, band: int):
    out = {"band_name": band_name, "stats": {}}

    for layer in ogr.Open(json.dumps(geojson)):
        for feature in layer:
            geom = feature.geometry()
            out["stats"][feature.GetField("uuid")] = get_stats_for_geom(
                raster_path, band_name, band, geom
            )

    return out


def calculate_statistics(params: Dict) -> Job:
    stats = {}

    for band in params["band_datas"]:
        results = _calc_features_stats(
            params["error_polygons"], params["path"], band["name"], band["index"]
        )
        stats[results["band_name"]] = results["stats"]

    # Before reorganizing the dictionary ensure all stats have the same set of uuids
    band_names = [*stats.keys()]
    uuids = [*stats[band_names[0]].keys()]
    if len([*stats.keys()]) > 1:
        for band_name in band_names[1:]:
            these_uuids = [*stats[band_name].keys()]
            assert set(uuids) == set(these_uuids)

    # reorganize stats so they are keyed by uuid and then by band_name
    out = {}
    for uuid in uuids:
        stat_dict = {band_name: stats[band_name][uuid] for band_name in stats.keys()}
        out[uuid] = stat_dict

    return JsonResults(name="sdg-15-3-1-statistics", data={"stats": out})
