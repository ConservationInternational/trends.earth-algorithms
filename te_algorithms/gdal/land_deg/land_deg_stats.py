import json
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from os import stat
from typing import Dict

import numpy as np
from osgeo import gdal
from osgeo import ogr
from te_schemas.error_recode import ErrorRecodePolygons
from te_schemas.jobs import Job
from te_schemas.results import JsonResults

from . import config
from ..util_numba import calc_cell_area


def calculate_statistics(params: Dict) -> Job:
    stats = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = []
        band_names = []
        for band in params["band_datas"]:
            band_names.append(band["name"])
            res.append(
                executor.submit(
                    _calc_stats,
                    params["error_polygons"],
                    params["path"],
                    band["name"],
                    band["index"],
                )
            )

    for this_res, band_name in zip(as_completed(res), band_names):
        stats[band_name] = this_res.result()

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


def _calc_stats(geojson, raster, band_name, band: int):
    rds = gdal.Open(raster, gdal.GA_ReadOnly)
    if not rds:
        raise Exception("Failed to open raster.")
    rb = rds.GetRasterBand(band)
    if not rb:
        raise Exception("Band {} not found.".format(rb))
    ul_x, x_res, _, ul_y, _, y_res = rds.GetGeoTransform()
    lr_x = ul_x + x_res * rds.RasterXSize
    lr_y = ul_y + y_res * rds.RasterYSize

    raster_bounds = ogr.CreateGeometryFromWkt(
        f"""
        POLYGON((
            {ul_x} {ul_y},
            {lr_x} {ul_y},
            {lr_x} {lr_y},
            {ul_x} {lr_y}, 
            {ul_x} {ul_y}
        ))
    """
    )

    nodata = rb.GetNoDataValue()

    out = {}

    for layer in ogr.Open(json.dumps(geojson)):
        for feature in layer:
            geom = feature.geometry()
            # Ignore any areas of polygon that fall outside of the raster
            geom = geom.Intersection(raster_bounds)
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

            mem_drv = ogr.GetDriverByName("Memory")
            mem_ds = mem_drv.CreateDataSource("out")
            mem_layer = mem_ds.CreateLayer("poly", None, ogr.wkbPolygon)
            ogr_geom = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
            ft = ogr.Feature(mem_layer.GetLayerDefn())
            ft.SetGeometry(ogr_geom)
            mem_layer.CreateFeature(ft)
            ft.Destroy()

            driver = gdal.GetDriverByName("MEM")
            rvds = driver.Create("", width, height, 1, gdal.GDT_Byte)
            rvds.SetGeoTransform(new_gt)
            gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
            rv_array = rvds.ReadAsArray()
            src_array = np.nan_to_num(src_array)
            masked = np.ma.MaskedArray(
                src_array,
                mask=np.logical_not(rv_array),
            )

            # Convert areas to hectares
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

            out[feature.GetField("uuid")] = _get_stats_for_band(
                band_name, masked, cell_areas, nodata
            )

    return out


def _get_stats_for_band(band_name, masked, cell_areas, nodata):
    this_out = {"area_ha": np.sum(np.logical_not(masked.mask) * cell_areas)}
    if band_name in [
        config.SDG_BAND_NAME,
        config.SDG_STATUS_BAND_NAME,
        config.LC_DEG_BAND_NAME,
        config.LC_DEG_COMPARISON_BAND_NAME,
    ]:
        this_out["degraded_ha"] = np.sum((masked == -1) * cell_areas)
        this_out["stable_ha"] = np.sum((masked == 0) * cell_areas)
        this_out["improved_ha"] = np.sum((masked == 1) * cell_areas)
    elif band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        this_out["degraded_ha"] = np.sum(
            np.sum(np.logical_or(masked == 1, masked == 2) * cell_areas)
        )
        this_out["stable_ha"] = np.sum(
            np.sum(np.logical_or(masked == 3, masked == 4) * cell_areas)
        )
        this_out["improved_ha"] = np.sum((masked == 5) * cell_areas)
    elif band_name == config.SOC_DEG_BAND_NAME:
        this_out["degraded_ha"] = np.sum(
            np.logical_and(masked <= -10, masked >= -101) * cell_areas
        )
        this_out["stable_ha"] = np.sum((masked == 0) * cell_areas)
        this_out["improved_ha"] = np.sum((masked >= 10) * cell_areas)

    this_out["nodata"] = np.sum((masked == nodata) * cell_areas)

    # Convert from numpy types so they can be serialized
    checksum = 0
    for key, value in this_out.items():
        this_out[key] = float(value)
        if key != "area_ha":
            checksum += float(value)
    this_out["checksum"] = checksum

    return this_out
