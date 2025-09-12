import json
import logging
from typing import Dict

import numpy as np
from osgeo import gdal, ogr
from te_schemas.jobs import Job
from te_schemas.results import JsonResults

from ..util_numba import calc_cell_area
from . import config

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


def _get_degraded_mask(band_name, masked):
    """Get boolean mask for degraded values based on band type."""
    if band_name in [
        config.SDG_BAND_NAME,
        config.LC_DEG_BAND_NAME,
        config.LC_DEG_COMPARISON_BAND_NAME,
    ]:
        return masked == -1
    elif band_name in [config.SDG_STATUS_BAND_NAME]:
        return np.isin(masked, [1, 2, 3])
    elif band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        return np.isin(masked, [1, 2])
    elif band_name == config.SOC_DEG_BAND_NAME:
        return np.logical_and(masked <= -10, masked >= -101)
    else:
        return np.zeros_like(masked, dtype=bool)


def _get_stable_mask(band_name, masked):
    """Get boolean mask for stable values based on band type."""
    if band_name in [
        config.SDG_BAND_NAME,
        config.LC_DEG_BAND_NAME,
        config.LC_DEG_COMPARISON_BAND_NAME,
        config.SOC_DEG_BAND_NAME,
    ]:
        return masked == 0
    elif band_name in [config.SDG_STATUS_BAND_NAME]:
        return masked == 4
    elif band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        return np.isin(masked, [3, 4])
    else:
        return np.zeros_like(masked, dtype=bool)


def _get_improved_mask(band_name, masked):
    """Get boolean mask for improved values based on band type."""
    if band_name in [
        config.SDG_BAND_NAME,
        config.LC_DEG_BAND_NAME,
        config.LC_DEG_COMPARISON_BAND_NAME,
    ]:
        return masked == 1
    elif band_name in [config.SDG_STATUS_BAND_NAME]:
        return np.isin(masked, [5, 6, 7])
    elif band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        return masked == 5
    elif band_name == config.SOC_DEG_BAND_NAME:
        return masked >= 10
    else:
        return np.zeros_like(masked, dtype=bool)


def _get_nodata_mask(band_name, masked, nodata):
    """Get boolean mask for nodata values based on band type."""
    nodata_mask = masked == nodata

    # For some band types, 0 is also treated as nodata
    if band_name in [
        config.JRC_LPD_BAND_NAME,
        config.FAO_WOCAT_LPD_BAND_NAME,
        config.TE_LPD_BAND_NAME,
        config.PROD_DEG_COMPARISON_BAND_NAME,
    ]:
        nodata_mask = np.logical_or(nodata_mask, masked == 0)

    return nodata_mask


def _recode_to_common_classes(band_name, masked, nodata):
    """
    Recode band values to common classes: -1=degraded, 0=stable, 1=improved, nodata=nodata
    """
    # Use int16 to accommodate the nodata value (-32768)
    recoded = np.full_like(masked, nodata, dtype=np.int16)

    # Apply masks using helper functions
    recoded = np.where(_get_degraded_mask(band_name, masked), -1, recoded)
    recoded = np.where(_get_stable_mask(band_name, masked), 0, recoded)
    recoded = np.where(_get_improved_mask(band_name, masked), 1, recoded)
    recoded = np.where(_get_nodata_mask(band_name, masked, nodata), nodata, recoded)

    return recoded


def _get_stats_crosstab(
    band_1_name,
    band_2_name,
    masked_1,
    masked_2,
    cell_areas,
    band_1_nodata,
    band_2_nodata,
):
    """
    Generate crosstab statistics for two bands showing degraded/stable/improved transitions.

    Args:
        band_1_name: Name of first band
        band_2_name: Name of second band
        masked_1: Masked array for first band
        masked_2: Masked array for second band
        cell_areas: Array of cell areas in hectares
        band_1_nodata: No data value for first band
        band_2_nodata: No data value for second band

    Returns:
        Dictionary with crosstab statistics
    """
    # Recode both bands to common classes using their respective nodata values
    recoded_1 = _recode_to_common_classes(band_1_name, masked_1, band_1_nodata)
    recoded_2 = _recode_to_common_classes(band_2_name, masked_2, band_2_nodata)

    # Calculate total area (excluding nodata in either band)
    valid_mask = np.logical_and(recoded_1 != band_1_nodata, recoded_2 != band_2_nodata)
    total_area_ha = np.sum(valid_mask * cell_areas)

    if total_area_ha == 0:
        # No valid data
        return {
            "total_area_ha": 0.0,
            "crosstab": {},
            "band_1_name": band_1_name,
            "band_2_name": band_2_name,
        }

    # Define class names
    class_names = {-1: "degraded", 0: "stable", 1: "improved"}

    # Initialize crosstab dictionary
    crosstab = {}
    for val_1 in [-1, 0, 1]:
        class_1 = class_names[val_1]
        crosstab[class_1] = {}
        for val_2 in [-1, 0, 1]:
            class_2 = class_names[val_2]
            # Calculate area for this combination
            mask_combo = np.logical_and(recoded_1 == val_1, recoded_2 == val_2)
            area_ha = np.sum(mask_combo * cell_areas)
            area_pct = (area_ha / total_area_ha * 100) if total_area_ha > 0 else 0.0

            crosstab[class_1][class_2] = {
                "area_ha": float(area_ha),
                "area_pct": float(area_pct),
            }

    # Calculate marginal totals for each band
    marginals_1 = {}
    marginals_2 = {}

    for val in [-1, 0, 1]:
        class_name = class_names[val]

        # Band 1 marginals
        mask_1 = np.logical_and(recoded_1 == val, valid_mask)
        area_1 = np.sum(mask_1 * cell_areas)
        marginals_1[class_name] = {
            "area_ha": float(area_1),
            "area_pct": float(
                (area_1 / total_area_ha * 100) if total_area_ha > 0 else 0.0
            ),
        }

        # Band 2 marginals
        mask_2 = np.logical_and(recoded_2 == val, valid_mask)
        area_2 = np.sum(mask_2 * cell_areas)
        marginals_2[class_name] = {
            "area_ha": float(area_2),
            "area_pct": float(
                (area_2 / total_area_ha * 100) if total_area_ha > 0 else 0.0
            ),
        }

    return {
        "total_area_ha": float(total_area_ha),
        "crosstab": crosstab,
        "marginals_1": marginals_1,
        "marginals_2": marginals_2,
        "band_1_name": band_1_name,
        "band_2_name": band_2_name,
    }


def _get_stats_for_band(band_name, masked, cell_areas, nodata):
    this_out = {"area_ha": np.sum(np.logical_not(masked.mask) * cell_areas)}

    # Use helper functions for consistent classification logic
    degraded_mask = _get_degraded_mask(band_name, masked)
    stable_mask = _get_stable_mask(band_name, masked)
    improved_mask = _get_improved_mask(band_name, masked)
    nodata_mask = _get_nodata_mask(band_name, masked, nodata)

    this_out["degraded_pct"] = (
        np.sum(degraded_mask * cell_areas) / this_out["area_ha"] * 100
    )
    this_out["stable_pct"] = (
        np.sum(stable_mask * cell_areas) / this_out["area_ha"] * 100
    )
    this_out["improved_pct"] = (
        np.sum(improved_mask * cell_areas) / this_out["area_ha"] * 100
    )
    this_out["nodata_pct"] = (
        np.sum(nodata_mask * cell_areas) / this_out["area_ha"] * 100
    )

    # Convert from numpy types so they can be serialized
    for key, value in this_out.items():
        this_out[key] = float(value)

    logger.debug("Got stats")
    return this_out


def get_stats_for_geom(raster_path, band_info, geom, nodata_value=None, crosstabs=None):
    """
    Calculate statistics for multiple bands over the same geometry.

    Args:
        raster_path: Path to the raster file
        band_info: List of dicts with 'name' and 'index' keys
        geom: Geometry to calculate stats for
        nodata_value: Optional nodata value to use (if None, uses band's nodata value)
        crosstabs: Optional list of dicts with 'band_1' and 'band_2' keys specifying
                  band names for crosstab calculations. Band names must exist in band_info.

    Returns:
        Dict with band names as keys and their stats as values, plus 'crosstabs' key
        if crosstabs were requested
    """

    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if not rds:
        raise Exception("Failed to open raster.")

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

    # Convert areas to hectares (calculate once for all bands)
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
    cell_areas = np.repeat(cell_areas_raw, width, axis=1)

    # Calculate stats for each band
    results = {}
    band_arrays = {}  # Store arrays for crosstab calculations
    logger.debug("Getting stats for {} bands".format(len(band_info)))

    for band_data in band_info:
        band_name = band_data["name"]
        band_index = band_data["index"]

        rb = rds.GetRasterBand(band_index)
        if not rb:
            raise Exception("Band {} not found.".format(band_index))

        src_array = rb.ReadAsArray(*src_offset)
        src_array = np.nan_to_num(src_array)
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_not(rv_array),
        )

        band_nodata_value = nodata_value
        if band_nodata_value is None:
            band_nodata_value = rb.GetNoDataValue()

        results[band_name] = _get_stats_for_band(
            band_name, masked, cell_areas, band_nodata_value
        )

        # Store masked array for potential crosstab calculations
        if crosstabs:
            band_arrays[band_name] = masked

    # Calculate crosstabs if requested
    if crosstabs:
        # Validate crosstab specifications
        band_names_set = {band["name"] for band in band_info}
        for crosstab_spec in crosstabs:
            if "band_1" not in crosstab_spec or "band_2" not in crosstab_spec:
                raise ValueError(
                    "Each crosstab specification must have 'band_1' and 'band_2' keys"
                )
            if crosstab_spec["band_1"] not in band_names_set:
                raise ValueError(
                    f"Band '{crosstab_spec['band_1']}' not found in band_info"
                )
            if crosstab_spec["band_2"] not in band_names_set:
                raise ValueError(
                    f"Band '{crosstab_spec['band_2']}' not found in band_info"
                )

        results["crosstabs"] = []
        for crosstab_spec in crosstabs:
            band_1_name = crosstab_spec["band_1"]
            band_2_name = crosstab_spec["band_2"]

            # Get nodata values for both bands
            band_1_nodata = nodata_value
            band_2_nodata = nodata_value
            if band_1_nodata is None:
                band_1_index = next(
                    b["index"] for b in band_info if b["name"] == band_1_name
                )
                band_1_nodata = rds.GetRasterBand(band_1_index).GetNoDataValue()
            if band_2_nodata is None:
                band_2_index = next(
                    b["index"] for b in band_info if b["name"] == band_2_name
                )
                band_2_nodata = rds.GetRasterBand(band_2_index).GetNoDataValue()

            # Use default nodata values if not specified
            if band_1_nodata is None:
                band_1_nodata = -32768  # Default nodata value
            if band_2_nodata is None:
                band_2_nodata = -32768  # Default nodata value

            crosstab_stats = _get_stats_crosstab(
                band_1_name,
                band_2_name,
                band_arrays[band_1_name],
                band_arrays[band_2_name],
                cell_areas,
                band_1_nodata,
                band_2_nodata,
            )
            results["crosstabs"].append(crosstab_stats)

    return results


def _calc_features_stats(geojson, raster_path, band_datas, crosstabs=None):
    """Calculate stats for multiple bands efficiently by processing each geometry once."""
    # Prepare band info list
    band_info = [{"name": band["name"], "index": band["index"]} for band in band_datas]

    # Initialize output structure
    stats = {band["name"]: {} for band in band_datas}
    if crosstabs:
        stats["crosstabs"] = {}

    for layer in ogr.Open(json.dumps(geojson)):
        for feature in layer:
            geom = feature.geometry()
            uuid = feature.GetField("uuid")

            # Get stats for all bands at once
            multi_band_stats = get_stats_for_geom(
                raster_path, band_info, geom, crosstabs=crosstabs
            )

            # Distribute results to each band's stats
            for band_name, band_stats in multi_band_stats.items():
                if band_name == "crosstabs":
                    stats["crosstabs"][uuid] = band_stats
                else:
                    stats[band_name][uuid] = band_stats

    return stats


def calculate_statistics(params: Dict) -> Job:
    crosstabs = params.get("crosstabs", None)
    stats = _calc_features_stats(
        params["error_polygons"],
        params["path"],
        params["band_datas"],
        crosstabs=crosstabs,
    )

    # Before reorganizing the dictionary ensure all stats have the same set of uuids
    band_names = [key for key in stats.keys() if key != "crosstabs"]
    if band_names:
        uuids = [*stats[band_names[0]].keys()]
        if len(band_names) > 1:
            for band_name in band_names[1:]:
                these_uuids = [*stats[band_name].keys()]
                assert set(uuids) == set(these_uuids)
    else:
        uuids = []

    # reorganize stats so they are keyed by uuid and then by band_name
    out = {}
    for uuid in uuids:
        stat_dict = {band_name: stats[band_name][uuid] for band_name in band_names}
        if "crosstabs" in stats:
            stat_dict["crosstabs"] = stats["crosstabs"][uuid]
        out[uuid] = stat_dict

    return JsonResults(name="sdg-15-3-1-statistics", data={"stats": out})
