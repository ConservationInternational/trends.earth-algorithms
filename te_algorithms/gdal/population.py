"""Population units and native-grid normalization before resampling."""

import os
import tempfile
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

from .util_numba import slice_area

COUNTS = "people per pixel"
DENSITY_HA = "people per hectare"
DENSITY_KM2 = "people per square kilometre"
NODATA = -32768


def validate_population_count_grid(dataset):
    """Reject grids unsupported by the native latitude-row area calculation."""
    if dataset is None or not dataset.GetProjection():
        raise ValueError("Population raster has no valid CRS.")
    crs = osr.SpatialReference()
    crs.ImportFromWkt(dataset.GetProjection())
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    for reference in (crs, wgs84):
        reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = dataset.GetGeoTransform()
    if not crs.IsSame(wgs84) or transform[2] != 0 or transform[4] != 0:
        raise ValueError(
            "People-per-pixel population requires an unrotated WGS84 "
            "(EPSG:4326) raster. Convert counts to density on the original "
            "grid, then select people per hectare or people per square "
            "kilometre. Do not reproject raw counts first."
        )


def population_to_counts(values, cell_areas_km2, units=COUNTS):
    values = np.array(values, dtype=np.float64, copy=True)
    invalid = ~np.isfinite(values) | (values == NODATA)
    if units == DENSITY_HA:
        values *= cell_areas_km2 * 100
    elif units == DENSITY_KM2:
        values *= cell_areas_km2
    elif units != COUNTS:
        raise ValueError(f"Unsupported population units: {units!r}")
    invalid |= ~np.isfinite(values)
    values[invalid] = NODATA
    return values


def _cell_areas_hectares(transform, yoff, height):
    """Vectorize the summaries' latitude-strip formula for one block of rows."""
    latitudes = transform[3] + np.arange(yoff, yoff + height + 1) * transform[5]
    if np.any(~np.isfinite(latitudes) | (np.abs(latitudes) > 90)):
        raise ValueError("Cannot calculate valid population pixel areas")
    strips = slice_area(np.deg2rad(latitudes))
    return (np.abs(np.diff(strips) * (transform[1] / 360)) / 10000)[:, None]


def normalize_population_raster(source_path, output_path, units=COUNTS, band_number=1):
    """Write Float32 people/ha on the original grid, respecting masks and nodata.

    Pixel areas use the same WGS84 ellipsoid and latitude-strip functions as
    the land degradation and drought summaries. Counts require an unrotated
    WGS84 grid; density inputs do not require source pixel-area calculations.
    """
    if Path(source_path).resolve() == Path(output_path).resolve():
        raise ValueError("Population source and output must be different files")
    with tempfile.NamedTemporaryFile(
        suffix=".tif", prefix="te_density_", dir=Path(output_path).parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        _write_population_density(source_path, temporary_path, units, band_number)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_population_density(source_path, output_path, units, band_number):
    if units not in (COUNTS, DENSITY_HA, DENSITY_KM2):
        raise ValueError(f"Unsupported population units: {units!r}")
    source = gdal.Open(str(source_path))
    if source is None or not source.GetProjection():
        raise ValueError(f"Population raster has no valid CRS: {source_path}")
    if band_number < 1 or band_number > source.RasterCount:
        raise ValueError(f"Invalid population band number: {band_number}")
    transform = source.GetGeoTransform()
    if units == COUNTS:
        validate_population_count_grid(source)
    band = source.GetRasterBand(band_number)
    destination = gdal.GetDriverByName("GTiff").Create(
        str(output_path),
        source.RasterXSize,
        source.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    destination.SetProjection(source.GetProjection())
    destination.SetGeoTransform(source.GetGeoTransform())
    output = destination.GetRasterBand(1)
    output.SetNoDataValue(NODATA)
    output.SetMetadataItem("units", DENSITY_HA)
    nodata = band.GetNoDataValue()
    scale = band.GetScale()
    offset = band.GetOffset()
    mask = None if band.GetMaskFlags() & gdal.GMF_ALL_VALID else band.GetMaskBand()
    for yoff in range(0, source.RasterYSize, 256):
        height = min(256, source.RasterYSize - yoff)
        row_areas = (
            _cell_areas_hectares(transform, yoff, height) if units == COUNTS else None
        )
        if row_areas is not None and np.any(~np.isfinite(row_areas) | (row_areas <= 0)):
            raise ValueError("Cannot calculate valid population pixel areas")
        for xoff in range(0, source.RasterXSize, 256):
            width = min(256, source.RasterXSize - xoff)
            values = band.ReadAsArray(xoff, yoff, width, height).astype(np.float64)
            invalid = ~np.isfinite(values) | (values == NODATA)
            if nodata is not None:
                invalid |= values == nodata
            if mask is not None:
                invalid |= mask.ReadAsArray(xoff, yoff, width, height) == 0
            if scale is not None and scale != 1:
                values *= scale
            if offset is not None and offset != 0:
                values += offset
            if np.any((values < 0) & ~invalid):
                raise ValueError("Population values must be non-negative")
            if units == COUNTS:
                values /= row_areas
            elif units == DENSITY_KM2:
                values /= 100
            invalid |= ~np.isfinite(values)
            if np.any((values > np.finfo(np.float32).max) & ~invalid):
                raise ValueError("Population density exceeds the Float32 range")
            values[invalid] = NODATA
            output.WriteArray(values.astype(np.float32), xoff, yoff)
    destination.FlushCache()
    destination = None
    source = None
