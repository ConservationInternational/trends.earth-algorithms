import dataclasses
import json
import os
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from osgeo import gdal

from . import util
from te_schemas.datafile import DataFile

NODATA_VALUE = -32768
MASK_VALUE = -32767

logger = logging.getLogger(__name__)

# Function to get a temporary filename that handles closing the file created by
# NamedTemporaryFile - necessary when the file is for usage in another process
# (i.e. GDAL)
def _get_temp_filename(suffix):
    f = NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return f.name


@dataclasses.dataclass()
class Clip:
    in_file: Path
    out_file: Path
    output_bounds: list
    geojson: dict

    def progress_callback(self, *args, **kwargs):
        util.log_progress(*args, **kwargs)

    def work(self):
        json_file = _get_temp_filename('.geojson')
        with open(json_file, 'w') as f:
            json.dump(self.geojson, f, separators=(',', ': '))

        gdal.UseExceptions()
        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format='GTiff',
            cutlineDSName=json_file,
            srcNodata=NODATA_VALUE,
            outputBounds=self.output_bounds,
            dstNodata=MASK_VALUE,
            dstSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=[
                'NUM_THREADS=ALL_CPUs',
                'GDAL_CACHEMAX=500',
            ],
            creationOptions=[
                'COMPRESS=LZW',
                'NUM_THREADS=ALL_CPUs',
                'TILED=YES'
            ],
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None


@dataclasses.dataclass()
class Warp:
    in_file: Path
    out_file: Path

    def progress_callback(self, *args, **kwargs):
        '''Reimplement to display progress messages'''
        util.log_progress(*args, **kwargs)

    def work(self):
        gdal.UseExceptions()

        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format='GTiff',
            srcNodata=NODATA_VALUE,
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=[
                'NUM_THREADS=ALL_CPUS',
                'GDAL_CACHEMAX=500',
            ],
            creationOptions=[
                'COMPRESS=LZW',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUS',
                'NUM_THREADS=ALL_CPUS',
                'TILED=YES'
            ],
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback
        )
        if res:
            return True
        else:
            return None


def _get_bounding_box(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    return (xmin, ymin, xmax, ymax)


@dataclasses.dataclass()
class Mask:
    out_file: Path
    geojson: dict
    model_file: Path
    force_within_wgs84_globe: bool = True

    def progress_callback(self, *args, **kwargs):
        '''Reimplement to display progress messages'''
        util.log_progress(*args, **kwargs)

    def work(self):
        json_file = _get_temp_filename('.geojson')
        with open(json_file, 'w') as f:
            json.dump(self.geojson, f, separators=(',', ': '))

        gdal.UseExceptions()

        if self.model_file:
            # Assumes an image with no rotation
            ds = gdal.Open(self.model_file)
            gt = ds.GetGeoTransform()
            x_res = gt[1]
            y_res = gt[5]
            output_bounds = _get_bounding_box(ds)
        else:
            output_bounds = None
            x_res = None
            y_res = None

        res = gdal.Rasterize(
            self.out_file,
            json_file,
            format='GTiff',
            outputBounds=output_bounds,
            initValues=MASK_VALUE,  # Areas that are masked out
            burnValues=1,  # Areas that are NOT masked out
            xRes=x_res,
            yRes=y_res,
            outputSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            creationOptions=['COMPRESS=LZW'],
            callback=self.progress_callback
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None

@dataclasses.dataclass()
class Translate:
    out_file: Path
    in_file: Path

    def progress_callback(self, *args, **kwargs):
        '''Reimplement to display progress messages'''
        util.log_progress(*args, **kwargs)

    def work(self):
        gdal.UseExceptions()

        res = gdal.Translate(
            self.out_file,
            self.in_file,
            creationOptions=[
                'COMPRESS=LZW',
                'NUM_THREADS=ALL_CPUs',
                'TILED=YES'
            ],
            callback=self.progress_callback
        )

        if res:
            return True
        else:
            return None
