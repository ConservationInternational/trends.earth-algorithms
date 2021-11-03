import dataclasses
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from osgeo import gdal

from te_schemas.datafile import DataFile


# Function to get a temporary filename that handles closing the file created by
# NamedTemporaryFile - necessary when the file is for usage in another process
# (i.e. GDAL)
def _get_temp_filename(suffix):
    f = NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return f.name


@dataclasses.dataclass()
class DroughtSummaryParams:
    in_df: DataFile
    out_file: str
    drought_period: int
    mask_file: str


@dataclasses.dataclass()
class Clip:
    in_file: Path
    out_file: Path
    output_bounds: list
    geojson: dict

    def progress_callback(self, *args, **kwargs):
        '''Reimplement to display progress messages'''
        pass

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
            srcNodata=-32768,
            outputBounds=self.output_bounds,
            dstNodata=-32767,
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
                'GDAL_NUM_THREADS=ALL_CPUs',
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
        pass

    def work(self):
        gdal.UseExceptions()

        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format='GTiff',
            srcNodata=-32768,
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=[
                'NUM_THREADS=ALL_CPUs',
                'GDAL_CACHEMAX=500',
            ],
            creationOptions=[
                'COMPRESS=LZW',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUs',
                'GDAL_NUM_THREADS=ALL_CPUs',
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

@dataclasses.dataclass()
class Mask:
    out_file: Path
    geojson: dict
    model_file: Path

    def progress_callback(self, *args, **kwargs):
        '''Reimplement to display progress messages'''
        pass

    def work(self):
        json_file = _get_temp_filename('.geojson')
        with open(json_file, 'w') as f:
            json.dump(self.geojson, f, separators=(',', ': '))

        gdal.UseExceptions()

        if self.model_file:
            # Assumes an image with no rotation
            gt=gdal.Info(self.model_file, format='json')['geoTransform']
            x_size, y_size=gdal.Info(self.model_file, format='json')['size']
            x_min=min(gt[0], gt[0] + x_size * gt[1])
            x_max=max(gt[0], gt[0] + x_size * gt[1])
            y_min=min(gt[3], gt[3] + y_size * gt[5])
            y_max=max(gt[3], gt[3] + y_size * gt[5])
            output_bounds=[x_min, y_min, x_max, y_max]
            x_res=gt[1]
            y_res=gt[5]
        else:
            output_bounds=None
            x_res=None
            y_res=None

        res = gdal.Rasterize(
            self.out_file,
            json_file,
            format='GTiff',
            outputBounds=output_bounds,
            initValues=-32767,  # Areas that are masked out
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
        pass

    def work(self):
        gdal.UseExceptions()

        res = gdal.Translate(
            self.out_file,
            self.in_file,
            creationOptions=['COMPRESS=LZW'],
            callback=self.progress_callback
        )

        if res:
            return True
        else:
            return None
