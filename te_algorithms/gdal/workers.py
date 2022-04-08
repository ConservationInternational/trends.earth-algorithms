import dataclasses
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from osgeo import gdal

from . import util

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
        json_file = _get_temp_filename(".geojson")
        with open(json_file, "w") as f:
            json.dump(self.geojson, f, separators=(",", ": "))

        gdal.UseExceptions()
        gdal.SetConfigOption("GDAL_CACHEMAX", "500")
        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format="GTiff",
            cutlineDSName=json_file,
            srcNodata=NODATA_VALUE,
            outputBounds=self.output_bounds,
            dstNodata=MASK_VALUE,
            dstSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            targetAlignedPixels=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=["COMPRESS=LZW", "NUM_THREADS=ALL_CPUS", "TILED=YES"],
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback,
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None


@dataclasses.dataclass()
class CutParams:
    src_win: tuple
    in_file: Path
    out_file: Path
    datatype: int
    progress_callback: callable


def _next_power_of_two(x):
    i = 1

    while i < x:
        i *= 2

    return i


def _block_sizes_valid(x_bs, y_bs, img_width, img_height, max_pixel_per_cpu):
    if ((x_bs * y_bs) > max_pixel_per_cpu) or (x_bs > img_width) or (y_bs > img_height):
        return False
    else:
        return True


def _get_closest_multiple(base, target_value, max_value):
    "used to get closest multiple of tile size that includes target_value"
    out = base

    while out < target_value:
        out += base

    if out > max_value:
        return max_value

    return out


def _get_tile_size(
    img_width,
    img_height,
    x_bs_initial,
    y_bs_initial,
    n_cpus,
    min_tile_size=1024 * 4,
    max_tile_size=2048 * 6,
):
    n_pixels = img_width * img_height
    max_pixel_per_cpu = math.ceil(n_pixels / n_cpus)

    logger.info("max_pixel_per_cpu %s", max_pixel_per_cpu)

    x_bs_out = x_bs_initial
    y_bs_out = y_bs_initial

    x_stop = y_stop = False

    while not (x_stop and y_stop):
        x_bs_out += x_bs_initial

        if (
            not _block_sizes_valid(
                x_bs_out, y_bs_out, img_width, img_height, max_pixel_per_cpu
            )
            or x_bs_out > max_tile_size
        ):
            x_bs_out -= x_bs_initial
            x_stop = True

        y_bs_out += y_bs_initial

        if (
            not _block_sizes_valid(
                x_bs_out, y_bs_out, img_width, img_height, max_pixel_per_cpu
            )
            or y_bs_out > max_tile_size
        ):
            y_bs_out -= y_bs_initial
            y_stop = True

    if x_bs_out < min_tile_size:
        x_bs_out = _get_closest_multiple(x_bs_initial, min_tile_size, img_width)

    if y_bs_out < min_tile_size:
        y_bs_out = _get_closest_multiple(y_bs_initial, min_tile_size, img_height)

    return x_bs_out, y_bs_out


def _cut_tiles_multiprocess(n_cpus, params):
    logger.debug("Using multiprocessing")
    logger.debug("Params are %s ", params)
    results = []
    with multiprocessing.get_context("spawn").Pool(n_cpus) as p:
        for result in p.imap_unordered(cut_tile, params):
            logger.debug("Finished processing %s", result)
            results.append(result)
    return results


def _cut_tiles_sequential(params):
    logger.debug("Cutting tiles sequentially")
    n = 1

    logger.debug("Params are %s ", params)
    results = []
    for param in params:
        results.append(cut_tile(param))
        n += 1

    return results


def cut_tile(params):
    logger.info("Starting tile %s", str(params.out_file))
    gdal.SetConfigOption("GDAL_CACHEMAX", "500")
    res = gdal.Translate(
        str(params.out_file),
        str(params.in_file),
        srcWin=params.src_win,
        outputType=params.datatype,
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=[
            "BIGTIFF=YES",
            "COMPRESS=LZW",
            "NUM_THREADS=ALL_CPUs",
            "TILED=YES",
        ],
        callback=params.progress_callback,
        callback_data=[str(params.in_file), str(params.out_file)],
    )

    logger.info("Finished tile %s", str(params.out_file))

    if res:
        return params.out_file
    else:
        return None


@dataclasses.dataclass()
class CutTiles:
    in_file: Path
    n_cpus: int = max(multiprocessing.cpu_count() - 1, 16)
    out_file: Path = None
    datatype: int = gdal.GDT_Int16

    def work(self):
        gdal.UseExceptions()

        in_ds = gdal.Open(str(self.in_file))
        xmin, x_res, _, ymax, _, y_res = in_ds.GetGeoTransform()
        width, height = in_ds.RasterXSize, in_ds.RasterYSize

        band = in_ds.GetRasterBand(1)
        x_block_size, y_block_size = band.GetBlockSize()
        logger.debug(
            "Image size %s, %s, block size %s, %s",
            width,
            height,
            x_block_size,
            y_block_size,
        )

        tile_size = _get_tile_size(
            width, height, x_block_size, y_block_size, self.n_cpus
        )
        logger.info("Chose tile size %s", tile_size)

        x_starts = np.arange(0, width, tile_size[0])
        y_starts = np.arange(0, height, tile_size[1])

        src_wins = []

        logger.debug(
            "Tile generation x_starts are %s, y_starts are %s", x_starts, y_starts
        )

        for x_start in x_starts:
            for y_start in y_starts:
                if x_start == x_starts[-1]:
                    x_width = in_ds.RasterXSize
                else:
                    x_width = tile_size[0]

                if y_start == y_starts[-1]:
                    y_height = in_ds.RasterYSize
                else:
                    y_height = tile_size[1]

                src_wins.append((x_start, y_start, x_width, y_height))

        del in_ds

        logger.info("Generating %s tiles", len(src_wins))
        logger.debug("Tile src_wins are %s ", src_wins)

        if len(src_wins) > 1:
            out_files = [
                self.out_file.parent / (self.out_file.stem + f"_{n}.tif")
                for n in range(len(src_wins))
            ]
        else:
            out_files = [self.out_file]

        params = [
            CutParams(
                src_win, self.in_file, out_file, self.datatype, self.progress_callback
            )
            for src_win, out_file in zip(src_wins, out_files)
        ]

        if self.n_cpus > 1:
            _cut_tiles_multiprocess(self.n_cpus, params)
        else:
            _cut_tiles_sequential(params)

        return out_files

    def progress_callback(self, fraction, message, callback_data):
        logger.info(
            "%s - %.2f%%",
            f"Splitting {callback_data[0]} into tile " f"{callback_data[1]}",
            100 * fraction,
        )


@dataclasses.dataclass()
class Warp:
    in_file: Path
    out_file: Path
    compress: str = "LZW"

    def progress_callback(self, *args, **kwargs):
        """Reimplement to display progress messages"""
        util.log_progress(*args, **kwargs)

    def work(self):
        gdal.UseExceptions()

        creationOptions = ["BIGTIFF=YES", "NUM_THREADS=ALL_CPUS", "TILED=YES"]

        gdal.SetConfigOption("GDAL_CACHEMAX", "500")

        if self.compress is not None:
            creationOptions += [f"COMPRESS={self.compress}"]

        # in_ds = gdal.Open(self.in_file)
        # gt = in_ds.GetGeoTransform()
        # x_res = gt[1]
        # y_res = gt[5]

        res = gdal.Warp(
            self.out_file,
            self.in_file,
            format="GTiff",
            srcNodata=NODATA_VALUE,
            outputType=gdal.GDT_Int16,
            resampleAlg=gdal.GRA_NearestNeighbour,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=creationOptions,
            # targetAlignedPixels=True,
            # xRes=x_res,
            # yRes=y_res,
            multithread=True,
            warpMemoryLimit=500,
            callback=self.progress_callback,
        )

        if res:
            return True
        else:
            return None


def _get_bounding_box(ds):
    """Return list of corner coordinates from a gdal Dataset"""
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

    def progress_callback(self, *args, **kwargs):
        """Reimplement to display progress messages"""
        util.log_progress(*args, **kwargs)

    def work(self):
        json_file = _get_temp_filename(".geojson")
        with open(json_file, "w") as f:
            json.dump(self.geojson, f, separators=(",", ": "))

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
            format="GTiff",
            outputBounds=output_bounds,
            initValues=MASK_VALUE,  # Areas that are masked out
            burnValues=1,  # Areas that are NOT masked out
            xRes=x_res,
            yRes=y_res,
            outputSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            creationOptions=["COMPRESS=LZW", "NUM_THREADS=ALL_CPUS", "TILED=YES"],
            callback=self.progress_callback,
        )
        os.remove(json_file)

        if res:
            return True
        else:
            return None


@dataclasses.dataclass()
class Rasterize:
    out_file: Path
    model_file: Path
    geojson: dict
    attribute: str

    def progress_callback(self, *args, **kwargs):
        """Reimplement to display progress messages"""
        util.log_progress(*args, **kwargs)

    def work(self):
        json_file = _get_temp_filename(".geojson")
        with open(json_file, "w") as f:
            json.dump(self.geojson, f, separators=(",", ": "))

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
            format="GTiff",
            outputBounds=output_bounds,
            initValues=NODATA_VALUE,
            attribute=self.attribute,
            xRes=x_res,
            yRes=y_res,
            outputSRS="epsg:4326",
            outputType=gdal.GDT_Int16,
            creationOptions=["COMPRESS=LZW", "NUM_THREADS=ALL_CPUS", "TILED=YES"],
            callback=self.progress_callback,
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
        """Reimplement to display progress messages"""
        util.log_progress(*args, **kwargs)

    def work(self):
        gdal.UseExceptions()

        res = gdal.Translate(
            self.out_file,
            self.in_file,
            creationOptions=["COMPRESS=LZW", "NUM_THREADS=ALL_CPUs", "TILED=YES"],
            callback=self.progress_callback,
        )

        if res:
            return True
        else:
            return None
