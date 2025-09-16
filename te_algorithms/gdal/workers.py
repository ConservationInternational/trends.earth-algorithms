import dataclasses
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional, Union

import numpy as np
from osgeo import gdal

from . import util

NODATA_VALUE = -32768
MASK_VALUE = -32767

logger = logging.getLogger(__name__)

# Set global GDAL optimizations for faster tile processing
gdal.SetConfigOption("GDAL_CACHEMAX", "1024")  # Larger cache for better performance
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")  # Use all CPU threads
gdal.SetConfigOption(
    "GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR"
)  # Skip directory scanning
gdal.SetConfigOption("VSI_CACHE", "TRUE")  # Enable VSI caching
gdal.SetConfigOption("VSI_CACHE_SIZE", "50000000")  # 50MB VSI cache
gdal.SetConfigOption("GDAL_MAX_DATASET_POOL_SIZE", "100")  # Increase dataset pool


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
    progress_callback: Callable


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
    min_tile_size=1024 * 2,  # Reduced from 4KB to 2KB for faster processing
    max_tile_size=2048 * 4,  # Reduced from 6KB to 4KB for faster processing
):
    """Optimized tile size calculation using smart algorithm"""
    n_pixels = img_width * img_height

    # Memory-aware scaling: estimate available memory per CPU
    try:
        import psutil

        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # More aggressive memory usage: 1GB per CPU for faster processing
        memory_per_cpu_gb = 1.0
        max_cpus_by_memory = max(1, int(available_memory_gb / memory_per_cpu_gb))
        effective_cpus = max(1, min(n_cpus, max_cpus_by_memory))
        if effective_cpus < n_cpus:
            logger.info(
                f"Memory-limited: using {effective_cpus} CPUs instead of {n_cpus}"
            )
    except ImportError:
        effective_cpus = n_cpus

    # More aggressive tiling: smaller tiles for better parallelization
    # Target more tiles per CPU for better load balancing
    target_pixels_per_tile = n_pixels / (
        effective_cpus * 1.5
    )  # 50% more tiles for better parallelization

    # Smart tile size calculation using geometric approach
    # Start with square tiles close to target size
    target_tile_side = int(math.sqrt(target_pixels_per_tile))

    # Align to block sizes for optimal I/O
    x_bs_out = _round_to_multiple(target_tile_side, x_bs_initial)
    y_bs_out = _round_to_multiple(target_tile_side, y_bs_initial)

    # Ensure tiles fit constraints
    x_bs_out = max(min_tile_size, min(x_bs_out, max_tile_size, img_width))
    y_bs_out = max(min_tile_size, min(y_bs_out, max_tile_size, img_height))

    # Optimize for better load balancing - adjust to minimize edge tile size difference
    x_bs_out = _optimize_tile_size_for_balance(img_width, x_bs_out, x_bs_initial)
    y_bs_out = _optimize_tile_size_for_balance(img_height, y_bs_out, y_bs_initial)

    n_tiles = math.ceil(img_width / x_bs_out) * math.ceil(img_height / y_bs_out)

    logger.info(
        f"Optimized tile size: {x_bs_out}x{y_bs_out} "
        f"({n_tiles} tiles, ~{target_pixels_per_tile:.0f} pixels/tile)"
    )

    return x_bs_out, y_bs_out


def _round_to_multiple(value, multiple):
    """Round value to nearest multiple"""
    return max(multiple, ((value + multiple // 2) // multiple) * multiple)


def _optimize_tile_size_for_balance(img_dimension, tile_size, block_size):
    """Optimize tile size to minimize load imbalance from edge tiles"""
    if img_dimension <= tile_size:
        return img_dimension

    n_full_tiles = img_dimension // tile_size
    remainder = img_dimension % tile_size

    # If remainder is too small, redistribute to balance load
    if remainder > 0 and remainder < tile_size * 0.3:  # Less than 30% of tile size
        # Reduce tile size slightly to create more balanced tiles
        new_tile_size = img_dimension // (n_full_tiles + 1)
        # Align to block boundaries
        new_tile_size = _round_to_multiple(new_tile_size, block_size)

        # Only use if it doesn't create too small tiles
        if new_tile_size >= block_size * 4:
            return new_tile_size

    return tile_size


def _cut_tiles_multiprocess(n_cpus, params):
    """Optimized multiprocessing with chunking and error handling"""
    logger.info(
        "Using multiprocessing with %d CPUs to process %d tiles", n_cpus, len(params)
    )

    # Optimize chunking for faster tile processing
    # Use smaller chunks to improve responsiveness and load balancing
    chunk_size = max(
        1, min(len(params) // (n_cpus * 6), 4)
    )  # More chunks, better balance
    logger.info("Using chunk size: %d for faster processing", chunk_size)

    results = []
    failed_tiles = []

    try:
        with multiprocessing.get_context("spawn").Pool(n_cpus) as p:
            # Use imap with chunking for better memory efficiency and load balancing
            for i, result in enumerate(p.imap(cut_tile, params, chunksize=chunk_size)):
                if result is None:
                    logger.error(f"Failed to process tile {i}: {params[i].out_file}")
                    failed_tiles.append(i)
                    # Terminate pool immediately on first failure
                    p.terminate()
                    p.join()
                    raise RuntimeError(
                        f"Tile processing failed for tile {i}. This indicates a critical error in tile boundary calculation or GDAL processing. All tiles must be processed successfully to ensure complete analysis."
                    )
                else:
                    logger.debug("Finished processing %s", result)
                results.append(result)

                # More frequent progress updates for better visibility
                if (i + 1) % max(5, len(params) // 20) == 0:
                    logger.info(
                        f"Tile splitting progress: {i + 1}/{len(params)} tiles ({100 * (i + 1) / len(params):.1f}%)"
                    )

    except Exception as e:
        logger.error(f"Multiprocessing failed: {e}")

        # Log system resource information for debugging GDAL errors
        util.log_system_resources(
            context_message="Memory and disk usage at time of GDAL multiprocessing error",
            error_logger=logger,
            input_file=params[0].in_file if params and len(params) > 0 else None,
        )

        raise

    logger.info(
        f"Completed tile splitting: {len(results)} tiles processed successfully"
    )
    return results


def _cut_tiles_sequential(params):
    logger.debug("Cutting tiles sequentially")
    n = 1

    logger.debug("Params are %s ", params)
    results = []
    for param in params:
        result = cut_tile(param)
        if result is None:
            logger.error(f"Failed to process tile {n - 1}: {param.out_file}")
            raise RuntimeError(
                f"Tile processing failed for tile {n - 1}. This indicates a critical error in tile boundary calculation or GDAL processing. All tiles must be processed successfully to ensure complete analysis."
            )
        results.append(result)
        n += 1

    return results


def cut_tile(params):
    logger.debug("Starting tile %s", str(params.out_file))
    #
    gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    try:
        res = gdal.Translate(
            str(params.out_file),
            str(params.in_file),
            srcWin=params.src_win,
            outputType=params.datatype,
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=[
                "BIGTIFF=IF_SAFER",  # Only use BIGTIFF when necessary
                "COMPRESS=LZW",
                "NUM_THREADS=ALL_CPUS",
                "TILED=YES",
                "BLOCKXSIZE=256",  # Smaller block size for faster I/O with smaller tiles
                "BLOCKYSIZE=256",
                "PREDICTOR=2",  # Better compression for integer data
            ],
            callback=params.progress_callback,
            callback_data=[str(params.in_file), str(params.out_file)],
        )
    except Exception as gdal_error:
        logger.error(f"GDAL Translate error for tile {params.out_file}: {gdal_error}")

        # Log system resource information for debugging GDAL errors
        util.log_system_resources(
            context_message="Memory and disk usage at time of GDAL Translate error",
            error_logger=logger,
            input_file=str(params.in_file),
            output_file=str(params.out_file),
            src_win=params.src_win,
        )

        # Re-raise the original error
        raise gdal_error

    logger.debug("Finished tile %s", str(params.out_file))  # Reduced to debug level

    if res:
        return params.out_file
    else:
        # Log detailed error information for debugging
        logger.error(
            f"gdal.Translate failed for tile: {params.out_file}\n"
            f"  Source file: {params.in_file}\n"
            f"  Source window (x_off, y_off, x_size, y_size): {params.src_win}\n"
            f"  This usually indicates invalid tile boundaries that extend beyond the raster extent."
        )
        return None


@dataclasses.dataclass()
class CutTiles:
    in_file: Path
    n_cpus: int = min(
        multiprocessing.cpu_count() - 1, 8
    )  # Cap at 8 CPUs max, not min 16
    out_file: Optional[Path] = None
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

        # Adaptive CPU usage based on data size - more aggressive for better performance
        n_pixels = width * height
        effective_cpus = self.n_cpus

        if n_pixels > 100_000_000:  # 100M pixels - very large
            effective_cpus = min(
                self.n_cpus, 8
            )  # Increased from 6 to 8 for better performance
            logger.info(
                f"Very large dataset ({n_pixels} pixels): "
                f"using {effective_cpus} CPUs for faster processing"
            )
        elif n_pixels < 5_000_000:  # 5M pixels - small dataset
            effective_cpus = min(
                self.n_cpus, 4
            )  # Increased from 2 to 4 for small datasets
            logger.info(
                f"Small dataset ({n_pixels} pixels): using {effective_cpus} CPUs"
            )
        else:  # Medium dataset
            effective_cpus = self.n_cpus  # Use all available CPUs for medium datasets
            logger.info(
                f"Medium dataset ({n_pixels} pixels): using all {effective_cpus} CPUs"
            )

        tile_size = _get_tile_size(
            width, height, x_block_size, y_block_size, effective_cpus
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
                    x_width = (
                        width - x_start
                    )  # Calculate remaining pixels from current position
                else:
                    x_width = tile_size[0]

                if y_start == y_starts[-1]:
                    y_height = (
                        height - y_start
                    )  # Calculate remaining pixels from current position
                else:
                    y_height = tile_size[1]

                src_wins.append((x_start, y_start, x_width, y_height))

        # Validate all tile boundaries before processing
        for i, (x_start, y_start, x_width, y_height) in enumerate(src_wins):
            if x_start + x_width > width or y_start + y_height > height:
                raise ValueError(
                    f"Invalid tile {i}: boundaries extend beyond raster extent.\n"
                    f"  Raster size: {width}x{height}\n"
                    f"  Tile window: x={x_start}, y={y_start}, width={x_width}, height={y_height}\n"
                    f"  Tile end: x={x_start + x_width}, y={y_start + y_height}"
                )
            if x_width <= 0 or y_height <= 0:
                raise ValueError(
                    f"Invalid tile {i}: negative or zero dimensions.\n"
                    f"  Tile window: x={x_start}, y={y_start}, width={x_width}, height={y_height}"
                )

        del in_ds

        logger.info("Reprojecting into %s tile(s)", len(src_wins))
        logger.debug("Tile src_wins are %s ", src_wins)

        # Handle output file naming
        if self.out_file is None:
            base_out_file = self.in_file.with_suffix(".tif")
        else:
            base_out_file = self.out_file

        if len(src_wins) > 1:
            out_files = [
                base_out_file.parent / (base_out_file.stem + f"_tiles_{n}.tif")
                for n in range(len(src_wins))
            ]
        else:
            out_files = [base_out_file]

        params = [
            CutParams(
                src_win, self.in_file, out_file, self.datatype, self.progress_callback
            )
            for src_win, out_file in zip(src_wins, out_files)
        ]

        if effective_cpus > 1:
            _cut_tiles_multiprocess(effective_cpus, params)
        else:
            _cut_tiles_sequential(params)

        return out_files

    def progress_callback(self, fraction, message, callback_data):
        # Only log significant progress milestones to reduce overhead
        if fraction in [0.25, 0.5, 0.75, 1.0]:
            logger.info(
                "%s - %.0f%%",
                f"Splitting {callback_data[0]} into tile {callback_data[1]}",
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
            x_res = abs(gt[1])
            y_res = abs(gt[5])
            output_bounds = _get_bounding_box(ds)
        else:
            output_bounds = None
            x_res = None
            y_res = None

        logger.debug(
            f"Using output resolution {x_res}, {y_res}, "
            f"and output bounds {output_bounds}"
        )

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
    attributes: Union[str, list]

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
            ds = gdal.Open(str(self.model_file))
            gt = ds.GetGeoTransform()
            x_res = abs(gt[1])  # Use absolute value for resolution
            y_res = abs(gt[5])  # Use absolute value for resolution
            output_bounds = _get_bounding_box(ds)
        else:
            output_bounds = None
            x_res = None
            y_res = None

        res = gdal.Rasterize(
            str(self.out_file),
            json_file,
            format="GTiff",
            outputBounds=output_bounds,
            initValues=NODATA_VALUE,
            attribute=self.attributes,
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
            creationOptions=["COMPRESS=LZW", "NUM_THREADS=ALL_CPUS", "TILED=YES"],
            callback=self.progress_callback,
        )

        if res:
            return True
        else:
            return None
