import logging
import os
from typing import Union

import numpy as np
from osgeo import gdal, osr

from .. import util
from ..util_numba import calc_cell_area
from . import config, models

logger = logging.getLogger(__name__)


class DegradationSummary:
    def __init__(
        self,
        params: Union[
            models.DegradationSummaryParams,
            models.DegradationStatusSummaryParams,
            models.DegradationErrorRecodeSummaryParams,
        ],
        processing_function,
    ):
        self.params = params
        self.processing_function = processing_function

    def is_killed(self):
        return False

    def emit_progress(self, *args, **kwargs):
        """Reimplement to display progress messages - only log significant milestones"""
        if len(args) > 0:
            fraction = args[0]
            # Only log at significant progress milestones to reduce overhead
            if fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                util.log_progress(*args, message="Processing land degradation summary")

    def work(self):
        gdal.UseExceptions()
        mask_ds = gdal.Open(self.params.mask_file)
        band_mask = mask_ds.GetRasterBand(1)

        logger.debug(f"Reading from {self.params.in_file}")
        src_ds = gdal.Open(str(self.params.in_file))

        model_band = src_ds.GetRasterBand(self.params.model_band_number)
        block_sizes = model_band.GetBlockSize()
        xsize = model_band.XSize
        ysize = model_band.YSize

        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]

        driver = gdal.GetDriverByName("GTiff")
        logger.debug(f"Writing to {self.params.out_file}")
        dst_ds = driver.Create(
            str(self.params.out_file),
            xsize,
            ysize,
            self.params.n_out_bands,
            gdal.GDT_Int16,
            options=[
                "COMPRESS=LZW",
                "BIGTIFF=YES",
                "NUM_THREADS=ALL_CPUS",
                "TILED=YES",
            ],
        )
        src_gt = src_ds.GetGeoTransform()
        dst_ds.SetGeoTransform(src_gt)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(src_ds.GetProjectionRef())
        dst_ds.SetProjection(dst_srs.ExportToWkt())

        # Width of cells in longitude
        long_width = src_gt[1]
        # Set initial lat ot the top left corner latitude
        lat = src_gt[3]
        # Width of cells in latitude
        pixel_height = src_gt[5]

        x_blocks = (xsize + x_block_size - 1) // x_block_size  # Ceiling division
        y_blocks = (ysize + y_block_size - 1) // y_block_size  # Ceiling division
        n_blocks = x_blocks * y_blocks

        # Use adaptive block sizing for better parallelism
        n_pixels = xsize * ysize
        if n_pixels > 10_000_000:  # 10M pixels
            # Use larger blocks for efficiency
            x_block_size = min(x_block_size * 2, 1024)
            y_block_size = min(y_block_size * 2, 1024)
            # Recalculate block counts
            x_blocks = (xsize + x_block_size - 1) // x_block_size
            y_blocks = (ysize + y_block_size - 1) // y_block_size
            n_blocks = x_blocks * y_blocks
            logger.info(
                f"Large dataset: Using {x_block_size}x{y_block_size} blocks "
                f"({n_blocks} total)"
            )
        elif n_pixels < 1_000_000:  # 1M pixels
            # Use smaller blocks for better granularity
            x_block_size = max(x_block_size // 2, 256)
            y_block_size = max(y_block_size // 2, 256)
            # Recalculate block counts
            x_blocks = (xsize + x_block_size - 1) // x_block_size
            y_blocks = (ysize + y_block_size - 1) // y_block_size
            n_blocks = x_blocks * y_blocks
            logger.info(
                f"Small dataset: Using {x_block_size}x{y_block_size} blocks "
                f"({n_blocks} total)"
            )

        # Pre-calculate all cell areas for the entire raster to avoid repeated
        # computation
        all_cell_areas = np.empty(ysize, dtype=np.float64)
        current_lat = lat
        for row in range(ysize):
            all_cell_areas[row] = (
                calc_cell_area(
                    current_lat,
                    current_lat + pixel_height,
                    long_width,
                )
                * 1e-6
            )  # Convert from meters to kilometers
            current_lat += pixel_height

        # Cache raster bands to avoid repeated access
        src_bands = [src_ds.GetRasterBand(i) for i in range(1, src_ds.RasterCount + 1)]
        dst_bands = [
            dst_ds.GetRasterBand(i) for i in range(1, self.params.n_out_bands + 1)
        ]

        progress_increment = 1.0 / n_blocks
        n = 0

        # Pre-allocate list with None values to avoid repeated memory reallocation
        out = [None] * n_blocks
        block_index = 0

        for y in range(0, ysize, y_block_size):
            win_ysize = min(y_block_size, ysize - y)

            cell_areas = all_cell_areas[y : y + win_ysize].copy()
            cell_areas = cell_areas.reshape(-1, 1)

            for x in range(0, xsize, x_block_size):
                if self.is_killed():
                    logger.info(
                        "Processing killed by user after processing "
                        f"{n} out of {n_blocks} blocks."
                    )

                    break
                self.emit_progress(n * progress_increment)

                win_xsize = min(x_block_size, xsize - x)

                # Use cached bands and more efficient array reading
                if src_ds.RasterCount == 1:
                    src_array = src_bands[0].ReadAsArray(
                        xoff=x, yoff=y, win_xsize=win_xsize, win_ysize=win_ysize
                    )
                else:
                    src_array = src_ds.ReadAsArray(
                        xoff=x, yoff=y, xsize=win_xsize, ysize=win_ysize
                    )

                mask_array = band_mask.ReadAsArray(
                    xoff=x, yoff=y, win_xsize=win_xsize, win_ysize=win_ysize
                )
                mask_array = mask_array == config.MASK_VALUE

                result = self.processing_function(
                    self.params, src_array, mask_array, x, y, cell_areas
                )

                out[block_index] = result[0]
                block_index += 1

                write_arrays = result[1]
                for band_num, data in enumerate(write_arrays):
                    dst_bands[band_num].WriteArray(
                        data["array"], data["xoff"], data["yoff"]
                    )

                n += 1

            if self.is_killed():
                break

            lat += pixel_height * win_ysize

        # Clean up cached references
        del src_bands, dst_bands

        # pr.disable()
        # pr.dump_stats('calculate_ld_stats')

        if self.is_killed():
            del dst_ds
            os.remove(self.params.out_file)
            return None
        else:
            self.emit_progress(1)
            del dst_ds
            # Filter out None values if processing was interrupted
            out = [item for item in out if item is not None]
            return out
