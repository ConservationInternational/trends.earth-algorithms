import logging
import os
from typing import Union

import numpy as np
from osgeo import gdal
from osgeo import osr

from . import config
from . import models
from .. import util
from ..util_numba import calc_cell_area

logger = logging.getLogger(__name__)


class DegradationSummary:
    def __init__(
        self,
        params: Union[
            models.DegradationSummaryParams,
            models.DegradationProgressSummaryParams,
            models.DegradationErrorRecodeSummaryParams,
        ],
        processing_function,
    ):
        self.params = params
        self.processing_function = processing_function

    def is_killed(self):
        return False

    def emit_progress(self, *args, **kwargs):
        """Reimplement to display progress messages"""
        util.log_progress(*args, message="Processing land degradation summary")

    def work(self):
        mask_ds = gdal.Open(self.params.mask_file)
        band_mask = mask_ds.GetRasterBand(1)

        src_ds = gdal.Open(str(self.params.in_file))

        model_band = src_ds.GetRasterBand(self.params.model_band_number)
        block_sizes = model_band.GetBlockSize()
        xsize = model_band.XSize
        ysize = model_band.YSize

        x_block_size = block_sizes[0]
        y_block_size = block_sizes[1]

        # Setup output file for SDG degradation indicator and combined
        # productivity bands
        driver = gdal.GetDriverByName("GTiff")
        dst_ds_deg = driver.Create(
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
        dst_ds_deg.SetGeoTransform(src_gt)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(src_ds.GetProjectionRef())
        dst_ds_deg.SetProjection(dst_srs.ExportToWkt())

        # Width of cells in longitude
        long_width = src_gt[1]
        # Set initial lat ot the top left corner latitude
        lat = src_gt[3]
        # Width of cells in latitude
        pixel_height = src_gt[5]

        n_blocks = len(np.arange(0, xsize, x_block_size)) * len(
            np.arange(0, ysize, y_block_size)
        )

        # pr = cProfile.Profile()
        # pr.enable()
        n = 0
        out = []

        for y in range(0, ysize, y_block_size):
            if y + y_block_size < ysize:
                win_ysize = y_block_size
            else:
                win_ysize = ysize - y

            cell_areas = (
                np.array(
                    [
                        calc_cell_area(
                            lat + pixel_height * n,
                            lat + pixel_height * (n + 1),
                            long_width,
                        )
                        for n in range(win_ysize)
                    ]
                )
                * 1e-6
            )  # 1e-6 is to convert from meters to kilometers
            cell_areas.shape = (cell_areas.size, 1)

            for x in range(0, xsize, x_block_size):
                if self.is_killed():
                    logger.info(
                        "Processing killed by user after processing "
                        f"{n} out of {n_blocks} blocks."
                    )

                    break
                self.emit_progress(n / n_blocks)

                if x + x_block_size < xsize:
                    win_xsize = x_block_size
                else:
                    win_xsize = xsize - x

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

                out.append(result[0])

                for band_num, data in enumerate(result[1], start=1):
                    dst_ds_deg.GetRasterBand(band_num).WriteArray(**data)

                n += 1

            if self.is_killed():
                break

            lat += pixel_height * win_ysize

        # pr.disable()
        # pr.dump_stats('calculate_ld_stats')

        if self.is_killed():
            del dst_ds_deg
            os.remove(self.params.out_file)

            return None
        else:
            self.emit_progress(1)

            return out
