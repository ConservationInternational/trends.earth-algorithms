from typing import List
import pathlib

import marshmallow_dataclass

from osgeo import gdal, osr

from te_schemas.jobs import JobBand, Path


@marshmallow_dataclass.dataclass
class DataFile:
    path: Path
    bands: List[JobBand]

    def array_rows_for_name(self, name_filter):
        names = [b.name for b in self.bands]

        return [
            index for index, name in enumerate(names)
            if name == name_filter
        ]

    def array_row_for_name(self, name_filter):
        '''throw an error if more than one result'''
        out = self.array_rows_for_name(name_filter)

        if len(out) > 1:
            raise RuntimeError(
                f'more than one band found for name {name_filter}'
            )
        else:
            return out[0]

    def append(self, datafile):
        '''
        Extends bands with those from another datafile

        This assumes that both DataFile share the same path (where the path
        is the one of the original DataFile)
        '''
        datafiles = [self, datafile]

        self.bands = [b for d in datafiles for b in d.bands]

    def extend(self, datafiles):
        '''
        Extends bands with those from another datafile

        This assumes that both DataFile share the same path (where the path
        is the one of the original DataFile)
        '''

        datafiles = [self] + datafiles

        self.bands = [b for d in datafiles for b in d.bands]


@marshmallow_dataclass.dataclass
class ImageInfo:
    x_size: int
    y_size: int
    x_block_size: int
    y_block_size: int
    pixel_width: float
    pixel_height: float

    def get_n_blocks(self):
        return (
            len([*range(0, self.x_size, self.x_block_size)]) *
            len([*range(0, self.y_size, self.y_block_size)])
        )


def get_image_info(path: pathlib.Path):
    ds = gdal.Open(str(path))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    block_sizes = band.GetBlockSize()

    return ImageInfo(
        band.XSize,
        band.YSize,
        block_sizes[0],
        block_sizes[1],
        gt[1],
        gt[5]
    )


def setup_output_image(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    n_bands: int,
    image_info: ImageInfo
):
    # Need two output bands for each four year period, plus one for the JRC 
    # layer

    # Setup output file for max drought and population counts
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        str(out_file),
        image_info.x_size,
        image_info.y_size,
        n_bands,
        gdal.GDT_Int16,
        options=['COMPRESS=LZW']
    )
    src_ds = gdal.Open(str(in_file))
    src_gt = src_ds.GetGeoTransform()
    dst_ds.SetGeoTransform(src_gt)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(src_ds.GetProjectionRef())
    dst_ds.SetProjection(dst_srs.ExportToWkt())
    return dst_ds
