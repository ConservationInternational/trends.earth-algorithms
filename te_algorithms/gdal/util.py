import json
import logging
import pathlib
import shutil
import tempfile
from typing import List

import marshmallow_dataclass
from defusedxml.ElementTree import parse
from osgeo import gdal, ogr, osr
from te_schemas.datafile import DataFile

from .util_numba import _accumulate_dicts

logger = logging.getLogger(__name__)

gdal.UseExceptions()


@marshmallow_dataclass.dataclass
class ImageInfo:
    x_size: int
    y_size: int
    x_block_size: int
    y_block_size: int
    pixel_width: float
    pixel_height: float

    def get_n_blocks(self):
        return len([*range(0, self.x_size, self.x_block_size)]) * len(
            [*range(0, self.y_size, self.y_block_size)]
        )


def get_image_info(path: pathlib.Path):
    ds = gdal.Open(str(path))
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    block_sizes = band.GetBlockSize()

    return ImageInfo(
        band.XSize, band.YSize, block_sizes[0], block_sizes[1], gt[1], gt[5]
    )


def setup_output_image(
    in_file: pathlib.Path, out_file: pathlib.Path, n_bands: int, image_info: ImageInfo
):
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        str(out_file),
        image_info.x_size,
        image_info.y_size,
        n_bands,
        gdal.GDT_Int16,
        options=["COMPRESS=LZW"],
    )
    src_ds = gdal.Open(str(in_file))
    src_gt = src_ds.GetGeoTransform()
    dst_ds.SetGeoTransform(src_gt)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(src_ds.GetProjectionRef())
    dst_ds.SetProjection(dst_srs.ExportToWkt())

    return dst_ds


def get_sourcefiles_in_vrt(vrt):
    vrt_tree = parse(vrt)
    vrt_root = vrt_tree.getroot()
    filenames = []

    for band in vrt_root.findall("VRTRasterBand"):
        sources = band.findall("SimpleSource")

        for source in sources:
            filenames.append(source.find("SourceFilename").text)

    return list(set(filenames))


def combine_all_bands_into_vrt(
    in_files: List[pathlib.Path],
    out_file: pathlib.Path,
    is_relative=True,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    """creates a vrt file combining all bands of in_files

    All bands must have the same extent, resolution, and crs
    """

    logger.debug("Making %s", out_file)

    logger.debug("In files: %s", in_files)

    if aws_access_key_id is not None:
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", aws_access_key_id)

    if aws_secret_access_key is not None:
        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)

    simple_source_raw = """    <SimpleSource>
        <SourceFilename relativeToVRT="{is_relative}">{source_path}</SourceFilename>
        <SourceBand>{source_band_num}</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="{out_Xsize}" ySize="{out_Ysize}"/>
        <DstRect xOff="0" yOff="0" xSize="{out_Xsize}" ySize="{out_Ysize}"/>
    </SimpleSource>"""

    gdal.VSICurlClearCache()

    for file_num, in_file in enumerate(in_files):
        logger.debug("Adding %s (file number %s)", in_file, file_num)
        in_ds = gdal.Open(str(in_file))
        this_gt = in_ds.GetGeoTransform()
        logger.debug("this_gt %s", this_gt)
        this_proj = in_ds.GetProjectionRef()

        if file_num == 0:
            out_gt = this_gt
            out_proj = this_proj
            logger.debug("out_gt %s", out_gt)
        else:
            out_gt_rounded = [round(x, 6) for x in out_gt]
            this_gt_rounded = [round(x, 6) for x in this_gt]
            assert out_gt_rounded == this_gt_rounded, (
                f"base file ({in_files[0]}) geotransform ({out_gt_rounded}) doesn't match "
                f"geotransform in {in_file} ({this_gt_rounded})"
            )
            assert out_proj == this_proj

        for band_num in range(1, in_ds.RasterCount + 1):
            this_dt = in_ds.GetRasterBand(band_num).DataType
            this_band = in_ds.GetRasterBand(band_num)
            this_Xsize = this_band.XSize
            this_Ysize = this_band.YSize

            if band_num == 1:
                out_Xsize = this_Xsize
                out_Ysize = this_Ysize

                if file_num == 0:
                    # If this is the first band of the first file, need to
                    # create the output VRT file
                    driver = gdal.GetDriverByName("VRT")
                    out_ds = driver.Create(
                        str(out_file),
                        out_Xsize,
                        out_Ysize,
                        0,
                    )
                    out_ds.SetGeoTransform(out_gt)
                    out_srs = osr.SpatialReference()
                    out_srs.ImportFromWkt(out_proj)
                    out_ds.SetProjection(out_srs.ExportToWkt())

            if file_num > 1:
                assert this_Xsize == out_Xsize
                assert this_Ysize == out_Ysize

            out_ds.AddBand(this_dt)
            # The new band will always be last band in out_ds
            band = out_ds.GetRasterBand(out_ds.RasterCount)

            md = {}
            md["source_0"] = simple_source_raw.format(
                is_relative=1 if is_relative else 0,
                source_path=in_file,
                source_band_num=band_num,
                out_Xsize=out_Xsize,
                out_Ysize=out_Ysize,
            )
            band.SetMetadata(md, "vrt_sources")

        in_ds = None
    out_ds = None

    # Use a regex to remove the parent elements from the paths for each band
    # (have to include them when setting metadata or else GDAL throws an error)
    fh, new_file = tempfile.mkstemp()
    new_file = pathlib.Path(new_file)
    with new_file.open("w", encoding="utf-8") as fh_new:
        with out_file.open() as fh_old:
            for line in fh_old:
                fh_new.write(line.replace(str(out_file.parents[0]) + "/", ""))
    out_file.unlink()
    shutil.copy(str(new_file), str(out_file))

    return True


def save_vrt(source_path: pathlib.Path, source_band_index: int) -> str:
    temporary_file = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False)
    temporary_file.close()
    gdal.BuildVRT(temporary_file.name, str(source_path), bandList=[source_band_index])

    return temporary_file.name


def save_vrt2(source_path: pathlib.Path, source_band_indices: List[int]) -> str:
    """Supports saving multiple bands"""
    temporary_file = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False)
    temporary_file.close()
    gdal.BuildVRT(temporary_file.name, str(source_path), bandList=source_band_indices)

    return temporary_file.name


def rm_duplicates(self) -> DataFile:
    """
    Remove duplicate bands from the DataFile and return a new Datafile
    """

    # TODO: Need to add tests for this function

    duplicates = []
    for outer_index in range(len(self.bands) - 1):
        if outer_index in duplicates:
            continue
        outer_band = self.bands[outer_index]
        for inner_index in range(outer_index + 1, len(self.bands)):
            if inner_index in duplicates:
                continue
            inner_band = self.bands[inner_index]

            if all(
                [
                    outer_band.name == inner_band.name,
                    outer_band.metadata == inner_band.metadata,
                ]
            ):
                duplicates.append(inner_index)

    bands = [band.copy() for band, i in enumerate(self.bands) if i not in duplicates]
    band_indices = [i for i in range(len(self.bands)) if i not in duplicates]
    return DataFile(path=save_vrt2(self.path, band_indices), bands=bands)


def wkt_geom_to_geojson_file_string(wkt):
    out_file = tempfile.NamedTemporaryFile(suffix=".geojson").name
    out_ds = ogr.GetDriverByName("GeoJSON").CreateDataSource(out_file)
    out_layer = out_ds.CreateLayer("wkt_geom", geom_type=ogr.wkbPolygon)
    feature_def = out_layer.GetLayerDefn()
    out_feat = ogr.Feature(feature_def)

    out_feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
    out_layer.CreateFeature(out_feat)
    out_layer = None
    out_ds = None

    with open(out_file) as f:
        return json.load(f)


def accumulate_dicts(z):
    # allow to handle items that may be None (comes up with
    # lc_trans_prod_bizonal for example, when initial year of cover is not
    # available to match with productivity data)
    z = [item for item in z if (item is not None and item != {})]

    if len(z) == 0:
        return {}
    elif len(z) == 1:
        return z[0]
    else:
        return _accumulate_dicts(z)


def log_progress(fraction, message=None, data=None):
    logger.info("%s - %.2f%%", message, 100 * fraction)
