import json
import logging
import pathlib
import shutil
import tempfile
from typing import List

import marshmallow_dataclass
from defusedxml.ElementTree import parse
from osgeo import gdal, ogr, osr

from .util_numba import _accumulate_dicts

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

gdal.UseExceptions()


def log_system_resources(
    context_message="System resource usage",
    error_logger=None,
    additional_paths=None,
    input_file=None,
    output_file=None,
    src_win=None,
):
    """
    Log comprehensive system resource information for debugging.

    Args:
        context_message (str): Context message to include in logs
        error_logger: Logger instance to use (defaults to module logger)
        additional_paths (list): Additional paths to check for disk usage
        input_file (str): Input file path to log information about
        output_file (str): Output file path to log information about
        src_win (tuple): Source window information (x_off, y_off, x_size, y_size)
    """
    if error_logger is None:
        error_logger = logger

    error_logger.error(f"{context_message}:")

    if psutil is None:
        error_logger.error(
            "psutil not available - cannot gather system resource information"
        )
        return

    try:
        from pathlib import Path

        # Get memory information
        memory = psutil.virtual_memory()
        error_logger.error("Memory usage:")
        error_logger.error(f"  - Total memory: {memory.total / (1024**3):.2f} GB")
        error_logger.error(
            f"  - Available memory: {memory.available / (1024**3):.2f} GB"
        )
        error_logger.error(
            f"  - Used memory: {memory.used / (1024**3):.2f} GB ({memory.percent:.1f}%)"
        )
        error_logger.error(f"  - Free memory: {memory.free / (1024**3):.2f} GB")

        # Get disk space information for standard paths
        standard_paths = [
            ("Working directory", "/work"),
            ("Tmp", "/tmp"),
            ("Downloads", "/downloads"),
        ]
        if additional_paths:
            standard_paths.extend(additional_paths)

        for path_name, path_str in standard_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    disk_usage = shutil.disk_usage(path)
                    total_gb = disk_usage.total / (1024**3)
                    free_gb = disk_usage.free / (1024**3)
                    used_gb = (disk_usage.total - disk_usage.free) / (1024**3)
                    used_percent = (used_gb / total_gb) * 100

                    error_logger.error(f"Disk usage for {path_name} ({path}):")
                    error_logger.error(f"  - Total: {total_gb:.2f} GB")
                    error_logger.error(
                        f"  - Used: {used_gb:.2f} GB ({used_percent:.1f}%)"
                    )
                    error_logger.error(f"  - Free: {free_gb:.2f} GB")
                except Exception as disk_err:
                    error_logger.error(
                        f"Could not get disk usage for {path_name} ({path}): {disk_err}"
                    )

        # Get CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        error_logger.error(f"CPU usage: {cpu_percent:.1f}% (cores: {cpu_count})")

        # Log file information if provided
        if input_file or output_file or src_win:
            error_logger.error("File/operation details:")

            if input_file:
                error_logger.error(f"  - Input file: {input_file}")
                try:
                    input_path = Path(input_file)
                    if input_path.exists():
                        file_size = input_path.stat().st_size / (1024**2)  # MB
                        error_logger.error(f"  - Input file size: {file_size:.2f} MB")
                    else:
                        error_logger.error("  - Input file does not exist")
                except Exception as file_err:
                    error_logger.error(f"  - Could not get input file info: {file_err}")

            if output_file:
                error_logger.error(f"  - Output file: {output_file}")

            if src_win:
                error_logger.error(
                    f"  - Source window (x_off, y_off, x_size, y_size): {src_win}"
                )

    except Exception as resource_err:
        error_logger.error(
            f"Could not gather system resource information: {resource_err}"
        )


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
                f"base file ({in_files[0]}) geotransform ({out_gt_rounded}) doesn't "
                f"match geotransform in {in_file} ({this_gt_rounded})"
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

            if file_num > 0:
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
    # Add more informative progress logging with reduced precision for cleaner output
    logger.info("%s - %.1f%%", message, 100 * fraction)
