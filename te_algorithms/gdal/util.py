import pathlib
import json
import tempfile
from typing import List

import marshmallow_dataclass

from osgeo import gdal, osr, ogr


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


def combine_all_bands_into_vrt(
    in_files: List[pathlib.Path],
    out_file: pathlib.Path
):
    '''creates a vrt file combining all bands of in_files

    All bands must have the same extent, resolution, and crs
    '''

    simple_source_raw = '''    <SimpleSource>
        <SourceFilename relativeToVRT="1">{relative_path}</SourceFilename>
        <SourceBand>{source_band_num}</SourceBand>
        <SrcRect xOff="0" yOff="0" xSize="{out_Xsize}" ySize="{out_Ysize}"/>
        <DstRect xOff="0" yOff="0" xSize="{out_Xsize}" ySize="{out_Ysize}"/>
    </SimpleSource>'''
    
    for file_num, in_file in enumerate(in_files):
        in_ds = gdal.Open(str(in_file))
        this_gt = in_ds.GetGeoTransform()
        this_proj = in_ds.GetProjectionRef()
        if file_num == 0:
            out_gt = this_gt
            out_proj = this_proj
        else:
            assert [round(x, 8) for x in out_gt] == [round(x, 8) for x in this_gt]
            assert out_proj == this_proj

        for band_num in range(1, in_ds.RasterCount + 1):
            this_dt = in_ds.GetRasterBand(band_num).DataType
            this_band = in_ds.GetRasterBand(band_num)
            this_Xsize = this_band.XSize
            this_Ysize = this_band.YSize
            if band_num == 1:
                out_dt = this_dt
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
                        out_dt
                    )
                    out_ds.SetGeoTransform(out_gt)
                    out_srs = osr.SpatialReference()
                    out_srs.ImportFromWkt(out_proj)
                    out_ds.SetProjection(out_srs.ExportToWkt())
            if file_num > 1:
                assert this_dt == out_dt
                assert this_Xsize == out_Xsize
                assert this_Ysize == out_Ysize

            out_ds.AddBand(out_dt)
            # The new band will always be last band in out_ds
            band = out_ds.GetRasterBand(out_ds.RasterCount)

            md = {}
            md['source_0'] = simple_source_raw.format(
                relative_path=in_file,
                source_band_num=band_num,
                out_Xsize=out_Xsize,
                out_Ysize=out_Ysize
            )
            band.SetMetadata(md, 'vrt_sources')

    out_ds = None

    # Use a regex to remove the parent elements from the paths for each band 
    # (have to include them when setting metadata or else GDAL throws an error)
    fh, new_file = tempfile.mkstemp()
    new_file = pathlib.Path(new_file)
    with new_file.open('w') as fh_new:
        with out_file.open() as fh_old:
            for line in fh_old:
                fh_new.write(
                    line.replace(str(out_file.parents[0]) + '/', '')
                )
    out_file.unlink()
    shutil.copy(str(new_file), str(out_file))

    return True


def save_vrt(source_path: pathlib.Path, source_band_index: int) -> str:
    temporary_file = tempfile.NamedTemporaryFile(suffix=".vrt", delete=False)
    temporary_file.close()
    gdal.BuildVRT(
        temporary_file.name,
        str(source_path),
        bandList=[source_band_index]
    )
    return temporary_file.name


def wkt_geom_to_geojson_file_string(wkt):
    out_file = tempfile.NamedTemporaryFile(suffix='.geojson').name
    out_ds = ogr.GetDriverByName('GeoJSON').CreateDataSource(out_file)
    out_layer = out_ds.CreateLayer('wkt_geom', geom_type=ogr.wkbPolygon)
    feature_def = out_layer.GetLayerDefn()
    out_feat = ogr.Feature(feature_def)

    out_feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
    out_layer.CreateFeature(out_feat)
    out_layer = None
    out_ds = None

    with open(out_file, 'r') as f:
        return json.load(f)
