from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "marshmallow_dataclass", reason="marshmallow-dataclass not available"
)
pytest.importorskip("osgeo.gdal", reason="GDAL not available")
pytest.importorskip("osgeo.osr", reason="OSR not available")

from osgeo import gdal, osr

from te_algorithms.gdal import workers
from te_algorithms.gdal.land_deg import worker
from te_algorithms.gdal.land_deg.land_deg import _population_band_to_counts


def _write_raster(path, values):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 2, 2, 1, gdal.GDT_Int16)
    dataset.SetGeoTransform((0, 1, 0, 2, 0, -1))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None


def test_population_band_to_counts_masks_non_finite_values():
    population = np.array([[1.0, np.nan], [np.inf, -32768.0]])

    counts, masked_counts = _population_band_to_counts(
        population,
        np.ones((2, 1), dtype=np.float64),
        {"units": "people per hectare"},
    )

    assert counts[0, 0] == 100.0
    assert counts[0, 1] == -32768.0
    assert counts[1, 0] == -32768.0
    assert counts[1, 1] == -32768.0
    np.testing.assert_array_equal(masked_counts, [[100.0, 0.0], [0.0, 0.0]])


def test_degradation_summary_writes_population_as_float32(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    integer_output_path = tmp_path / "summary.tif"
    population_output_path = tmp_path / "summary_population.tif"
    _write_raster(source_path, np.zeros((2, 2), dtype=np.int16))
    _write_raster(mask_path, np.zeros((2, 2), dtype=np.int16))

    params = SimpleNamespace(
        in_file=source_path,
        mask_file=mask_path,
        model_band_number=1,
        out_file=integer_output_path,
        population_out_file=population_output_path,
        n_out_bands=3,
        n_population_out_bands=1,
    )

    def process_block(_params, _source, _mask, xoff, yoff, _cell_areas):
        return (
            {0: 1.0},
            [
                {
                    "array": np.array([[-1, 0], [1, 0]], dtype=np.int16),
                    "xoff": xoff,
                    "yoff": yoff,
                },
                {
                    "array": np.array([[40000.5, 0.25], [1.5, 2.75]], dtype=np.float64),
                    "xoff": xoff,
                    "yoff": yoff,
                },
                {
                    "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
                    "xoff": xoff,
                    "yoff": yoff,
                },
            ],
        )

    worker.DegradationSummary(params, process_block).work()

    integer_dataset = gdal.Open(str(integer_output_path))
    population_dataset = gdal.Open(str(population_output_path))
    assert integer_dataset.RasterCount == 2
    assert integer_dataset.GetRasterBand(1).DataType == gdal.GDT_Int16
    assert integer_dataset.GetRasterBand(2).DataType == gdal.GDT_Int16
    assert population_dataset.RasterCount == 1
    assert population_dataset.GetRasterBand(1).DataType == gdal.GDT_Float32
    assert population_dataset.GetRasterBand(1).ReadAsArray()[0, 0] == pytest.approx(
        40000.5
    )


def test_degradation_summary_supports_non_population_parameters(tmp_path):
    source_path = tmp_path / "source.tif"
    mask_path = tmp_path / "mask.tif"
    output_path = tmp_path / "summary.tif"
    _write_raster(source_path, np.zeros((2, 2), dtype=np.int16))
    _write_raster(mask_path, np.zeros((2, 2), dtype=np.int16))

    params = SimpleNamespace(
        in_file=source_path,
        mask_file=mask_path,
        model_band_number=1,
        out_file=output_path,
        n_out_bands=1,
    )

    def process_block(_params, _source, _mask, xoff, yoff, _cell_areas):
        return (
            {0: 1.0},
            [
                {
                    "array": np.array([[-1, 0], [1, 0]], dtype=np.int16),
                    "xoff": xoff,
                    "yoff": yoff,
                }
            ],
        )

    worker.DegradationSummary(params, process_block).work()

    dataset = gdal.Open(str(output_path))
    assert dataset.RasterCount == 1
    assert dataset.GetRasterBand(1).DataType == gdal.GDT_Int16


def test_degradation_summary_materializes_mixed_type_source_bands(tmp_path):
    integer_path = tmp_path / "integer.tif"
    float_path = tmp_path / "float.tif"
    source_vrt = tmp_path / "source.vrt"
    mask_path = tmp_path / "mask.tif"
    materialized_vrt = tmp_path / "materialized.vrt"
    _write_raster(integer_path, np.array([[1, 2], [3, 4]], dtype=np.int16))
    float_dataset = gdal.GetDriverByName("GTiff").Create(
        str(float_path), 2, 2, 1, gdal.GDT_Float32
    )
    float_dataset.SetGeoTransform((0, 1, 0, 2, 0, -1))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
    float_dataset.SetProjection(spatial_reference.ExportToWkt())
    float_dataset.GetRasterBand(1).WriteArray(
        np.array([[1.25, 2.5], [3.75, 4.5]], dtype=np.float32)
    )
    float_dataset = None
    gdal.BuildVRT(str(source_vrt), [str(integer_path), str(float_path)], separate=True)
    _write_raster(mask_path, np.zeros((2, 2), dtype=np.int16))
    tile_path = workers.CutTiles(
        source_vrt,
        n_cpus=1,
        out_file=tmp_path / "tile.tif",
        output_format="VRT",
    ).work()[0]

    tile_dataset = gdal.Open(str(tile_path))
    assert tile_dataset.GetRasterBand(1).DataType == gdal.GDT_Int16
    assert tile_dataset.GetRasterBand(2).DataType == gdal.GDT_Float32

    params = SimpleNamespace(
        in_file=tile_path,
        mask_file=mask_path,
        model_band_number=1,
        out_file=tmp_path / "summary.tif",
        n_out_bands=1,
    )

    def process_block(_params, source, _mask, xoff, yoff, _cell_areas):
        return (
            {0: 1.0},
            [
                {
                    "array": np.zeros((2, 2), dtype=np.int16),
                    "xoff": xoff,
                    "yoff": yoff,
                }
            ],
        )

    worker.DegradationSummary(
        params,
        process_block,
        materialized_path=materialized_vrt,
    ).work()

    dataset = gdal.Open(str(materialized_vrt))
    assert dataset.RasterCount == 2
    assert dataset.GetRasterBand(1).DataType == gdal.GDT_Int16
    assert dataset.GetRasterBand(2).DataType == gdal.GDT_Float32
    np.testing.assert_array_equal(
        dataset.GetRasterBand(1).ReadAsArray(), [[1, 2], [3, 4]]
    )
    np.testing.assert_array_equal(
        dataset.GetRasterBand(2).ReadAsArray(), [[1.25, 2.5], [3.75, 4.5]]
    )
