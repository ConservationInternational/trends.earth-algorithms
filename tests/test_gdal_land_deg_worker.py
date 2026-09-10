from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "marshmallow_dataclass", reason="marshmallow-dataclass not available"
)
pytest.importorskip("osgeo.gdal", reason="GDAL not available")
pytest.importorskip("osgeo.osr", reason="OSR not available")

from osgeo import gdal, osr

from te_algorithms.gdal.land_deg import worker


def _write_raster(path, values):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 2, 2, 1, gdal.GDT_Int16)
    dataset.SetGeoTransform((0, 1, 0, 2, 0, -1))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None


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
