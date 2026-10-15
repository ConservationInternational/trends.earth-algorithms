from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from osgeo import gdal, osr
from te_schemas.results import Band

from te_algorithms.gdal.land_deg.land_deg import (
    _population_band_to_counts,
    _prepare_population_dfs,
    _process_single_period,
)
from te_algorithms.gdal.population import (
    COUNTS,
    DENSITY_HA,
    DENSITY_KM2,
    _cell_areas_hectares,
    normalize_population_raster,
)
from te_algorithms.gdal.util_numba import calc_cell_area


@pytest.mark.parametrize("latitude", [-70, -34, 0, 34, 70])
@pytest.mark.parametrize("pixel_width,pixel_height", [(0.003, -0.003), (-0.003, 0.003)])
def test_native_cell_areas_match_summary_formula(
    tmp_path, latitude, pixel_width, pixel_height
):
    source = tmp_path / "source.tif"
    transform = (73, pixel_width, 0, latitude, 0, pixel_height)
    _write_population(source, np.ones((5, 6)), transform=transform)
    actual = _cell_areas_hectares(transform, 1, 2)
    expected = np.array(
        [
            abs(
                calc_cell_area(
                    latitude + row * pixel_height,
                    latitude + (row + 1) * pixel_height,
                    pixel_width,
                )
            )
            / 10000
            for row in (1, 2)
        ]
    )[:, None]
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "epsg,transform",
    [
        (3857, (0, 300, 0, 300, 0, -300)),
        (4326, (73, 0.003, 0.001, 34, 0, -0.003)),
        (4326, (73, 0.003, 0, 34, 0.001, -0.003)),
    ],
)
def test_count_normalization_rejects_unsupported_grids(tmp_path, epsg, transform):
    source = tmp_path / "projected.tif"
    output = tmp_path / "out.tif"
    _write_population(source, [[500.25]], epsg=epsg, transform=transform)
    with pytest.raises(ValueError, match="unrotated WGS84"):
        normalize_population_raster(source, output, COUNTS)
    assert not output.exists()
    assert not list(tmp_path.glob("te_density_*"))
    normalize_population_raster(source, output, DENSITY_HA)
    result = gdal.Open(str(output))
    assert result.ReadAsArray()[0, 0] == pytest.approx(500.25)
    assert result.GetGeoTransform() == transform


def _write_population(path, values, epsg=4326, transform=None, nodata=-9999):
    values = np.asarray(values, dtype=np.float64)
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Float64
    )
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(epsg)
    dataset.SetProjection(projection.ExportToWkt())
    dataset.SetGeoTransform(transform or (73, 0.003, 0, 34, 0, -0.003))
    dataset.GetRasterBand(1).SetNoDataValue(nodata)
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None


def test_native_counts_resampled_to_ten_times_finer_grid_preserve_total(tmp_path):
    source = tmp_path / "counts.tif"
    _write_population(source, [[500.25]])
    metadata = {"year": 2020, "type": "total"}
    params = {
        "layer_population_bands": [
            Band("Population (number of people)", metadata=metadata).__dict__
        ],
        "layer_population_band_indexes": [1],
        "layer_population_paths": [str(source)],
    }
    prepared = _prepare_population_dfs(params, tmp_path)
    assert prepared[0].bands[0].metadata["units"] == DENSITY_HA
    assert "units" not in metadata
    vrt = gdal.BuildVRT(
        str(tmp_path / "resampled.vrt"),
        [prepared[0].path],
        xRes=0.0003,
        yRes=0.0003,
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    values = vrt.ReadAsArray()
    assert values.shape == (10, 10)
    gt = vrt.GetGeoTransform()
    areas = np.array(
        [
            calc_cell_area(gt[3] + row * gt[5], gt[3] + (row + 1) * gt[5], gt[1]) / 1e6
            for row in range(10)
        ]
    )[:, None]
    counts, _ = _population_band_to_counts(values, areas, prepared[0].bands[0].metadata)
    assert counts.sum() == pytest.approx(500.25, rel=1e-6)
    # The old path duplicates each source count into all 100 output pixels.
    old = gdal.BuildVRT(
        str(tmp_path / "old.vrt"), [str(source)], xRes=0.0003, yRes=0.0003
    )
    assert old.ReadAsArray().sum() == pytest.approx(50025)


@pytest.mark.parametrize(
    "units,value,expected",
    [
        (DENSITY_HA, 1.25, 1.25),
        (DENSITY_KM2, 125, 1.25),
    ],
)
def test_density_units_and_nodata_are_normalized(tmp_path, units, value, expected):
    source = tmp_path / "source.tif"
    output = tmp_path / "density.tif"
    _write_population(source, [[value, -9999, np.nan, np.inf, 0]])
    normalize_population_raster(source, output, units)
    dataset = gdal.Open(str(output))
    band = dataset.GetRasterBand(1)
    assert band.DataType == gdal.GDT_Float32
    assert band.GetNoDataValue() == -32768
    np.testing.assert_array_equal(
        band.ReadAsArray(), [[expected, -32768, -32768, -32768, 0]]
    )


@pytest.mark.parametrize("units", [COUNTS, DENSITY_HA, DENSITY_KM2])
def test_normalization_rejects_negative_population(tmp_path, units):
    source = tmp_path / "negative.tif"
    _write_population(source, [[-5]])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_population_raster(source, tmp_path / "out.tif", units)
    assert not (tmp_path / "out.tif").exists()
    assert not list(tmp_path.glob("te_density_*"))


def test_unknown_units_are_not_silently_treated_as_counts(tmp_path):
    with pytest.raises(ValueError, match="Unsupported population units"):
        normalize_population_raster("unused", tmp_path / "out.tif", "unknown")


def test_normalization_reuses_row_areas_across_horizontal_blocks(tmp_path):
    source = tmp_path / "wide.tif"
    output = tmp_path / "density.tif"
    _write_population(source, np.full((300, 600), 500.25))
    with patch(
        "te_algorithms.gdal.population._cell_areas_hectares",
        wraps=_cell_areas_hectares,
    ) as calculate:
        normalize_population_raster(source, output, COUNTS)
    assert calculate.call_count == 2
    result = gdal.Open(str(output))
    values = result.ReadAsArray()
    np.testing.assert_array_equal(values[:, 0], values[:, -1])


def test_normalization_preserves_mask_scale_and_offset(tmp_path):
    source = tmp_path / "scaled.tif"
    output = tmp_path / "density.tif"
    _write_population(source, [[1, 2, -9999]])
    dataset = gdal.Open(str(source), gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    band.SetScale(2)
    band.SetOffset(0.5)
    band.CreateMaskBand(gdal.GMF_PER_DATASET)
    band.GetMaskBand().WriteArray(np.array([[255, 0, 255]], dtype=np.uint8))
    dataset = None
    normalize_population_raster(source, output, DENSITY_HA)
    result = gdal.Open(str(output))
    np.testing.assert_array_equal(result.ReadAsArray(), [[2.5, -32768, -32768]])


def test_period_population_temporary_directory_is_removed_on_failure():
    directories = []

    def fail(_params, directory):
        assert directory.is_dir()
        directories.append(directory)
        (directory / "population_0.tif").touch()
        raise RuntimeError("Cancelled by user.")

    with (
        patch(
            "te_algorithms.gdal.land_deg.land_deg._prepare_land_cover_dfs",
            return_value=[],
        ),
        patch(
            "te_algorithms.gdal.land_deg.land_deg._prepare_soil_organic_carbon_dfs",
            return_value=[],
        ),
        patch(
            "te_algorithms.gdal.land_deg.land_deg._prepare_population_dfs",
            side_effect=fail,
        ),
        pytest.raises(RuntimeError, match="Cancelled"),
    ):
        _process_single_period(
            {"name": "baseline", "params": {}}, None, None, Path("output.json"), 1
        )
    assert len(directories) == 1
    assert not directories[0].exists()
