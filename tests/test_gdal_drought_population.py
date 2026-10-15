import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal")
pytest.importorskip("marshmallow_dataclass")

from osgeo import gdal, osr
from te_schemas.datafile import DataFile
from te_schemas.results import Band, DataType, RasterFileType, RasterResults, RasterType

from te_algorithms.gdal import drought as dr


def _inputs(by_sex, n_years=1, shape=(2, 4)):
    bands = [
        Band(name=dr.SPI_BAND_NAME, metadata={"year": 2000 + i}) for i in range(n_years)
    ]
    arrays = [np.full(shape, -1500.0) for _ in range(n_years)]
    for pop_type in ("male", "female") if by_sex else ("total",):
        for i in range(n_years):
            bands.append(
                Band(
                    name=dr.POPULATION_BAND_NAME,
                    metadata={"year": 2000 + i, "type": pop_type},
                )
            )
            arrays.append(
                np.resize(
                    [0.25, 40000.5, dr.NODATA_VALUE, np.nan, np.inf, 2.75, 0, 1.5],
                    shape,
                )
            )
    bands.extend(
        [
            Band(name=dr.JRC_BAND_NAME, metadata={}),
            Band(name=dr.WATER_MASK_BAND_NAME, metadata={}),
        ]
    )
    arrays.extend([np.zeros(shape), np.zeros(shape)])
    for spi in arrays[:n_years]:
        spi[-1, -1] = dr.NODATA_VALUE
    arrays[-1][-1, min(1, shape[1] - 1)] = 1
    return bands, np.array(arrays, dtype=np.float32)


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize("n_years", [1, 5])
@pytest.mark.parametrize("shape", [(2, 4), (1, 8), (8, 1), (1, 1)])
def test_population_at_max_drought_preserves_fractions_and_nodata(
    by_sex, n_years, shape
):
    bands, arrays = _inputs(by_sex, n_years, shape)
    original = arrays.copy()
    params = dr.DroughtSummaryParams(
        DataFile("in.vrt", bands), "out.tif", 4, "mask.tif"
    )
    summary, writes = dr._process_block(
        params, arrays, np.zeros(shape, dtype=bool), 0, 0, np.ones((shape[0], 1))
    )
    expected = original[n_years].astype(np.float64)
    invalid = (
        ~np.isfinite(expected)
        | (expected == dr.NODATA_VALUE)
        | (original[0] == dr.NODATA_VALUE)
        | (original[-1] == 1)
    )
    expected = np.where(invalid, dr.NODATA_VALUE, -expected)
    for period in range(len(range(0, n_years, 4))):
        total_expected = expected.copy()
        if by_sex:
            total_expected[~invalid] *= 2
        np.testing.assert_array_equal(writes[2 * period + 2]["array"], total_expected)
    if by_sex:
        for band_index in (len(writes) - 1, len(writes)):
            np.testing.assert_array_equal(writes[band_index]["array"], expected)
    np.testing.assert_array_equal(arrays, original)
    valid = np.isfinite(original[n_years]) & (original[n_years] != dr.NODATA_VALUE)
    valid &= original[-1] != 1
    assert summary.annual_population_by_drought_class_total[0].get(
        2, 0.0
    ) == pytest.approx(
        original[n_years][valid & (original[0] == -1500)].sum() * (2 if by_sex else 1)
    )


def _write_raster(path, arrays, datatype=gdal.GDT_Float32):
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), arrays.shape[2], arrays.shape[1], len(arrays), datatype
    )
    dataset.SetGeoTransform((0, 0.001, 0, 2, 0, -0.001))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    for i, array in enumerate(arrays, 1):
        dataset.GetRasterBand(i).WriteArray(array)
        dataset.GetRasterBand(i).SetNoDataValue(dr.NODATA_VALUE)
    dataset = None


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize("n_years", [1, 5, 9])
def test_drought_summary_splits_rasters_and_metadata(
    tmp_path, monkeypatch, by_sex, n_years
):
    bands, arrays = _inputs(by_sex, n_years)
    source = tmp_path / "source.tif"
    mask = tmp_path / "mask.tif"
    output = tmp_path / "summary.tif"
    _write_raster(source, arrays)
    _write_raster(mask, np.zeros((1, *arrays.shape[1:]), dtype=np.float32))
    params = dr.DroughtSummaryParams(DataFile(source, bands), str(output), 4, str(mask))
    worker = dr.DroughtSummary(params)
    summary = worker.process_lines(worker.get_line_params())

    n_periods = len(range(0, n_years, 4))
    output_tiles = {
        DataType.INT16: [output],
        DataType.FLOAT32: [dr._get_population_output_path(output)],
    }
    for datatype, paths in output_tiles.items():
        dataset = gdal.Open(str(paths[0]))
        assert dataset.RasterCount == (
            n_periods
            if datatype == DataType.INT16
            else n_periods + (2 if by_sex else 0)
        )
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            assert band.DataType == (
                gdal.GDT_Int16 if datatype == DataType.INT16 else gdal.GDT_Float32
            )
            assert band.GetNoDataValue() == dr.NODATA_VALUE
            if datatype == DataType.FLOAT32:
                population = band.ReadAsArray()
                assert population[0, 0] == (
                    -0.5 if by_sex and i <= n_periods else -0.25
                )
                assert population[0, 1] == (
                    -80001 if by_sex and i <= n_periods else -40000.5
                )
                np.testing.assert_array_equal(population[0, 2:], dr.NODATA_VALUE)
            else:
                np.testing.assert_array_equal(band.ReadAsArray(), arrays[0])
        dataset = None

    monkeypatch.setattr(dr, "_prepare_dfs", lambda *_: [DataFile(source, bands)])
    monkeypatch.setattr(
        dr, "_prepare_population_dfs", lambda *_: [DataFile(source, bands)]
    )
    monkeypatch.setattr(
        dr, "_compute_drought_summary_table", lambda **_: (summary, output_tiles)
    )
    monkeypatch.setattr(dr, "save_reporting_json", lambda *_: {})
    monkeypatch.setattr(dr, "save_summary_table_excel", lambda *_, **__: None)
    job = SimpleNamespace(
        task_name="test",
        params={
            "layer_spi_path": str(source),
            "layer_spi_bands": [Band.Schema().dump(bands[0])],
            "layer_spi_band_indices": [1],
            "layer_spi_years": list(range(2000, 2000 + n_years)),
            "layer_spi_lag": 12,
            "layer_population_path": str(source),
            "layer_population_bands": [],
            "layer_population_band_indices": [],
            "layer_jrc_path": str(source),
            "layer_jrc_band": {},
            "layer_jrc_band_index": 1,
        },
    )
    results = dr.summarise_drought_vulnerability(
        job, None, tmp_path / "job.json", n_cpus=1
    )
    assert list(results.rasters) == [DataType.INT16.value, DataType.FLOAT32.value]
    assert results.uri.uri.suffix == ".vrt"
    for datatype in output_tiles:
        raster = results.rasters[datatype.value]
        assert raster.datatype == datatype
        assert raster.type == RasterType.ONE_FILE_RASTER
        assert raster.filetype == RasterFileType.GEOTIFF
        assert raster.uri.uri == output_tiles[datatype][0]
    combined = gdal.Open(str(results.uri.uri))
    assert combined.RasterCount == len(results.get_bands())
    for i, band in enumerate(results.get_bands(), start=1):
        assert combined.GetRasterBand(i).DataType == (
            gdal.GDT_Float32
            if band.name == dr.POP_AT_MAX_DROUGHT_BAND_NAME
            else gdal.GDT_Int16
        )
        assert combined.GetRasterBand(i).GetNoDataValue() == dr.NODATA_VALUE
        assert combined.GetRasterBand(i).GetDescription()
    combined = None
    with (tmp_path / "job_band_key.json").open() as f:
        key = json.load(f)
    assert key["path"] == results.uri.uri.name
    assert key["bands"] == [Band.Schema().dump(b) for b in results.get_bands()]
    assert {u.uri for u in results.get_all_uris()} == {
        results.uri.uri,
        output,
        dr._get_population_output_path(output),
    }


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize("n_cpus", [1, 2])
def test_mixed_integer_spi_and_float_population_survive_tiling(
    tmp_path, by_sex, n_cpus
):
    bands, arrays = _inputs(by_sex)
    inputs = []
    for i, (band, array) in enumerate(zip(bands, arrays)):
        source = tmp_path / f"input_{i}.tif"
        datatype = (
            gdal.GDT_Float32 if band.name == dr.POPULATION_BAND_NAME else gdal.GDT_Int16
        )
        _write_raster(source, array[np.newaxis], datatype)
        inputs.append(DataFile(source, [band]))

    summary, outputs, error = dr._summarize_over_aoi(
        wkt_aoi="POLYGON ((0 1.998, 0.004 1.998, 0.004 2, 0 2, 0 1.998))",
        pixel_aligned_bbox=(0, 1.998, 0.004, 2),
        in_dfs=inputs,
        output_tif_path=tmp_path / "summary.tif",
        mask_worker_process_name="mask",
        drought_worker_process_name="drought",
        drought_period=4,
        n_cpus=n_cpus,
        parallel_backend="thread",
    )
    assert not error
    assert summary.annual_population_by_drought_class_total[0][2] == pytest.approx(
        40000.75 * (2 if by_sex else 1)
    )
    integer = gdal.Open(str(outputs[DataType.INT16][0]))
    assert integer.GetRasterBand(1).DataType == gdal.GDT_Int16
    integer = None
    vrt = gdal.BuildVRT(
        str(tmp_path / "summary.vrt"), [str(p) for p in outputs[DataType.FLOAT32]]
    )
    population = vrt.GetRasterBand(1)
    assert population.DataType == gdal.GDT_Float32
    assert population.GetNoDataValue() == dr.NODATA_VALUE
    values = population.ReadAsArray()
    assert values[0, 0] == (-0.5 if by_sex else -0.25)
    assert values[0, 1] == (-80001 if by_sex else -40000.5)
    np.testing.assert_array_equal(values[0, 2:], dr.NODATA_VALUE)


def test_non_drought_population_remains_positive():
    bands, arrays = _inputs(False)
    arrays[0, 0, :2] = [1000, 0]
    params = dr.DroughtSummaryParams(
        DataFile("in.vrt", bands), "out.tif", 4, "mask.tif"
    )
    _, writes = dr._process_block(
        params, arrays, np.zeros((2, 4), bool), 0, 0, np.ones((2, 1))
    )
    np.testing.assert_array_equal(writes[2]["array"][0, :2], [0.25, 40000.5])
    np.testing.assert_array_equal(writes[2]["array"][0, 2:], dr.NODATA_VALUE)


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize(
    "units,factor",
    [
        ("people per hectare", 100),
        ("people per square kilometre", 1),
    ],
)
def test_drought_population_density_converts_to_counts(by_sex, units, factor):
    bands, arrays = _inputs(by_sex)
    for band in bands:
        if band.name == dr.POPULATION_BAND_NAME:
            band.metadata["units"] = units
    params = dr.DroughtSummaryParams(
        DataFile("in.vrt", bands), "out.tif", 4, "mask.tif"
    )
    _, writes = dr._process_block(
        params,
        arrays,
        np.zeros((2, 4), bool),
        0,
        0,
        np.full((2, 1), 0.001),
    )
    expected = -0.25 * 0.001 * factor * (2 if by_sex else 1)
    assert writes[2]["array"][0, 0] == pytest.approx(expected)


def test_missing_one_sex_does_not_mask_the_other_sex():
    bands, arrays = _inputs(True)
    arrays[2, 0, 0] = dr.NODATA_VALUE
    params = dr.DroughtSummaryParams(
        DataFile("in.vrt", bands), "out.tif", 4, "mask.tif"
    )
    _, writes = dr._process_block(
        params, arrays, np.zeros((2, 4), bool), 0, 0, np.ones((2, 1))
    )
    assert writes[2]["array"][0, 0] == dr.NODATA_VALUE
    assert writes[3]["array"][0, 0] == dr.NODATA_VALUE
    assert writes[4]["array"][0, 0] == -0.25


def _job_for_inputs(source, bands):
    def indices_for(name):
        return [i for i, band in enumerate(bands, start=1) if band.name == name]

    spi_indices = indices_for(dr.SPI_BAND_NAME)
    population_indices = indices_for(dr.POPULATION_BAND_NAME)
    jrc_index = indices_for(dr.JRC_BAND_NAME)[0]
    water_index = indices_for(dr.WATER_MASK_BAND_NAME)[0]
    return SimpleNamespace(
        task_name="drought split outputs",
        params={
            "layer_spi_path": str(source),
            "layer_spi_bands": [Band.Schema().dump(bands[i - 1]) for i in spi_indices],
            "layer_spi_band_indices": spi_indices,
            "layer_spi_years": [bands[i - 1].metadata["year"] for i in spi_indices],
            "layer_spi_lag": 12,
            "layer_population_path": str(source),
            "layer_population_bands": [
                Band.Schema().dump(bands[i - 1]) for i in population_indices
            ],
            "layer_population_band_indices": population_indices,
            "layer_jrc_path": str(source),
            "layer_jrc_band": Band.Schema().dump(bands[jrc_index - 1]),
            "layer_jrc_band_index": jrc_index,
            "layer_water_path": str(source),
            "layer_water_band": Band.Schema().dump(bands[water_index - 1]),
            "layer_water_band_index": water_index,
        },
    )


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("nodata", [dr.NODATA_VALUE, -9999])
@pytest.mark.parametrize(
    "units",
    [None, "people per pixel", "people per hectare", "people per square kilometre"],
)
def test_coarse_population_summary_preserves_counts(
    tmp_path, monkeypatch, by_sex, units, nodata, backend
):
    bands, arrays = _inputs(by_sex, shape=(10, 20))
    arrays[:] = 0
    arrays[0] = -1500
    source = tmp_path / "indicators.tif"
    _write_raster(source, arrays)
    population_bands = [band for band in bands if band.name == dr.POPULATION_BAND_NAME]
    population = tmp_path / "population.tif"
    cell_area_ha = dr.calc_cell_area(2, 1.99, 0.01) / 10000
    value = 500.25
    if units == "people per hectare":
        value /= cell_area_ha
    elif units == "people per square kilometre":
        value /= cell_area_ha / 100
    _write_raster(
        population,
        np.array([[[value, nodata]]] * len(population_bands), dtype=np.float32),
    )
    dataset = gdal.Open(str(population), gdal.GA_Update)
    dataset.SetGeoTransform((0, 0.01, 0, 2, 0, -0.01))
    for index in range(1, dataset.RasterCount + 1):
        dataset.GetRasterBand(index).SetNoDataValue(nodata)
    dataset = None
    for band in population_bands:
        if units is not None:
            band.metadata["units"] = units
    job = _job_for_inputs(source, bands)
    job.params.update(
        layer_population_path=str(population),
        layer_population_bands=[Band.Schema().dump(b) for b in population_bands],
        layer_population_band_indices=list(range(1, len(population_bands) + 1)),
    )
    original_metadata = deepcopy(job.params["layer_population_bands"])
    aoi = SimpleNamespace(
        meridian_split=lambda **_: [
            "POLYGON ((0 1.99, 0.02 1.99, 0.02 2, 0 2, 0 1.99))"
        ],
        get_aligned_output_bounds=lambda _: [(0, 1.99, 0.02, 2)],
    )
    summaries = []
    monkeypatch.setattr(
        dr,
        "save_reporting_json",
        lambda _path, summary, *_: summaries.append(summary) or {},
    )
    monkeypatch.setattr(dr, "save_summary_table_excel", lambda *_, **__: None)
    normalized_paths = []
    prepare = dr._prepare_population_dfs

    def track_population_inputs(params, directory):
        dfs = prepare(params, directory)
        for df in dfs:
            assert df.bands[0].metadata["units"] == "people per hectare"
            assert Path(df.path).exists()
            normalized_paths.append(Path(df.path))
        return dfs

    monkeypatch.setattr(dr, "_prepare_population_dfs", track_population_inputs)
    result = dr.summarise_drought_vulnerability(
        job, aoi, tmp_path / "job.json", n_cpus=2, parallel_backend=backend
    )
    assert normalized_paths
    assert all(not path.parent.exists() for path in normalized_paths)
    expected = 500.25 * (2 if by_sex else 1)
    assert summaries[0].annual_population_by_drought_class_total[0][2] == pytest.approx(
        expected, rel=1e-6
    )
    for band_index, band in enumerate(result.get_bands(), start=1):
        if band.name != dr.POP_AT_MAX_DROUGHT_BAND_NAME:
            continue
        dataset = gdal.Open(str(result.uri.uri))
        values = dataset.GetRasterBand(band_index).ReadAsArray()
        assert values.shape == (10, 20)
        np.testing.assert_array_equal(values[:, 10:], dr.NODATA_VALUE)
        expected_band = expected if band.metadata["type"] == "total" else 500.25
        assert -values[:, :10].sum() == pytest.approx(expected_band, rel=1e-6)
        dataset = None
    assert job.params["layer_population_bands"] == original_metadata


def test_population_temporary_directory_is_removed_on_failure(tmp_path, monkeypatch):
    directories = []

    def fail(_params, directory):
        assert directory.is_dir()
        directories.append(directory)
        (directory / "population_0.tif").touch()
        raise RuntimeError("Cancelled by user.")

    bands, arrays = _inputs(False)
    source = tmp_path / "source.tif"
    _write_raster(source, arrays)
    monkeypatch.setattr(dr, "_prepare_population_dfs", fail)
    with pytest.raises(RuntimeError, match="Cancelled"):
        dr.summarise_drought_vulnerability(
            _job_for_inputs(source, bands), None, tmp_path / "job.json", n_cpus=1
        )
    assert len(directories) == 1
    assert not directories[0].exists()


@pytest.mark.parametrize(
    "units,value,message",
    [
        ("unsupported", 1, "Unsupported population units"),
        ("people per pixel", -1, "non-negative"),
    ],
)
def test_invalid_population_fails_before_drought_resampling(
    tmp_path, monkeypatch, units, value, message
):
    bands, arrays = _inputs(False)
    arrays[1] = value
    bands[1].metadata["units"] = units
    source = tmp_path / "source.tif"
    _write_raster(source, arrays)
    directories = []
    prepare = dr._prepare_population_dfs

    def track_preparation(params, directory):
        directories.append(directory)
        return prepare(params, directory)

    def unexpected_summary(**kwargs):
        pytest.fail("Invalid population must be rejected before summary resampling")

    monkeypatch.setattr(dr, "_prepare_population_dfs", track_preparation)
    monkeypatch.setattr(dr, "_compute_drought_summary_table", unexpected_summary)
    with pytest.raises(ValueError, match=message):
        dr.summarise_drought_vulnerability(
            _job_for_inputs(source, bands), None, tmp_path / "job.json", n_cpus=1
        )
    assert len(directories) == 1
    assert not directories[0].exists()


@pytest.mark.parametrize("by_sex", [False, True])
@pytest.mark.parametrize("n_regions", [1, 2])
@pytest.mark.parametrize("backend", ["thread", "process"])
def test_split_results_mosaic_metadata_and_relocation(
    tmp_path, monkeypatch, by_sex, n_regions, backend
):
    bands, arrays = _inputs(by_sex, n_years=5)
    source = tmp_path / "source.tif"
    _write_raster(source, arrays)
    if n_regions == 1:
        bounds = [(0, 1.998, 0.004, 2)]
    else:
        bounds = [(0, 1.998, 0.002, 2), (0.002, 1.998, 0.004, 2)]
    polygons = [
        f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
        for x0, y0, x1, y1 in bounds
    ]
    aoi = SimpleNamespace(
        meridian_split=lambda **_: polygons,
        get_aligned_output_bounds=lambda _: bounds,
    )
    monkeypatch.setattr(dr.workers, "_get_tile_size", lambda *_: (2, 1))
    monkeypatch.setattr(dr, "save_reporting_json", lambda *_: {})
    monkeypatch.setattr(dr, "save_summary_table_excel", lambda *_, **__: None)

    result = dr.summarise_drought_vulnerability(
        _job_for_inputs(source, bands),
        aoi,
        tmp_path / "job.json",
        n_cpus=2,
        parallel_backend=backend,
    )
    result = RasterResults.Schema().load(RasterResults.Schema().dump(result))
    assert list(result.rasters) == [DataType.INT16.value, DataType.FLOAT32.value]
    for datatype, n_bands in (
        (DataType.INT16, 2),
        (DataType.FLOAT32, 4 if by_sex else 2),
    ):
        raster = result.rasters[datatype.value]
        assert raster.type == RasterType.TILED_RASTER
        assert len(raster.tile_uris) == 4
        assert raster.datatype == datatype
        for uri in raster.tile_uris:
            dataset = gdal.Open(str(uri.uri))
            assert dataset.RasterCount == n_bands
            for i in range(1, n_bands + 1):
                assert dataset.GetRasterBand(i).DataType == (
                    gdal.GDT_Int16 if datatype == DataType.INT16 else gdal.GDT_Float32
                )
                assert dataset.GetRasterBand(i).GetNoDataValue() == dr.NODATA_VALUE
                assert dataset.GetRasterBand(i).GetDescription()
            dataset = None

    expected = np.array(
        [
            [-0.25, -40000.5, dr.NODATA_VALUE, dr.NODATA_VALUE],
            [dr.NODATA_VALUE, dr.NODATA_VALUE, 0, dr.NODATA_VALUE],
        ]
    )
    expected_total = expected.copy()
    if by_sex:
        expected_total[expected_total != dr.NODATA_VALUE] *= 2
    original_descriptions = []
    dataset = gdal.Open(str(result.uri.uri))
    for i, band in enumerate(result.get_bands(), start=1):
        actual = dataset.GetRasterBand(i)
        assert actual.GetNoDataValue() == dr.NODATA_VALUE
        original_descriptions.append(actual.GetDescription())
        if band.name == dr.SPI_MIN_OVER_PERIOD_BAND_NAME:
            assert actual.DataType == gdal.GDT_Int16
            np.testing.assert_array_equal(actual.ReadAsArray(), arrays[0])
        else:
            assert actual.DataType == gdal.GDT_Float32
            np.testing.assert_array_equal(
                actual.ReadAsArray(),
                expected_total if band.metadata["type"] == "total" else expected,
            )
    actual = None
    dataset = None

    with (tmp_path / "job_band_key.json").open() as f:
        key = json.load(f)
    assert key["bands"] == [Band.Schema().dump(b) for b in result.get_bands()]

    # Every VRT source must be tracked and portable with the result bundle.
    moved = tmp_path / "moved"
    moved.mkdir()
    paths = {uri.uri for uri in result.get_all_uris()}
    assert len(paths) == 11  # main VRT, two mosaics, and eight physical tiles
    for path in paths:
        shutil.copy2(path, moved / path.name)
    for path in paths:
        path.unlink()
    result.update_uris(moved / "job.json")
    assert {uri.uri for uri in result.get_all_uris()} == {
        moved / path.name for path in paths
    }
    dataset = gdal.Open(str(result.uri.uri))
    np.testing.assert_array_equal(
        dataset.GetRasterBand(3).ReadAsArray(), expected_total
    )
    assert [
        dataset.GetRasterBand(i).GetDescription()
        for i in range(1, dataset.RasterCount + 1)
    ] == original_descriptions
    dataset = None


def test_population_output_path_cannot_overwrite_indicators(tmp_path):
    path = str(tmp_path / "summary.tif")
    with pytest.raises(ValueError, match="different paths"):
        dr.DroughtSummaryParams(DataFile(path, []), path, 4, "mask.tif", path)


def test_custom_worker_must_write_population_output(tmp_path):
    bands, arrays = _inputs(False)
    source = tmp_path / "source.tif"
    output = tmp_path / "summary.tif"
    _write_raster(source, arrays)

    def incomplete_worker(params):
        _write_raster(output, arrays[:1], gdal.GDT_Int16)
        return dr.SummaryTableDrought([], [], [], [], (0.0, 0))

    tile_input = dr.SummarizeTileInputs(
        tile=source,
        out_file=output,
        aoi="POLYGON ((0 1.998, 0.004 1.998, 0.004 2, 0 2, 0 1.998))",
        drought_period=4,
        in_dfs=[DataFile(source, bands)],
        drought_worker_function=incomplete_worker,
        drought_worker_params={},
    )
    with pytest.raises(RuntimeError, match="summary_population.tif"):
        dr._aoi_process_sequential([tile_input])


@pytest.mark.parametrize("spi_name", [dr.SPI_BAND_NAME, dr.SPEI_BAND_NAME])
def test_split_writer_preserves_period_and_sex_band_order(tmp_path, spi_name):
    bands, arrays = _inputs(True, n_years=9)
    for year in range(9):
        bands[year].name = spi_name
        arrays[year] = -1000 - 100 * year
        arrays[9 + year] = 0.25 + year
        arrays[18 + year] = 20.5 + year
    source = tmp_path / "source.tif"
    mask = tmp_path / "mask.tif"
    output = tmp_path / "summary.tif"
    _write_raster(source, arrays)
    _write_raster(mask, np.zeros((1, 2, 4)))
    params = dr.DroughtSummaryParams(DataFile(source, bands), str(output), 4, str(mask))
    worker = dr.DroughtSummary(params)
    worker.process_lines(worker.get_line_params())

    dataset = gdal.Open(str(output))
    assert dataset.RasterCount == 3
    for band_index, driest_year in enumerate([3, 7, 8], start=1):
        np.testing.assert_array_equal(
            dataset.GetRasterBand(band_index).ReadAsArray(), arrays[driest_year]
        )
    dataset = None
    dataset = gdal.Open(str(dr._get_population_output_path(output)))
    assert dataset.RasterCount == 5
    for band_index, driest_year in enumerate([3, 7, 8], start=1):
        expected = -(20.75 + 2 * driest_year)
        assert dataset.GetRasterBand(band_index).ReadAsArray()[0, 0] == expected
    assert dataset.GetRasterBand(4).ReadAsArray()[0, 0] == -28.5
    assert dataset.GetRasterBand(5).ReadAsArray()[0, 0] == -8.25
    dataset = None
