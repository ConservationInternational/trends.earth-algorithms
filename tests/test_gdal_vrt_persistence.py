import ntpath
import shutil
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal")

from osgeo import gdal, osr
from te_schemas.datafile import DataFile
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band, DataType

from te_algorithms.gdal import util
from te_algorithms.gdal.land_deg import config, land_deg


def _write_raster(path, values, datatype=gdal.GDT_Int16):
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path), 2, 2, len(values), datatype
    )
    dataset.SetGeoTransform((0, 1, 0, 2, 0, -1))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    for index, value in enumerate(values, start=1):
        dataset.GetRasterBand(index).Fill(value)
    dataset = None


def _assert_local_dependencies(path, directory, seen=None):
    seen = set() if seen is None else seen
    if path in seen:
        return
    seen.add(path)
    for source in ElementTree.parse(path).iter("SourceFilename"):
        assert source.get("relativeToVRT") == "1"
        assert not Path(source.text).is_absolute()
        dependency = (path.parent / source.text).resolve()
        assert dependency.is_relative_to(directory.resolve())
        assert dependency.is_file()
        if dependency.suffix == ".vrt":
            _assert_local_dependencies(dependency, directory, seen)


def test_period_outputs_survive_temporary_cleanup_and_relocation(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    source = temporary / "input.tif"
    _write_raster(source, [3, 5, 7])
    materialized = output / "baseline_materialized.vrt"
    paths = []
    for index in range(1, 4):
        path = output / f"materialized_band_{index}.tif"
        dataset = gdal.Translate(
            str(path), str(source), format="GTiff", bandList=[index]
        )
        dataset = None
        paths.append(str(path))
    dataset = gdal.BuildVRT(str(materialized), paths, separate=True)
    dataset = None
    integer = output / "baseline_sdg.tif"
    population = output / "baseline_population.tif"
    _write_raster(integer, [-1])
    _write_raster(population, [40000.5], gdal.GDT_Float32)
    bands = [
        Band(config.LC_DEG_BAND_NAME, metadata={}, no_data_value=-32768),
        Band(config.SOC_DEG_BAND_NAME, metadata={}, no_data_value=-32768),
        Band(
            config.POPULATION_BAND_NAME,
            metadata={"type": "total"},
            no_data_value=-32768,
        ),
    ]
    input_df = DataFile(materialized, bands)
    period = {
        "period": {"year_initial": 2000, "year_final": 2015},
        "periods": {"productivity": {"year_initial": 2000, "year_final": 2015}},
    }
    original_save = util.save_vrt2

    def save_selection(source_path, band_indices, output_path=None):
        if output_path is None:
            output_path = temporary / f"selection_{band_indices[0]}.vrt"
            dataset = gdal.BuildVRT(
                str(output_path), str(source_path), bandList=band_indices
            )
            assert dataset is not None
            dataset = None
            return str(output_path)
        return original_save(source_path, band_indices, output_path=output_path)

    monkeypatch.setattr(util, "save_vrt2", save_selection)
    datafiles, vrts = land_deg._build_period_rasters(
        integer,
        population,
        materialized,
        [input_df],
        [DataFile(materialized, [bands[-1]])],
        ProductivityMode.JRC_5_CLASS_LPD.value,
        period,
        output / "summary.json",
        output / "summary_baseline.json",
        "baseline",
    )
    overall = output / "summary.vrt"
    util.combine_all_bands_into_vrt(list(vrts.values()), overall)
    shutil.rmtree(temporary)
    relocated = tmp_path / "relocated"
    shutil.move(str(output), relocated)
    _assert_local_dependencies(relocated / overall.name, relocated)
    dataset = gdal.Open(str(relocated / overall.name))
    assert dataset.RasterCount == 5
    for index, value in enumerate([-1, 3, 5, 40000.5, 7], start=1):
        np.testing.assert_allclose(dataset.GetRasterBand(index).ReadAsArray(), value)
    assert datafiles[DataType.INT16].bands[1].name == config.LC_DEG_BAND_NAME
    assert datafiles[DataType.FLOAT32].bands[1].name == config.POPULATION_BAND_NAME
    dataset = None


@pytest.mark.parametrize("custom_worker", [False, True])
def test_saved_tile_uses_materialized_inputs(tmp_path, monkeypatch, custom_worker):
    source = tmp_path / "temporary_input.tif"
    _write_raster(source, [9, 4])
    tile = tmp_path / "baseline.vrt"
    dataset = gdal.BuildVRT(str(tile), [str(source)])
    dataset = None
    materialized = tmp_path / "baseline_materialized.vrt"
    inputs = land_deg.SummarizeTileInputs(
        in_file=tile,
        wkt_aoi="unused",
        in_dfs=[
            DataFile(
                tile,
                [
                    Band(config.LC_DEG_BAND_NAME, metadata={}, no_data_value=-32768),
                    Band(
                        config.POPULATION_BAND_NAME,
                        metadata={"type": "total"},
                        no_data_value=-32768,
                    ),
                ],
            )
        ],
        prod_mode=ProductivityMode.JRC_5_CLASS_LPD.value,
        lc_legend_nesting=None,
        lc_trans_matrix=None,
        period_name="baseline",
        periods={},
        mask_worker_function=lambda *args: True,
        materialized_path=materialized,
    )
    monkeypatch.setattr(util, "wkt_geom_to_geojson_file_string", lambda _: {})

    class Summarizer:
        def __init__(self, params, *args):
            self.params = params

        def work(self):
            persistent = tmp_path / "materialized_band_1.tif"
            dataset = gdal.Translate(str(persistent), str(tile), format="GTiff")
            assert dataset is not None
            dataset = None
            dataset = gdal.BuildVRT(str(materialized), [str(persistent)])
            dataset = None
            _write_raster(self.params.out_file, [-1])
            _write_raster(Path(self.params.population_out_file), [1])
            return [Mock()]

    monkeypatch.setattr(land_deg.worker, "DegradationSummary", Summarizer)
    if custom_worker:

        def summarize(params):
            _write_raster(params.out_file, [-1])
            _write_raster(Path(params.population_out_file), [1])
            return [Mock()]

        inputs.deg_worker_function = summarize
    monkeypatch.setattr(land_deg.models, "accumulate_summarytableld", lambda _: Mock())
    monkeypatch.setattr(
        land_deg.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(name=str(tmp_path / "mask.tif")),
    )
    result, _, error = land_deg._summarize_tile(inputs)
    assert result is not None
    assert error is None
    source.unlink()
    _assert_local_dependencies(tile, tmp_path)
    dataset = gdal.Open(str(tile))
    np.testing.assert_array_equal(
        dataset.GetRasterBand(1).ReadAsArray(), np.full((2, 2), 9)
    )
    dataset = None


def test_combined_vrt_escapes_paths_and_survives_relocation(tmp_path):
    output = tmp_path / "original"
    output.mkdir()
    source = output / "input & data.tif"
    _write_raster(source, [2])
    combined = output / "combined.vrt"
    util.combine_all_bands_into_vrt([source], combined)
    relocated = tmp_path / "relocated"
    shutil.move(str(output), relocated)
    _assert_local_dependencies(relocated / combined.name, relocated)
    dataset = gdal.Open(str(relocated / combined.name))
    np.testing.assert_array_equal(dataset.ReadAsArray(), np.full((2, 2), 2))
    dataset = None


def test_windows_vrt_source_paths(monkeypatch):
    monkeypatch.setattr(util.os.path, "relpath", ntpath.relpath)
    source = PureWindowsPath(r"C:\datasets\job\input & data.tif")
    output = PureWindowsPath(r"C:\datasets\job\output.vrt")
    assert util._get_vrt_source_path(source, output, True) == (
        "input &amp; data.tif",
        1,
    )
    source = PureWindowsPath(r"D:\datasets\input.tif")
    assert util._get_vrt_source_path(source, output, True) == (str(source), 0)


def test_vsi_source_paths_are_not_relative():
    source = "/vsis3/bucket/input.tif"
    assert util._get_vrt_source_path(source, Path("/output/result.vrt"), True) == (
        source,
        0,
    )


def test_band_selection_preserves_temporary_default(tmp_path):
    source = tmp_path / "source.tif"
    _write_raster(source, [2, 7])
    selection = Path(util.save_vrt2(source, [2]))
    try:
        dataset = gdal.Open(str(selection))
        assert dataset.RasterCount == 1
        np.testing.assert_array_equal(dataset.ReadAsArray(), np.full((2, 2), 7))
        dataset = None
    finally:
        selection.unlink()
