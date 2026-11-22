"""Focused tests for blockwise LDN prioritization and selection."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal")
pytest.importorskip("osgeo.ogr")

from osgeo import gdal, ogr, osr

from te_algorithms.gdal import util
from te_algorithms.gdal.land_deg import ldn_priority, ldn_priority_inputs


def _write_raster(path, values, dtype=gdal.GDT_Float32, nodata=-32768):
    values = np.asarray(values)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1, dtype
    )
    ds.SetGeoTransform((0, 0.1, 0, 5, 0, -0.1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(values)
    ds = None


def _criterion(path, arr_class, **kwargs):
    return {
        "name": f"{arr_class} criterion",
        "path": str(path),
        "field": "",
        "arr_class": arr_class,
        "band_index": 1,
        "source": "fixture",
        "date": "2026-01",
        "units": "index",
        "license": "CC0",
        "weight": 1.0,
        "direction": "higher",
        "minimum": 0,
        "maximum": 10,
        "resampling": "nearest",
        **kwargs,
    }


def _inputs(tmp_path):
    arr_path = tmp_path / "arr.tif"
    values_path = tmp_path / "criterion.tif"
    _write_raster(
        arr_path,
        np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int16),
        dtype=gdal.GDT_Int16,
    )
    _write_raster(
        values_path,
        np.array([[0, -32768, 5], [10, 2, 8]], dtype=np.float32),
    )
    criteria = [
        _criterion(values_path, "avoid"),
        _criterion(values_path, "reduce"),
        _criterion(values_path, "reverse"),
    ]
    return arr_path, values_path, criteria


def test_input_sampling_categories_and_selection_estimates(tmp_path):
    path = tmp_path / "scores.tif"
    _write_raster(path, np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]))
    info = util.inspect_source(str(path))
    assert info["raster"]["bands"][0]["data_type"] == "Float32"
    stats = util.observed_band_stats(str(path), 1)
    assert stats["minimum"] == pytest.approx(0.1)
    categories = util.discover_categories(str(path), "raster")
    assert len(categories["counts"]) == 10
    threshold = ldn_priority_inputs.estimate_selection(
        str(path), 1, "threshold", 0.75, 0, 10
    )
    assert threshold["selected_fraction"] == pytest.approx(0.3)
    assert threshold["selected_area_km2"] == pytest.approx(3)
    area = ldn_priority_inputs.estimate_selection(str(path), 1, "area", 0, 2, 10)
    assert area["selected_area_km2"] == 2
    assert area["binding_score"] == pytest.approx(0.82)
    uncapped = ldn_priority_inputs.estimate_selection(str(path), 1, "area", 0, 50, 10)
    assert uncapped["selected_area_km2"] == 10
    assert uncapped["binding_score"] is None


def test_vector_sampling_counts_nulls_and_keeps_category_tie_order(
    tmp_path, monkeypatch
):
    path = tmp_path / "categories.gpkg"
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = dataset.CreateLayer("categories", geom_type=ogr.wkbNone)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
    for value in (3, 2, None, 1):
        feature = ogr.Feature(layer.GetLayerDefn())
        if value is not None:
            feature.SetField("value", value)
        layer.CreateFeature(feature)
    dataset = None
    stats = util.observed_field_stats(str(path), "value")
    assert stats["null_count"] == 1
    assert stats["mean"] == 2
    monkeypatch.setattr(util, "MAX_DISCOVERED_CATEGORIES", 2)
    categories = util.discover_categories(str(path), "vector", field="value")
    assert list(categories["counts"]) == [3.0, 2.0]
    assert categories["truncated"]
    with pytest.raises(ValueError, match="input_kind"):
        util.discover_categories(str(path), "other")


def test_sample_limits_nodata_and_first_vector_layer(tmp_path, monkeypatch):
    path = tmp_path / "raster.tif"
    _write_raster(path, [[1, -32768], [float("nan"), 2]])
    sample = util.sampled_raster_values(str(path), 1)
    assert sample["values"].tolist() == [1, 2]
    assert not sample["sampled"]
    monkeypatch.setattr(util, "STATS_SAMPLE_PIXELS", 1)
    assert util.sampled_raster_values(str(path), 1)["sampled"]

    path = tmp_path / "layers.gpkg"
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    for name, values in (("first", [1, 2, 3]), ("second", [99])):
        layer = dataset.CreateLayer(name, geom_type=ogr.wkbNone)
        layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
        for value in values:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("value", value)
            layer.CreateFeature(feature)
    dataset = None
    monkeypatch.setattr(util, "VECTOR_STATS_FEATURE_LIMIT", 2)
    sample = util.sampled_field_values(str(path), "value")
    assert sample["values"].tolist() == [1, 2]
    assert sample["sampled"]
    stats = util.observed_field_stats(str(path), "value")
    assert stats["mean"] == 1.5
    assert stats["sampled"]
    with pytest.raises(ValueError, match="no field"):
        util.sampled_field_values(str(path), "missing")


def test_validate_criteria_normalizes_weights_and_rejects_all_zero(tmp_path):
    arr_path, path, criteria = _inputs(tmp_path)
    second_avoid = _criterion(path, "avoid", name="second avoid", weight=1.0)
    criteria[0]["weight"] = 2.0
    raw_criteria = [criteria[0], second_avoid, *criteria[1:]]
    normalized = ldn_priority.validate_criteria(raw_criteria)
    assert normalized[0]["weight"] == pytest.approx(2 / 3)
    assert normalized[1]["weight"] == pytest.approx(1 / 3)
    summary, _ = ldn_priority.compute_prioritization(
        str(arr_path), raw_criteria, str(tmp_path / "normalized_scores.tif")
    )
    assert summary["criteria"][0]["weight"] == pytest.approx(2 / 3)
    assert summary["criteria"][1]["weight"] == pytest.approx(1 / 3)

    criteria[0]["license"] = ""
    with pytest.raises(ValueError, match="license"):
        ldn_priority.validate_criteria(criteria)

    assert str(path) in criteria[0]["path"]


def test_validate_criteria_rejects_all_zero_class_weights(tmp_path):
    _, _, criteria = _inputs(tmp_path)
    criteria[0]["weight"] = 0.0
    with pytest.raises(ValueError, match="positive total"):
        ldn_priority.validate_criteria(criteria)


def test_compute_scores_complete_case_coverage_and_provenance(tmp_path):
    arr_path, _, criteria = _inputs(tmp_path)
    score_path = tmp_path / "priority.tif"
    summary, output = ldn_priority.compute_prioritization(
        str(arr_path), criteria, str(score_path)
    )
    assert output == str(score_path)
    assert summary["scoring_policy"] == "complete_case_weighted_mean"
    assert summary["classes"]["avoid"]["coverage_fraction"] == pytest.approx(0.5)
    assert summary["classes"]["reduce"]["coverage_fraction"] == pytest.approx(1.0)
    assert summary["classes"]["reverse"]["mean_score_area_weighted"] == pytest.approx(
        0.5
    )
    assert summary["criteria"][0]["source"] == "fixture"
    assert summary["criteria"][0][
        "mean_weighted_contribution_area_weighted"
    ] == pytest.approx(0.0)
    assert summary["criteria"][2][
        "mean_normalized_score_area_weighted"
    ] == pytest.approx(0.5)
    assert summary["provenance"]["arr_sha256"]

    ds = gdal.Open(str(score_path))
    assert ds.RasterCount == 3
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32
    scores = ds.GetRasterBand(1).ReadAsArray()
    assert scores[0, 0] == pytest.approx(0.0)
    assert scores[0, 1] == -32768
    ds = None
    coverage_ds = gdal.Open(summary["coverage_path"])
    coverage = coverage_ds.GetRasterBand(1).ReadAsArray()
    assert coverage[0, 0] == 1
    assert coverage[0, 1] == 0
    coverage_ds = None


def test_land_type_summary_uses_supplied_labels(tmp_path):
    arr_path, _, criteria = _inputs(tmp_path)
    land_type_path = tmp_path / "land_types.tif"
    _write_raster(
        land_type_path,
        np.array([[1, 2, 2], [1, 1, 3]], dtype=np.int16),
        dtype=gdal.GDT_Int16,
    )
    summary, _ = ldn_priority.compute_prioritization(
        str(arr_path),
        criteria,
        str(tmp_path / "labelled.tif"),
        land_type_path=str(land_type_path),
        land_type_labels={"1": "Forest_Clay", 2: "Crop_Sand"},
    )
    names = {
        (row["arr_class"], row["land_type_id"]): row["name"]
        for row in summary["by_land_type"]
    }
    assert names[("avoid", 1)] == "Forest_Clay"
    assert names[("avoid", 2)] == "Crop_Sand"
    assert names[("reverse", 3)] == "3"


def test_zero_weight_missing_criterion_does_not_reduce_coverage(tmp_path):
    arr_path, _, criteria = _inputs(tmp_path)
    missing_path = tmp_path / "missing_criterion.tif"
    _write_raster(missing_path, np.full((2, 3), -32768, dtype=np.float32))
    criteria.append(
        _criterion(
            missing_path,
            "avoid",
            name="optional zero-weight criterion",
            weight=0.0,
        )
    )

    summary, _ = ldn_priority.compute_prioritization(
        str(arr_path), criteria, str(tmp_path / "zero_weight_score.tif")
    )
    assert summary["classes"]["avoid"]["coverage_fraction"] == pytest.approx(0.5)
    assert summary["criteria"][-1]["coverage_fraction"] == 0.0


def test_top_area_is_deterministic_and_threshold_includes_zero_score(tmp_path):
    arr_path, _, criteria = _inputs(tmp_path)
    score_path = tmp_path / "priority.tif"
    summary, _ = ldn_priority.compute_prioritization(
        str(arr_path), criteria, str(score_path)
    )
    pixel_area = summary["classes"]["reverse"]["eligible_area_km2"] / 2
    mask_path = tmp_path / "selected.tif"
    selection_summary, _ = ldn_priority.select_priority_areas(
        str(score_path),
        str(mask_path),
        {
            "reverse": {"mode": "area", "area_km2": pixel_area},
            "avoid": {"mode": "threshold", "threshold": 0.0},
        },
    )
    assert selection_summary["selections"]["reverse"]["selected_pixels"] == 1
    assert selection_summary["selections"]["reverse"]["selected_area_km2"] <= pixel_area
    assert selection_summary["selections"]["reverse"][
        "unspent_area_km2"
    ] == pytest.approx(0.0)
    assert selection_summary["score_sha256"]
    ds = gdal.Open(str(mask_path))
    mask = ds.GetRasterBand(1).ReadAsArray()
    assert mask[1, 2] == 3
    assert mask[1, 1] == 0
    assert mask[0, 0] == 1
    ds = None


def test_equal_score_top_area_tie_uses_row_major_pixel_order(tmp_path):
    score_path = tmp_path / "tie_scores.tif"
    ds = gdal.GetDriverByName("GTiff").Create(
        str(score_path), 2, 2, 3, gdal.GDT_Float32
    )
    ds.SetGeoTransform((0, 0.1, 0, 5, 0, -0.1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for band_index in (1, 2, 3):
        band = ds.GetRasterBand(band_index)
        band.SetNoDataValue(-32768)
        band.Fill(-32768)
    ds.GetRasterBand(3).WriteArray(
        np.array([[0.5, -32768], [-32768, 0.5]], dtype=np.float32)
    )
    projection = ds.GetProjection()
    ds = None

    pixel_area = ldn_priority.ldn_planning._compute_pixel_areas_km2(
        (0, 0.1, 0, 5, 0, -0.1), 0, 1, 2, projection
    )[0, 0]
    mask_path = tmp_path / "tie_selected.tif"
    summary, _ = ldn_priority.select_priority_areas(
        str(score_path),
        str(mask_path),
        {"reverse": {"mode": "area", "area_km2": pixel_area}},
    )
    assert summary["selections"]["reverse"]["selected_pixels"] == 1
    ds = gdal.Open(str(mask_path))
    selected = ds.GetRasterBand(1).ReadAsArray()
    assert selected[0, 0] == 3
    assert selected[1, 1] == 0
    ds = None


def test_top_area_never_exceeds_cap_and_skips_pixels_that_do_not_fit(tmp_path):
    score_path = tmp_path / "heterogeneous_scores.tif"
    ds = gdal.GetDriverByName("GTiff").Create(
        str(score_path), 1, 2, 3, gdal.GDT_Float32
    )
    geotransform = (0, 0.1, 0, 80, 0, -40)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    projection = srs.ExportToWkt()
    ds.SetProjection(projection)
    for band_index in (1, 2, 3):
        band = ds.GetRasterBand(band_index)
        band.SetNoDataValue(-32768)
        band.Fill(-32768)
    areas = ldn_priority.ldn_planning._compute_pixel_areas_km2(
        geotransform, 0, 2, 1, projection
    )[:, 0]
    large_row = int(np.argmax(areas))
    small_row = 1 - large_row
    reverse_scores = np.full((2, 1), -32768, dtype=np.float32)
    reverse_scores[large_row, 0] = 1.0
    reverse_scores[small_row, 0] = 0.5
    ds.GetRasterBand(3).WriteArray(reverse_scores)
    ds = None

    cap = float((areas[large_row] + areas[small_row]) / 2)
    selected_summary, _ = ldn_priority.select_priority_areas(
        str(score_path),
        str(tmp_path / "heterogeneous_selected.tif"),
        {"reverse": {"mode": "area", "area_km2": cap}},
    )
    selected = selected_summary["selections"]["reverse"]
    assert selected["selected_pixels"] == 1
    assert selected["selected_area_km2"] == pytest.approx(areas[small_row])
    assert selected["selected_area_km2"] <= cap
    assert selected["unspent_area_km2"] == pytest.approx(cap - areas[small_row])

    empty_summary, empty_path = ldn_priority.select_priority_areas(
        str(score_path),
        str(tmp_path / "smaller_than_pixel.tif"),
        {"reverse": {"mode": "area", "area_km2": float(areas.min() / 2)}},
    )
    assert empty_summary["selections"]["reverse"]["selected_pixels"] == 0
    assert empty_summary["selections"]["reverse"]["selected_area_km2"] == 0.0
    assert empty_summary["selections"]["reverse"]["unspent_area_km2"] > 0
    ds = gdal.Open(empty_path)
    assert not np.any(ds.GetRasterBand(1).ReadAsArray() == 3)
    ds = None


def test_provenance_hash_includes_vrt_referenced_raster(tmp_path):
    source_path = tmp_path / "vrt_source.tif"
    _write_raster(source_path, np.array([[1]], dtype=np.float32))
    vrt_path = tmp_path / "source.vrt"
    vrt = gdal.BuildVRT(str(vrt_path), [str(source_path)])
    assert vrt is not None
    vrt = None

    vrt_ds = gdal.Open(str(vrt_path))
    listed_files = {str(Path(item).resolve()) for item in vrt_ds.GetFileList()}
    vrt_ds = None
    assert str(source_path.resolve()) in listed_files

    before = ldn_priority._sha256(str(vrt_path))
    _write_raster(source_path, np.array([[9]], dtype=np.float32))
    after = ldn_priority._sha256(str(vrt_path))
    assert before != after


def test_vector_field_criterion_is_rasterized_and_scored(tmp_path):
    arr_path = tmp_path / "arr.tif"
    _write_raster(arr_path, np.array([[1, 1]], dtype=np.int16), gdal.GDT_Int16)
    vector_path = tmp_path / "criterion.gpkg"
    datasource = ogr.GetDriverByName("GPKG").CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = datasource.CreateLayer("criteria", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTReal))
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in [(0, 5), (0.1, 5), (0.1, 4.9), (0, 4.9), (0, 5)]:
        ring.AddPoint(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("value", 5.0)
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)
    datasource = None

    criterion = _criterion(
        vector_path,
        "avoid",
        field="value",
        minimum=0,
        maximum=10,
    )
    summary, _ = ldn_priority.compute_prioritization(
        str(arr_path), [criterion], str(tmp_path / "vector_score.tif")
    )
    assert summary["classes"]["avoid"]["scored_pixels"] == 1
    assert summary["classes"]["avoid"]["mean_score_area_weighted"] == pytest.approx(0.5)


def test_cancelled_prioritization_removes_partial_outputs(tmp_path):
    arr_path, _, criteria = _inputs(tmp_path)
    score_path = tmp_path / "cancelled.tif"
    with pytest.raises(RuntimeError, match="cancelled"):
        ldn_priority.compute_prioritization(
            str(arr_path),
            criteria,
            str(score_path),
            killed_callback=lambda: True,
        )
    assert not score_path.exists()
    assert not (tmp_path / "cancelled_coverage.tif").exists()
