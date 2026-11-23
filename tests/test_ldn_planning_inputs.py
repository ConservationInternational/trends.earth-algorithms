"""Tests for reusable LDN planning input and reporting utilities."""

import copy
import hashlib
import json

import numpy as np
import pytest
from openpyxl import load_workbook

pytest.importorskip("osgeo.gdal")
pytest.importorskip("osgeo.ogr")
from osgeo import gdal, ogr, osr

from te_algorithms.common.checksum import file_checksum
from te_algorithms.gdal import util
from te_algorithms.gdal.land_deg import (
    ldn_planning,
    ldn_planning_inputs as inputs,
    ldn_planning_report as report,
)


def test_effectiveness_and_polygon_validation():
    assert inputs.target_effectiveness(0) == 0
    assert inputs.target_effectiveness(80, "percent") == 0.8
    for value in (float("nan"), float("inf"), -0.1, 80, True):
        with pytest.raises(ValueError):
            inputs.target_effectiveness(value)
    target = inputs.polygon_target(
        "POLYGON ((0 0, 1 0, 1 1, 0 0))", "reverse", 0, "EPSG:4326"
    )
    assert target["effectiveness"] == 0
    assert "WGS 84" in target["geometry_crs"]
    with pytest.raises(ValueError):
        inputs.polygon_target("POINT (0 0)", "avoid", 0.5, "EPSG:4326")
    with pytest.raises(ValueError):
        inputs.polygon_target(target["wkt_geometry"], "avoid", 0.5, "")


def test_checksums_include_vrt_sources_and_geojson(tmp_path):
    raster = tmp_path / "source.tif"
    vrt = tmp_path / "source.vrt"
    ds = gdal.GetDriverByName("GTiff").Create(str(raster), 1, 1, 1, gdal.GDT_Byte)
    ds.GetRasterBand(1).WriteArray(np.array([[1]], dtype=np.uint8))
    ds = None
    ds = gdal.Translate(str(vrt), str(raster), format="VRT")
    ds = None
    original = util.dataset_checksums(vrt)
    ds = gdal.Open(str(raster), gdal.GA_Update)
    ds.GetRasterBand(1).WriteArray(np.array([[2]], dtype=np.uint8))
    ds = None
    changed = util.dataset_checksums(vrt)
    assert original[str(vrt)] == changed[str(vrt)]
    assert original[str(raster)] != changed[str(raster)]
    assert changed[str(raster)] == file_checksum(raster)
    assert file_checksum(raster) == hashlib.sha256(raster.read_bytes()).hexdigest()
    geometry = {"type": "Point", "coordinates": [1, 2]}
    expected = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    assert inputs.area_checksum(geometry) == expected


def _target_file(path, epsg, with_feature=False):
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = dataset.CreateLayer("targets", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("intervention", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("effectiveness", ogr.OFTReal))
    if with_feature:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(
            ogr.CreateGeometryFromWkt("POLYGON ((0 0, 111319 0, 111319 111325, 0 0))")
        )
        feature.SetField("intervention", " Reverse ")
        feature.SetField("effectiveness", 80)
        layer.CreateFeature(feature)
    dataset = None


def test_copy_targets_inspects_normalizes_and_reprojects(tmp_path):
    source, destination = tmp_path / "source.gpkg", tmp_path / "dest.gpkg"
    _target_file(source, 3857, True)
    _target_file(destination, 4326)
    gdal.UseExceptions()
    info = util.inspect_source(str(source))
    assert info["vector"]["layers"][0]["feature_count"] == 1
    assert inputs.inspect_targets_file(str(source))["effectiveness_range"] == (80, 80)
    inputs.copy_planning_targets(
        source,
        destination,
        {
            "intervention_field": "intervention",
            "effectiveness_field": "effectiveness",
            "effectiveness_units": "percent",
        },
    )
    dataset = ogr.Open(str(destination))
    feature = dataset.GetLayer(0).GetNextFeature()
    assert feature.GetField("intervention") == "reverse"
    assert feature.GetField("effectiveness") == pytest.approx(0.8)
    envelope = feature.GetGeometryRef().GetEnvelope()
    assert envelope == pytest.approx((0, 1, 0, 1), abs=0.001)
    with pytest.raises(ValueError, match="Cannot open spatial input"):
        util.inspect_source(str(tmp_path / "missing.gpkg"))


def test_scenario_comparison_requires_matching_provenance():
    projection = dict(
        scenario_gains_km2=3,
        scenario_avoided_losses_km2=2,
        scenario_degraded_km2=5,
        remaining_shortfall_km2=1,
        gap_closed_pct=80,
    )
    previous = {key: value - 1 for key, value in projection.items()}
    old = {"arr_layer_path": "arr.tif", "input_checksums": {"arr.tif": "unchanged"}}
    params = dict(
        old,
        comparison={
            "job_id": "old",
            "task_name": "old scenario",
            "projection": previous,
            "parameters": old,
        },
    )
    comparison = ldn_planning.compare_scenarios(params, projection)
    assert all(row["difference"] == 1 for row in comparison["metrics"])
    for key, value in (
        ("arr_layer_path", "other"),
        ("status_baseline_type", "3class"),
        ("land_type_band_index", 2),
        ("aoi_checksum", "different"),
        ("input_checksums", {"arr.tif": "changed"}),
    ):
        with pytest.raises(ValueError, match=key):
            ldn_planning.compare_scenarios(dict(params, **{key: value}), projection)


def test_scenario_comparison_uses_geojson_jurisdiction_checksums(tmp_path):
    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"shapeName": "District"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            }
        ],
    }
    old_path = tmp_path / "old_admin_zones.geojson"
    new_path = tmp_path / "new_admin_zones.geojson"
    for path in (old_path, new_path):
        path.write_text(json.dumps(boundary))

    projection = dict.fromkeys(
        (
            "scenario_gains_km2",
            "scenario_avoided_losses_km2",
            "scenario_degraded_km2",
            "remaining_shortfall_km2",
            "gap_closed_pct",
        ),
        0,
    )
    old = {
        "arr_layer_path": "arr.tif",
        "jurisdictions_path": str(old_path),
        "input_checksums": {
            "arr_layer_path": {"arr.tif": "unchanged"},
            "jurisdictions_path": util.dataset_checksums(old_path),
        },
    }
    params = {
        **old,
        "jurisdictions_path": str(new_path),
        "input_checksums": {
            "arr_layer_path": {"arr.tif": "unchanged"},
            "jurisdictions_path": util.dataset_checksums(new_path),
        },
        "comparison": {
            "job_id": "old",
            "task_name": "old scenario",
            "projection": projection,
            "parameters": old,
        },
    }
    original = copy.deepcopy(params)
    old_path.unlink()
    assert (
        ldn_planning.compare_scenarios(params, projection)["previous_job_id"] == "old"
    )
    assert params == original

    boundary["features"][0]["properties"]["shapeName"] = "Changed"
    new_path.write_text(json.dumps(boundary))
    changed = copy.deepcopy(params)
    changed["input_checksums"]["jurisdictions_path"] = util.dataset_checksums(new_path)
    with pytest.raises(ValueError, match="jurisdiction boundary checksums"):
        ldn_planning.compare_scenarios(changed, projection)

    for invalid_checksums in (
        {},
        {"unrelated.geojson": "a" * 64},
        {str(new_path): ""},
        {str(new_path): "x" * 64},
    ):
        unverified = copy.deepcopy(params)
        unverified["input_checksums"]["jurisdictions_path"] = invalid_checksums
        with pytest.raises(ValueError):
            ldn_planning.compare_scenarios(unverified, projection)

    multifile = copy.deepcopy(params)
    multifile["input_checksums"]["jurisdictions_path"]["other-file"] = "a" * 64
    with pytest.raises(ValueError, match="jurisdictions_path"):
        ldn_planning.compare_scenarios(multifile, projection)

    geopackage = copy.deepcopy(params)
    geopackage["jurisdictions_path"] = str(tmp_path / "new.gpkg")
    geopackage["input_checksums"]["jurisdictions_path"] = {
        geopackage["jurisdictions_path"]: "a" * 64
    }
    old_geopackage = geopackage["comparison"]["parameters"]
    old_geopackage["jurisdictions_path"] = str(tmp_path / "old.gpkg")
    old_geopackage["input_checksums"]["jurisdictions_path"] = {
        old_geopackage["jurisdictions_path"]: "a" * 64
    }
    with pytest.raises(ValueError, match="jurisdictions_path"):
        ldn_planning.compare_scenarios(geopackage, projection)

    same_shapefile = copy.deepcopy(params)
    shapefile = str(tmp_path / "districts.shp")
    shapefile_checksums = {
        shapefile: "a" * 64,
        str(tmp_path / "districts.dbf"): "b" * 64,
    }
    same_shapefile["jurisdictions_path"] = shapefile
    same_shapefile["input_checksums"]["jurisdictions_path"] = shapefile_checksums
    old_shapefile = same_shapefile["comparison"]["parameters"]
    old_shapefile["jurisdictions_path"] = shapefile
    old_shapefile["input_checksums"]["jurisdictions_path"] = shapefile_checksums.copy()
    assert ldn_planning.compare_scenarios(same_shapefile, projection)
    same_shapefile["input_checksums"]["jurisdictions_path"][
        str(tmp_path / "districts.dbf")
    ] = "c" * 64
    with pytest.raises(ValueError, match="jurisdiction boundary checksums"):
        ldn_planning.compare_scenarios(same_shapefile, projection)

    changed_arr = copy.deepcopy(params)
    changed_arr["input_checksums"]["arr_layer_path"]["arr.tif"] = "changed"
    with pytest.raises(ValueError, match="input_checksums"):
        ldn_planning.compare_scenarios(changed_arr, projection)

    missing = copy.deepcopy(params)
    missing["comparison"]["parameters"]["jurisdictions_path"] = str(new_path)
    missing["comparison"]["parameters"]["input_checksums"].pop("jurisdictions_path")
    missing["input_checksums"].pop("jurisdictions_path")
    with pytest.raises(ValueError, match="checksum"):
        ldn_planning.compare_scenarios(missing, projection)


def test_reports_preserve_shapes_and_escape_formulas(tmp_path):
    path = tmp_path / "priority"
    summary = {"nested": {"=formula": "=1+1"}}
    params = {"criteria": [{"name": "=2+2"}]}
    result = report.save_priority_reports(path, summary, params)
    assert json.loads(path.with_suffix(".json").read_text()) == result
    workbook = load_workbook(path.with_suffix(".xlsx"))
    assert workbook.sheetnames == ["Planning priorities", "Settings and provenance"]
    assert workbook.worksheets[0]["B2"].value == "=1+1"
    assert workbook.worksheets[0]["B2"].data_type == "s"
    assert workbook.worksheets[1]["B2"].data_type == "s"
    comparison = {
        "metrics": [{"metric": "gain", "previous": 1, "current": 3, "difference": 2}],
        "note": "Planning estimates",
    }
    report.save_ldn_planning_excel(
        tmp_path / "scenario.xlsx", scenario_comparison=comparison
    )
    workbook = load_workbook(tmp_path / "scenario.xlsx")
    assert workbook.sheetnames == ["Scenario comparison"]
    assert list(workbook.active.values)[1] == ("gain", 1, 3, 2)
    extra = {
        "target_provenance": [],
        "input_checksums": {},
        "aoi_checksum": "abc",
        "scenario_comparison": comparison,
    }
    result = report.save_ldn_planning_json(tmp_path / "scenario.json", extra_data=extra)
    assert json.loads((tmp_path / "scenario.json").read_text()) == result
    assert set(result) == {
        "task_name",
        "generated_at",
        "te_algorithms_version",
        "arr",
        "hotspots",
        "bau",
        "scenario",
        "projection",
        *extra,
    }
    with pytest.raises(ValueError):
        report.save_ldn_planning_json(
            tmp_path / "invalid.json", extra_data={"x": float("nan")}
        )


def test_geojson_bounds_rejects_empty_or_incomplete_geometries():
    polygon = {"type": "Polygon", "coordinates": [[[1, 2], [4, 2], [4, 5], [1, 2]]]}
    area = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": polygon, "properties": {}}],
    }
    assert util.geojson_bounds(area) == (1, 2, 4, 5)
    assert util.geojson_bounds(polygon) == (1, 2, 4, 5)
    assert util.geojson_bounds(area["features"][0]) == (1, 2, 4, 5)
    area["features"].append({"type": "Feature", "geometry": None, "properties": {}})
    with pytest.raises(ValueError):
        util.geojson_bounds(area)
    with pytest.raises(ValueError):
        util.geojson_bounds({"type": "FeatureCollection", "features": []})
