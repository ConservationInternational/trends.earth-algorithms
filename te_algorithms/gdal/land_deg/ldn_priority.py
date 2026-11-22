"""Blockwise, class-specific LDN land-prioritization algorithms."""

import hashlib
import json
import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

from ...common.checksum import file_checksum
from . import config, ldn_planning

logger = logging.getLogger(__name__)

gdal.UseExceptions()

ARR_CLASSES = {"avoid": 1, "reduce": 2, "reverse": 3}
CLASS_NAMES = {value: key for key, value in ARR_CLASSES.items()}
NODATA = int(config.NODATA_VALUE)
BLOCK_SIZE = 256
_RESAMPLING = {
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "average": gdal.GRA_Average,
}
_METADATA_FIELDS = ("source", "date", "units", "license")


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def validate_criteria(criteria: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize class-specific criterion settings.

    Criteria are weighted independently within each ARR class. Missing fields,
    invalid categories, and weights that do not sum to one are rejected before
    any raster is written.
    """
    if isinstance(criteria, (str, bytes)) or not isinstance(criteria, Sequence):
        raise ValueError("criteria must be a sequence of criterion mappings.")
    if not criteria:
        raise ValueError("At least one criterion is required for prioritization.")
    normalized = []
    weights: Dict[str, float] = {}
    for index, item in enumerate(criteria):
        if not isinstance(item, Mapping):
            raise ValueError(f"Criterion {index} must be a mapping.")
        criterion = dict(item)
        name = str(criterion.get("name") or "").strip()
        path = str(criterion.get("path") or "").strip()
        arr_class = str(criterion.get("arr_class", "")).strip().lower()
        if not name or not path:
            raise ValueError(f"Criterion {index} needs a name and a data path.")
        if arr_class not in ARR_CLASSES:
            raise ValueError(
                f"Criterion {name!r} arr_class must be avoid, reduce, or reverse."
            )
        for field in _METADATA_FIELDS:
            value = criterion.get(field)
            if value is None or not str(value).strip():
                raise ValueError(f"Criterion {name!r} needs {field} metadata.")
        direction = str(criterion.get("direction", "")).strip().lower()
        if direction not in {"higher", "lower"}:
            raise ValueError(
                f"Criterion {name!r} direction must be 'higher' or 'lower'."
            )
        weight = _finite_number(criterion.get("weight"), f"{name} weight")
        if weight < 0:
            raise ValueError(f"Criterion {name!r} weight cannot be negative.")
        criterion["weight"] = weight
        weights[arr_class] = weights.get(arr_class, 0.0) + weight

        band_index = criterion.get("band_index", 1)
        try:
            normalized_band_index = int(band_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Criterion {name!r} band_index must be a positive integer."
            ) from error
        if (
            isinstance(band_index, bool)
            or normalized_band_index < 1
            or float(band_index) != normalized_band_index
        ):
            raise ValueError(
                f"Criterion {name!r} band_index must be a positive integer."
            )
        criterion["band_index"] = normalized_band_index
        criterion["name"] = name
        criterion["path"] = path
        criterion["arr_class"] = arr_class
        criterion["direction"] = direction
        criterion["field"] = str(criterion.get("field") or "").strip()
        resampling = str(criterion.get("resampling", "nearest")).strip().lower()
        if resampling not in _RESAMPLING:
            raise ValueError(
                f"Criterion {name!r} resampling must be nearest, bilinear, or average."
            )
        criterion["resampling"] = resampling

        categories = criterion.get("categories")
        if categories:
            if not isinstance(categories, Mapping):
                raise ValueError(f"Criterion {name!r} categories must be a mapping.")
            if resampling != "nearest":
                raise ValueError(
                    f"Categorical criterion {name!r} must use nearest resampling."
                )
            normalized_categories = {}
            for category, score in categories.items():
                _finite_number(category, f"{name} category")
                score = _finite_number(score, f"{name} category score")
                if not 0 <= score <= 1:
                    raise ValueError(
                        f"Criterion {name!r} category scores must be in [0, 1]."
                    )
                normalized_categories[_category_key(category)] = score
            criterion["categories"] = normalized_categories
            criterion["minimum"] = None
            criterion["maximum"] = None
        else:
            minimum = _finite_number(criterion.get("minimum"), f"{name} minimum")
            maximum = _finite_number(criterion.get("maximum"), f"{name} maximum")
            if maximum <= minimum:
                raise ValueError(
                    f"Criterion {name!r} maximum must be greater than minimum."
                )
            criterion["minimum"] = minimum
            criterion["maximum"] = maximum
            criterion["categories"] = None
        normalized.append(criterion)

    for arr_class, total in weights.items():
        if not math.isfinite(total) or total <= 0:
            raise ValueError(
                f"Weights for {arr_class} criteria must have a positive total."
            )
    for criterion in normalized:
        criterion["weight"] /= weights[criterion["arr_class"]]
    return normalized


def _category_key(value: Any) -> str:
    number = _finite_number(value, "category")
    return str(int(number)) if number.is_integer() else str(number)


def _sha256(path: str) -> str:
    dataset = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_VECTOR)
    if dataset is None:
        datasource = ogr.Open(str(path), 0)
        if datasource is None:
            raise RuntimeError(f"Cannot open data source for provenance: {path}")
        dataset = datasource
    try:
        file_list = dataset.GetFileList() or [str(path)]
    finally:
        dataset = None

    dataset_path = os.path.abspath(path)
    dataset_directory = os.path.dirname(dataset_path)
    files = set()
    for file_path in file_list:
        if not os.path.isabs(file_path):
            file_path = os.path.join(dataset_directory, file_path)
        files.add(os.path.abspath(file_path))
    files.add(dataset_path)
    files = sorted(files)

    file_digests = []
    for file_path in files:
        file_digests.append((file_path, bytes.fromhex(file_checksum(file_path))))
    if len(file_digests) == 1:
        return file_digests[0][1].hex()

    manifest = hashlib.sha256()
    for file_path, file_digest in file_digests:
        manifest.update(os.path.relpath(file_path, dataset_directory).encode())
        manifest.update(b"\0")
        manifest.update(file_digest)
    return manifest.hexdigest()


def _valid_values(values: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.isfinite(values) & (values != NODATA)
    if nodata is not None:
        if math.isnan(nodata):
            return valid
        valid &= values != nodata
    return valid


def _prepare_vector(
    path: str,
    target_projection: str,
    field: Optional[str] = None,
    use_fid: bool = False,
) -> Tuple[List[Tuple[ogr.Geometry, float, str]], str]:
    datasource = ogr.Open(path, 0)
    if datasource is None:
        raise RuntimeError(f"Cannot open vector source: {path}")
    layer = datasource.GetLayer(0)
    source_srs = layer.GetSpatialRef()
    if source_srs is None:
        raise ValueError(f"Vector source {path} has no CRS.")
    source_srs = source_srs.Clone()
    target_srs = osr.SpatialReference()
    if not target_projection or target_srs.ImportFromWkt(target_projection) != 0:
        raise ValueError("Reference ARR raster has no valid CRS.")
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = (
        None
        if source_srs.IsSame(target_srs)
        else osr.CoordinateTransformation(source_srs, target_srs)
    )
    definition = layer.GetLayerDefn()
    field_index = definition.GetFieldIndex(field) if field else -1
    if field and field_index < 0:
        raise ValueError(f"Vector source {path} has no field named {field!r}.")
    name_index = next(
        (
            definition.GetFieldIndex(candidate)
            for candidate in ("name", "NAME", "zone_name", "NAME_1")
            if definition.GetFieldIndex(candidate) >= 0
        ),
        -1,
    )
    features = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty() or not geometry.IsValid():
            raise ValueError(f"Vector source {path} contains an invalid geometry.")
        if field:
            if not feature.IsFieldSetAndNotNull(field_index):
                continue
            value = _finite_number(feature.GetField(field_index), f"{field} field")
        elif use_fid:
            value = float(feature.GetFID())
        else:
            value = 1.0
        label = (
            str(feature.GetField(name_index))
            if name_index >= 0 and feature.IsFieldSetAndNotNull(name_index)
            else str(int(value))
        )
        geometry = geometry.Clone()
        if transform is not None and geometry.Transform(transform) != 0:
            raise ValueError(f"Could not transform vector feature from {path}.")
        features.append((geometry, value, label))
    datasource = None
    return features, str(field or "")


def _rasterize_vector_block(
    features: Sequence[Tuple[ogr.Geometry, float, str]],
    gt: Tuple,
    projection: str,
    x_off: int,
    y_off: int,
    x_size: int,
    y_size: int,
    nodata: float,
) -> np.ndarray:
    block_gt = (
        gt[0] + x_off * gt[1],
        gt[1],
        gt[2],
        gt[3] + y_off * gt[5],
        gt[4],
        gt[5],
    )
    dataset = gdal.GetDriverByName("MEM").Create(
        "", x_size, y_size, 1, gdal.GDT_Float64
    )
    dataset.SetGeoTransform(block_gt)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.Fill(nodata)
    vector_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    layer = vector_ds.CreateLayer("criteria", srs=srs, geom_type=ogr.wkbUnknown)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTReal))
    definition = layer.GetLayerDefn()
    xmin = block_gt[0]
    xmax = block_gt[0] + x_size * block_gt[1]
    ymax = block_gt[3]
    ymin = block_gt[3] + y_size * block_gt[5]
    for geometry, value, _ in features:
        envelope = geometry.GetEnvelope()
        if (
            envelope[1] < xmin
            or envelope[0] > xmax
            or envelope[3] < ymin
            or envelope[2] > ymax
        ):
            continue
        feature = ogr.Feature(definition)
        feature.SetField("value", value)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
    if gdal.RasterizeLayer(dataset, [1], layer, options=["ATTRIBUTE=value"]) != 0:
        raise RuntimeError("Could not rasterize vector criterion block.")
    result = band.ReadAsArray()
    dataset = None
    vector_ds = None
    return result


def _create_multiband_raster(
    path: str,
    reference: gdal.Dataset,
    bands: int,
    dtype: int,
    nodata: float,
    descriptions: Sequence[str],
) -> gdal.Dataset:
    output = gdal.GetDriverByName("GTiff").Create(
        path,
        reference.RasterXSize,
        reference.RasterYSize,
        bands,
        dtype,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    for index, description in enumerate(descriptions, start=1):
        band = output.GetRasterBand(index)
        band.SetNoDataValue(nodata)
        band.SetDescription(description)
        band.Fill(nodata)
    return output


def _metadata_criterion(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "name",
            "path",
            "field",
            "arr_class",
            "band_index",
            "source",
            "date",
            "units",
            "license",
            "weight",
            "direction",
            "minimum",
            "maximum",
            "categories",
            "resampling",
        )
    }


def _aoi_checksum(aoi) -> Optional[str]:
    if aoi is None:
        return None
    geojson = aoi.get_geojson() if hasattr(aoi, "get_geojson") else aoi
    encoded = json.dumps(geojson, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_prioritization(
    arr_path: str,
    criteria: Sequence[Mapping[str, Any]],
    output_path: str,
    arr_band_index: int = 1,
    exclusion_path: Optional[str] = None,
    land_type_path: Optional[str] = None,
    zones_path: Optional[str] = None,
    aoi=None,
    progress_callback=None,
    killed_callback=None,
) -> Tuple[Dict[str, Any], str]:
    """Compute complete-case weighted scores independently within each ARR class.

    The output contains three Float32 score bands (Avoid, Reduce, Reverse) and
    a companion three-band Byte coverage raster. Missing inputs never become
    zero; any missing required criterion leaves that class pixel unscored.
    """
    normalized = validate_criteria(criteria)
    if not os.path.isdir(os.path.dirname(os.path.abspath(output_path))):
        raise ValueError("Output directory does not exist.")
    coverage_path = str(
        Path(output_path).with_name(f"{Path(output_path).stem}_coverage.tif")
    )
    if os.path.abspath(coverage_path) == os.path.abspath(output_path):
        raise ValueError("Score and coverage outputs must have different paths.")

    arr_ds = gdal.Open(str(arr_path))
    if arr_ds is None:
        raise RuntimeError(f"Cannot open ARR raster: {arr_path}")
    arr_band = arr_ds.GetRasterBand(arr_band_index)
    if arr_band is None:
        raise RuntimeError(f"ARR band {arr_band_index} not found.")
    gt = arr_ds.GetGeoTransform()
    projection = arr_ds.GetProjection()
    if not projection or gt[2] != 0 or gt[4] != 0:
        raise ValueError("ARR reference raster needs a north-up grid and valid CRS.")
    xsize, ysize = arr_ds.RasterXSize, arr_ds.RasterYSize
    arr_nodata = arr_band.GetNoDataValue()
    if arr_nodata is None:
        arr_nodata = NODATA

    criterion_sources = []
    source_for_criterion = {}
    vector_criteria = {}
    criteria_by_class: Dict[int, List[Dict[str, Any]]] = {
        value: [] for value in ARR_CLASSES.values()
    }
    for criterion in normalized:
        class_code = ARR_CLASSES[criterion["arr_class"]]
        criteria_by_class[class_code].append(criterion)
        if criterion["field"]:
            features, _ = _prepare_vector(
                criterion["path"],
                projection,
                field=criterion["field"],
            )
            vector_criteria[id(criterion)] = features
            source_for_criterion[id(criterion)] = None
        else:
            raster_ds, raster_band = ldn_planning._aligned_band(
                criterion["path"],
                criterion["band_index"],
                arr_ds,
                resample_alg=_RESAMPLING[criterion["resampling"]],
            )
            source_for_criterion[id(criterion)] = (raster_ds, raster_band)
            criterion_sources.append(raster_ds)

    exclusion_ds = exclusion_band = None
    if exclusion_path:
        exclusion_ds, exclusion_band = ldn_planning._aligned_band(
            exclusion_path, 1, arr_ds
        )
    land_ds = land_band = None
    if land_type_path:
        land_ds, land_band = ldn_planning._aligned_band(land_type_path, 1, arr_ds)
    zone_features = zone_labels = None
    if zones_path:
        zone_features, _ = _prepare_vector(zones_path, projection, use_fid=True)
        zone_labels = {int(value): label for _, value, label in zone_features}
    aoi_geometries = ldn_planning._prepare_aoi_geometries(None, projection, aoi=aoi)

    score_ds = _create_multiband_raster(
        output_path,
        arr_ds,
        3,
        gdal.GDT_Float32,
        float(NODATA),
        [f"{CLASS_NAMES[code]} priority score" for code in (1, 2, 3)],
    )
    coverage_ds = _create_multiband_raster(
        coverage_path,
        arr_ds,
        3,
        gdal.GDT_Byte,
        255,
        [f"{CLASS_NAMES[code]} complete-case coverage" for code in (1, 2, 3)],
    )
    for band_index in range(1, 4):
        coverage_ds.GetRasterBand(band_index).Fill(0)

    class_acc = {
        code: {
            "arr_area": 0.0,
            "excluded_area": 0.0,
            "exclusion_mask_nodata_area": 0.0,
            "eligible_area": 0.0,
            "scored_area": 0.0,
            "score_area_sum": 0.0,
            "score_min": None,
            "score_max": None,
            "pixels": 0,
            "scored_pixels": 0,
        }
        for code in ARR_CLASSES.values()
    }
    criterion_acc = {
        id(criterion): {
            "eligible_area": 0.0,
            "valid_area": 0.0,
            "valid_score_area_sum": 0.0,
            "complete_area": 0.0,
            "weighted_contribution_area_sum": 0.0,
            "pixels": 0,
        }
        for criterion in normalized
    }
    land_acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    zone_acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    block_rows = 0
    total_blocks = math.ceil(ysize / BLOCK_SIZE)
    cancelled = False

    try:
        for y_off in range(0, ysize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                cancelled = True
                break
            y_size = min(BLOCK_SIZE, ysize - y_off)
            arr_values = arr_band.ReadAsArray(0, y_off, xsize, y_size)
            arr_valid = _valid_values(arr_values, arr_nodata)
            valid_class = np.isin(arr_values, list(ARR_CLASSES.values()))
            if np.any(arr_valid & ~valid_class):
                invalid = np.unique(arr_values[arr_valid & ~valid_class])
                raise ValueError(f"ARR raster contains invalid class values {invalid}.")
            aoi_mask = ldn_planning._rasterize_aoi_block(
                aoi_geometries,
                gt,
                projection,
                0,
                y_off,
                xsize,
                y_size,
            )
            if aoi_mask is None:
                aoi_mask = np.ones(arr_values.shape, dtype=bool)
            exclusion_values = exclusion_valid = None
            if exclusion_band is not None:
                exclusion_values = exclusion_band.ReadAsArray(0, y_off, xsize, y_size)
                exclusion_valid = _valid_values(
                    exclusion_values, exclusion_band.GetNoDataValue()
                )
            base = arr_valid & valid_class & aoi_mask
            excluded = np.zeros(base.shape, dtype=bool)
            exclusion_unknown = np.zeros(base.shape, dtype=bool)
            if exclusion_band is not None:
                exclusion_unknown = base & ~exclusion_valid
                excluded = base & (~exclusion_valid | (exclusion_values > 0))
            eligible = base & ~excluded
            areas = ldn_planning._compute_pixel_areas_km2(
                gt, y_off, y_size, xsize, projection
            )

            land_values = land_valid = None
            if land_band is not None:
                land_values = land_band.ReadAsArray(0, y_off, xsize, y_size)
                land_valid = _valid_values(land_values, land_band.GetNoDataValue())
            zone_values = (
                _rasterize_vector_block(
                    zone_features,
                    gt,
                    projection,
                    0,
                    y_off,
                    xsize,
                    y_size,
                    NODATA,
                )
                if zone_features is not None
                else None
            )

            normalized_values = {}
            criterion_validity = {}
            for criterion in normalized:
                if criterion["field"]:
                    raw = _rasterize_vector_block(
                        vector_criteria[id(criterion)],
                        gt,
                        projection,
                        0,
                        y_off,
                        xsize,
                        y_size,
                        NODATA,
                    )
                    nodata = NODATA
                else:
                    _, source_band = source_for_criterion[id(criterion)]
                    raw = source_band.ReadAsArray(0, y_off, xsize, y_size)
                    nodata = source_band.GetNoDataValue()
                is_valid = _valid_values(raw, nodata)
                if criterion["categories"]:
                    normalized_array = np.full(raw.shape, np.nan, dtype=np.float64)
                    for category, score in criterion["categories"].items():
                        normalized_array[is_valid & (raw == float(category))] = score
                    is_valid &= np.isfinite(normalized_array)
                else:
                    normalized_array = np.clip(
                        (raw.astype(np.float64) - criterion["minimum"])
                        / (criterion["maximum"] - criterion["minimum"]),
                        0.0,
                        1.0,
                    )
                if criterion["direction"] == "lower":
                    normalized_array = 1.0 - normalized_array
                normalized_values[id(criterion)] = normalized_array
                criterion_validity[id(criterion)] = is_valid

            score_blocks = {}
            for class_code, class_criteria in criteria_by_class.items():
                class_mask = eligible & (arr_values == class_code)
                acc = class_acc[class_code]
                acc["arr_area"] += float(
                    np.sum(areas[base & (arr_values == class_code)])
                )
                acc["excluded_area"] += float(
                    np.sum(areas[excluded & (arr_values == class_code)])
                )
                acc["exclusion_mask_nodata_area"] += float(
                    np.sum(areas[exclusion_unknown & (arr_values == class_code)])
                )
                acc["eligible_area"] += float(np.sum(areas[class_mask]))
                acc["pixels"] += int(np.count_nonzero(class_mask))
                if not class_criteria:
                    continue
                complete = class_mask.copy()
                weighted = np.zeros(arr_values.shape, dtype=np.float64)
                for criterion in class_criteria:
                    ident = id(criterion)
                    values = normalized_values[ident]
                    is_valid = criterion_validity[ident]
                    stats = criterion_acc[ident]
                    stats["eligible_area"] += float(np.sum(areas[class_mask]))
                    stats["valid_area"] += float(np.sum(areas[class_mask & is_valid]))
                    stats["valid_score_area_sum"] += float(
                        np.sum(
                            values[class_mask & is_valid] * areas[class_mask & is_valid]
                        )
                    )
                    stats["pixels"] += int(np.count_nonzero(class_mask & is_valid))
                    if criterion["weight"] > 0:
                        complete &= is_valid
                        weighted += np.where(
                            is_valid, values * criterion["weight"], 0.0
                        )
                acc["scored_area"] += float(np.sum(areas[complete]))
                acc["score_area_sum"] += float(
                    np.sum(weighted[complete] * areas[complete])
                )
                acc["scored_pixels"] += int(np.count_nonzero(complete))
                complete_area = float(np.sum(areas[complete]))
                for criterion in class_criteria:
                    stats = criterion_acc[id(criterion)]
                    stats["complete_area"] += complete_area
                    if criterion["weight"] > 0:
                        values = normalized_values[id(criterion)]
                        stats["weighted_contribution_area_sum"] += float(
                            np.sum(
                                values[complete] * criterion["weight"] * areas[complete]
                            )
                        )
                if np.any(complete):
                    scored_values = weighted[complete]
                    score_min = float(np.min(scored_values))
                    score_max = float(np.max(scored_values))
                    acc["score_min"] = (
                        score_min
                        if acc["score_min"] is None
                        else min(acc["score_min"], score_min)
                    )
                    acc["score_max"] = (
                        score_max
                        if acc["score_max"] is None
                        else max(acc["score_max"], score_max)
                    )
                score_blocks[class_code] = (complete, weighted)

                if land_values is not None and land_valid is not None:
                    for land_type in np.unique(land_values[class_mask & land_valid]):
                        land_type = int(land_type)
                        group = class_mask & land_valid & (land_values == land_type)
                        group_key = (class_code, land_type)
                        stats = land_acc.setdefault(
                            group_key,
                            {"eligible": 0.0, "scored": 0.0, "score_sum": 0.0},
                        )
                        stats["eligible"] += float(np.sum(areas[group]))
                        scored_group = group & complete
                        stats["scored"] += float(np.sum(areas[scored_group]))
                        stats["score_sum"] += float(
                            np.sum(weighted[scored_group] * areas[scored_group])
                        )
                if zone_values is not None:
                    for zone_id in np.unique(zone_values[class_mask]):
                        if zone_id == NODATA:
                            continue
                        zone_id = int(zone_id)
                        group = class_mask & (zone_values == zone_id)
                        group_key = (class_code, zone_id)
                        stats = zone_acc.setdefault(
                            group_key,
                            {"eligible": 0.0, "scored": 0.0, "score_sum": 0.0},
                        )
                        stats["eligible"] += float(np.sum(areas[group]))
                        scored_group = group & complete
                        stats["scored"] += float(np.sum(areas[scored_group]))
                        stats["score_sum"] += float(
                            np.sum(weighted[scored_group] * areas[scored_group])
                        )

            for class_code in ARR_CLASSES.values():
                score_band = score_ds.GetRasterBand(class_code)
                coverage_band = coverage_ds.GetRasterBand(class_code)
                class_criteria = criteria_by_class[class_code]
                score_array = np.full(arr_values.shape, NODATA, dtype=np.float32)
                coverage_array = np.full(arr_values.shape, 255, dtype=np.uint8)
                class_mask = eligible & (arr_values == class_code)
                coverage_array[class_mask] = 0
                coverage_array[excluded & (arr_values == class_code)] = 2
                if class_criteria:
                    complete, weighted = score_blocks[class_code]
                    score_array[complete] = weighted[complete].astype(np.float32)
                    coverage_array[complete] = 1
                score_band.WriteArray(score_array, 0, y_off)
                coverage_band.WriteArray(coverage_array, 0, y_off)

            block_rows += 1
            if progress_callback:
                progress_callback(100.0 * block_rows / total_blocks)

        if killed_callback and killed_callback():
            cancelled = True
        if cancelled:
            raise RuntimeError(
                "Prioritization cancelled; partial outputs were removed."
            )
        metadata = {
            "criteria": [
                {
                    **_metadata_criterion(item),
                    "sha256": _sha256(item["path"]),
                }
                for item in normalized
            ],
            "arr_path": arr_path,
            "arr_sha256": _sha256(arr_path),
            "exclusion_path": exclusion_path,
            "exclusion_sha256": (_sha256(exclusion_path) if exclusion_path else None),
            "land_type_path": land_type_path,
            "land_type_sha256": (_sha256(land_type_path) if land_type_path else None),
            "zones_path": zones_path,
            "zones_sha256": _sha256(zones_path) if zones_path else None,
            "aoi_sha256": _aoi_checksum(aoi),
            "gdal_version": gdal.VersionInfo("--version"),
            "scoring_policy": "complete_case_weighted_mean",
        }
        for criterion in normalized:
            stats = criterion_acc[id(criterion)]
            if (
                criterion["weight"] > 0
                and stats["eligible_area"]
                and not stats["valid_area"]
            ):
                raise ValueError(
                    f"Criterion {criterion['name']!r} has no valid pixels in its "
                    "eligible analysis area."
                )
        score_ds.SetMetadataItem("PRIORITIZATION_METADATA", json.dumps(metadata))
        coverage_ds.SetMetadataItem("PRIORITIZATION_METADATA", json.dumps(metadata))
        score_ds.FlushCache()
        coverage_ds.FlushCache()
    except Exception:
        score_ds = None
        coverage_ds = None
        for path in (output_path, coverage_path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise
    finally:
        score_ds = None
        coverage_ds = None
        arr_ds = None
        if exclusion_ds is not None:
            exclusion_ds = None
        if land_ds is not None:
            land_ds = None
        source_for_criterion.clear()
        criterion_sources.clear()

    class_summary = {}
    for class_code, acc in class_acc.items():
        scored_area = acc["scored_area"]
        class_summary[CLASS_NAMES[class_code]] = {
            "arr_area_km2": acc["arr_area"],
            "excluded_area_km2": acc["excluded_area"],
            "exclusion_mask_nodata_area_km2": acc["exclusion_mask_nodata_area"],
            "eligible_area_km2": acc["eligible_area"],
            "scored_area_km2": scored_area,
            "coverage_fraction": (
                scored_area / acc["eligible_area"] if acc["eligible_area"] else 0.0
            ),
            "mean_score_area_weighted": (
                acc["score_area_sum"] / scored_area if scored_area else None
            ),
            "score_min": acc["score_min"],
            "score_max": acc["score_max"],
            "eligible_pixels": acc["pixels"],
            "scored_pixels": acc["scored_pixels"],
        }
    criterion_summary = []
    for criterion in normalized:
        stats = criterion_acc[id(criterion)]
        criterion_summary.append(
            {
                **_metadata_criterion(criterion),
                "eligible_area_km2": stats["eligible_area"],
                "valid_area_km2": stats["valid_area"],
                "mean_normalized_score_area_weighted": (
                    stats["valid_score_area_sum"] / stats["valid_area"]
                    if stats["valid_area"]
                    else None
                ),
                "mean_weighted_contribution_area_weighted": (
                    stats["weighted_contribution_area_sum"] / stats["complete_area"]
                    if stats["complete_area"]
                    else None
                ),
                "coverage_fraction": (
                    stats["valid_area"] / stats["eligible_area"]
                    if stats["eligible_area"]
                    else 0.0
                ),
                "valid_pixels": stats["pixels"],
            }
        )

    def grouped_summary(accumulator, id_key, label_map=None):
        result = []
        for (class_code, group_id), stats in sorted(accumulator.items()):
            area = stats["scored"]
            result.append(
                {
                    "arr_class": CLASS_NAMES[class_code],
                    id_key: group_id,
                    "name": (
                        label_map.get(group_id, str(group_id))
                        if label_map
                        else str(group_id)
                    ),
                    "eligible_area_km2": stats["eligible"],
                    "scored_area_km2": area,
                    "coverage_fraction": (
                        area / stats["eligible"] if stats["eligible"] else 0.0
                    ),
                    "mean_score_area_weighted": (
                        stats["score_sum"] / area if area else None
                    ),
                }
            )
        return result

    summary = {
        "score_path": output_path,
        "coverage_path": coverage_path,
        "score_bands": {CLASS_NAMES[code]: code for code in (1, 2, 3)},
        "coverage_bands": {CLASS_NAMES[code]: code for code in (1, 2, 3)},
        "coverage_codes": {
            0: "eligible but unscored (incomplete required criteria)",
            1: "complete-case score available",
            2: "excluded by hard exclusion mask",
            255: "outside ARR/AOI or NoData",
        },
        "scoring_policy": "complete_case_weighted_mean",
        "classes": class_summary,
        "criteria": criterion_summary,
        "by_land_type": grouped_summary(land_acc, "land_type_id"),
        "by_jurisdiction": grouped_summary(zone_acc, "jurisdiction_id", zone_labels),
        "provenance": metadata,
    }
    return summary, output_path


def select_priority_areas(
    score_path: str,
    output_path: str,
    selections: Mapping[str, Mapping[str, Any]],
    progress_callback=None,
    killed_callback=None,
) -> Tuple[Dict[str, Any], str]:
    """Select score-threshold or deterministic top-area candidates by ARR class."""
    if not isinstance(selections, Mapping) or not selections:
        raise ValueError("selections must map ARR classes to selection settings.")
    if not os.path.isdir(os.path.dirname(os.path.abspath(output_path))):
        raise ValueError("Output directory does not exist.")
    normalized = {}
    for class_name, setting in selections.items():
        class_name = str(class_name).lower()
        if class_name not in ARR_CLASSES or not isinstance(setting, Mapping):
            raise ValueError(f"Invalid selection settings for {class_name!r}.")
        mode = str(setting.get("mode", "")).lower()
        if mode in {"area", "top_area", "top-area"}:
            area = _finite_number(
                setting.get("area_km2", setting.get("area")),
                f"{class_name} target area",
            )
            if area < 0:
                raise ValueError("Selection area cannot be negative.")
            normalized[class_name] = {"mode": "area", "area_km2": area}
        elif mode == "threshold":
            threshold = _finite_number(
                setting.get("threshold"), f"{class_name} score threshold"
            )
            if not 0 <= threshold <= 1:
                raise ValueError("Score threshold must be in [0, 1].")
            normalized[class_name] = {"mode": mode, "threshold": threshold}
        else:
            raise ValueError(
                f"Selection mode for {class_name!r} must be 'threshold' or 'area'."
            )

    score_ds = gdal.Open(str(score_path))
    if score_ds is None or score_ds.RasterCount < 3:
        raise RuntimeError("Score raster must contain three ARR-class bands.")
    gt = score_ds.GetGeoTransform()
    projection = score_ds.GetProjection()
    xsize, ysize = score_ds.RasterXSize, score_ds.RasterYSize
    score_bands = [score_ds.GetRasterBand(code) for code in (1, 2, 3)]
    if os.path.abspath(output_path) == os.path.abspath(score_path):
        raise ValueError("Selection output must not overwrite the score raster.")
    selection_db = f"{output_path}.selection.sqlite"
    try:
        os.remove(selection_db)
    except OSError:
        pass
    connection = sqlite3.connect(selection_db)
    connection.execute(
        "CREATE TABLE candidates "
        "(class INTEGER, score REAL, row INTEGER, col INTEGER, area REAL)"
    )
    connection.execute(
        "CREATE INDEX candidate_order ON candidates(class, score DESC, row, col)"
    )
    connection.execute(
        "CREATE TABLE selected (class INTEGER, row INTEGER, col INTEGER, "
        "PRIMARY KEY(class, row, col))"
    )
    score_nd = score_bands[0].GetNoDataValue()
    if score_nd is None:
        score_nd = NODATA
    output = None
    try:
        for y_off in range(0, ysize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                raise RuntimeError("Area selection cancelled.")
            y_size = min(BLOCK_SIZE, ysize - y_off)
            areas = ldn_planning._compute_pixel_areas_km2(
                gt, y_off, y_size, xsize, projection
            )
            rows = []
            for class_code, band in enumerate(score_bands, start=1):
                values = band.ReadAsArray(0, y_off, xsize, y_size)
                band_nodata = band.GetNoDataValue()
                if band_nodata is None:
                    band_nodata = score_nd
                valid = _valid_values(values, band_nodata)
                if not np.any(valid):
                    continue
                row_indices, col_indices = np.where(valid)
                rows.extend(
                    (
                        class_code,
                        float(values[row, col]),
                        y_off + int(row),
                        int(col),
                        float(areas[row, col]),
                    )
                    for row, col in zip(row_indices, col_indices, strict=False)
                )
            connection.executemany(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", rows
            )
            connection.commit()
            if progress_callback:
                progress_callback(50.0 * (y_off + y_size) / ysize)

        class_summaries = {}
        for class_name, setting in normalized.items():
            class_code = ARR_CLASSES[class_name]
            if setting["mode"] == "threshold":
                threshold = setting["threshold"]
                connection.execute(
                    "INSERT INTO selected SELECT class, row, col FROM candidates "
                    "WHERE class=? AND score>=?",
                    (class_code, threshold),
                )
                actual_area, selected_pixels = connection.execute(
                    "SELECT COALESCE(SUM(area),0), COUNT(*) FROM candidates "
                    "WHERE class=? AND score>=?",
                    (class_code, threshold),
                ).fetchone()
                class_summaries[class_name] = {
                    **setting,
                    "selected_area_km2": float(actual_area),
                    "selected_pixels": int(selected_pixels),
                }
            else:
                target_area = setting["area_km2"]
                selected_area = 0.0
                selected_pixels = 0
                cursor = connection.execute(
                    "SELECT row, col, area FROM candidates WHERE class=? "
                    "ORDER BY score DESC, row ASC, col ASC",
                    (class_code,),
                )
                pending = []
                for row, col, area in cursor:
                    remaining_area = target_area - selected_area
                    if remaining_area <= 0 or area > remaining_area:
                        continue
                    next_area = selected_area + area
                    if next_area > target_area:
                        continue
                    pending.append((class_code, row, col))
                    selected_area = next_area
                    selected_pixels += 1
                    if len(pending) >= 10000:
                        connection.executemany(
                            "INSERT INTO selected VALUES (?, ?, ?)", pending
                        )
                        pending.clear()
                if pending:
                    connection.executemany(
                        "INSERT INTO selected VALUES (?, ?, ?)", pending
                    )
                class_summaries[class_name] = {
                    **setting,
                    "selected_area_km2": selected_area,
                    "selected_pixels": selected_pixels,
                    "unspent_area_km2": max(0.0, target_area - selected_area),
                }
        connection.commit()

        score_sha256 = _sha256(score_path)
        output = gdal.GetDriverByName("GTiff").Create(
            output_path,
            xsize,
            ysize,
            1,
            gdal.GDT_Int16,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        output.SetGeoTransform(gt)
        output.SetProjection(projection)
        output_band = output.GetRasterBand(1)
        output_band.SetNoDataValue(NODATA)
        output_band.SetDescription("Selected ARR class code (0=not selected)")
        output_band.Fill(NODATA)
        for y_off in range(0, ysize, BLOCK_SIZE):
            if killed_callback and killed_callback():
                raise RuntimeError("Area selection cancelled.")
            y_size = min(BLOCK_SIZE, ysize - y_off)
            selected = np.zeros((y_size, xsize), dtype=np.int16)
            valid = np.zeros((y_size, xsize), dtype=bool)
            for class_code, band in enumerate(score_bands, start=1):
                values = band.ReadAsArray(0, y_off, xsize, y_size)
                band_nodata = band.GetNoDataValue()
                if band_nodata is None:
                    band_nodata = score_nd
                valid |= _valid_values(values, band_nodata)
            selected[~valid] = NODATA
            cursor = connection.execute(
                "SELECT class, row, col FROM selected WHERE row>=? AND row<?",
                (y_off, y_off + y_size),
            )
            for class_code, row, col in cursor:
                selected[row - y_off, col] = class_code
            output_band.WriteArray(selected, 0, y_off)
            if progress_callback:
                progress_callback(50.0 + 50.0 * (y_off + y_size) / ysize)
        output.SetMetadataItem(
            "SELECTION_METADATA",
            json.dumps(
                {
                    "score_path": score_path,
                    "score_sha256": score_sha256,
                    "selections": class_summaries,
                    "tie_rule": "score descending, then row-major pixel order",
                    "selection_policy": (
                        "Top-area adds ranked whole pixels only when they fit the "
                        "remaining area cap; non-fitting pixels are skipped."
                    ),
                }
            ),
        )
        output.FlushCache()
        output = None
        if killed_callback and killed_callback():
            raise RuntimeError("Area selection cancelled.")
    except Exception:
        connection.close()
        output = None
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    finally:
        connection.close()
        try:
            os.remove(selection_db)
        except OSError:
            pass
        score_ds = None

    summary = {
        "selection_path": output_path,
        "score_path": score_path,
        "score_sha256": score_sha256,
        "selections": class_summaries,
        "tie_rule": "score descending, then row-major pixel order",
        "selection_policy": (
            "Top-area adds ranked whole pixels only when they fit the remaining "
            "physical-area cap; non-fitting pixels are skipped."
        ),
    }
    return summary, output_path
