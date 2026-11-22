"""Validation and provenance for LDN planning inputs."""

import hashlib
import json
import math


def target_effectiveness(value, units="fraction"):
    if isinstance(value, bool):
        raise ValueError("Effectiveness must be a finite number.")
    number = float(value)
    maximum = 100 if units == "percent" else 1
    if units not in ("fraction", "percent"):
        raise ValueError("Effectiveness units must be fraction or percent.")
    if not math.isfinite(number) or not 0 <= number <= maximum:
        raise ValueError(f"Effectiveness must be between 0 and {maximum}.")
    return number / maximum


def area_checksum(geojson):
    """Hash GeoJSON using the planning scenario provenance format."""
    if geojson is None:
        return None
    geometry = json.dumps(geojson, sort_keys=True, allow_nan=False)
    return hashlib.sha256(geometry.encode("utf-8")).hexdigest()


def polygon_target(wkt, intervention, effectiveness, geometry_crs):
    from osgeo import ogr, osr

    if intervention not in ("avoid", "reduce", "reverse"):
        raise ValueError("Intervention must be avoid, reduce or reverse.")
    geometry = ogr.CreateGeometryFromWkt(wkt)
    if (
        geometry is None
        or geometry.IsEmpty()
        or not geometry.IsValid()
        or ogr.GT_Flatten(geometry.GetGeometryType())
        not in (ogr.wkbPolygon, ogr.wkbMultiPolygon)
    ):
        raise ValueError("Targets must be valid non-empty polygons.")
    reference = osr.SpatialReference()
    if not geometry_crs or reference.SetFromUserInput(geometry_crs) != 0:
        raise ValueError("Target geometry CRS is required.")
    return {
        "wkt_geometry": wkt,
        "geometry_crs": reference.ExportToWkt(),
        "intervention": intervention,
        "effectiveness": target_effectiveness(effectiveness),
    }


def inspect_targets_file(path: str) -> dict:
    """Inspect fields and intervention/effectiveness values in a targets file."""
    from osgeo import ogr

    result = {
        "field_names": [],
        "string_fields": [],
        "numeric_fields": [],
        "feature_count": 0,
        "intervention_counts": {},
        "effectiveness_range": None,
        "invalid_effectiveness_count": 0,
    }
    dataset = ogr.Open(path, 0)
    if dataset is None:
        raise ValueError(f"Cannot open targets file: {path}")
    try:
        layer = dataset.GetLayer(0)
        if layer is None:
            raise ValueError(f"Targets file has no vector layer: {path}")
        definition = layer.GetLayerDefn()
        for index in range(definition.GetFieldCount()):
            field = definition.GetFieldDefn(index)
            name = field.GetName()
            field_type = field.GetType()
            result["field_names"].append(name)
            if field_type in (ogr.OFTString, ogr.OFTWideString):
                result["string_fields"].append(name)
            elif field_type in (ogr.OFTReal, ogr.OFTInteger, ogr.OFTInteger64):
                result["numeric_fields"].append(name)
        result["feature_count"] = layer.GetFeatureCount()
        has_intervention = "intervention" in result["field_names"]
        has_effectiveness = "effectiveness" in result["field_names"]
        effectiveness_values = []
        layer.ResetReading()
        for feature in layer:
            if has_intervention:
                value = feature.GetField("intervention")
                key = (value or "").strip()
                counts = result["intervention_counts"]
                counts[key] = counts.get(key, 0) + 1
            if has_effectiveness:
                value = feature.GetField("effectiveness")
                if value is not None:
                    effectiveness_values.append(float(value))
        if effectiveness_values:
            result["effectiveness_range"] = (
                min(effectiveness_values),
                max(effectiveness_values),
            )
            result["invalid_effectiveness_count"] = sum(
                1 for value in effectiveness_values if value < 0 or value > 100
            )
    finally:
        dataset = None
    return result


def copy_planning_targets(source_path, destination_path, field_config):
    """Normalize target polygons into an existing destination layer and CRS."""
    from osgeo import ogr, osr

    source = ogr.Open(str(source_path), 0)
    if source is None:
        raise OSError(f"Cannot open source file: {source_path}")
    destination = ogr.Open(str(destination_path), 1)
    if destination is None:
        raise OSError(f"Cannot open destination GPKG: {destination_path}")

    try:
        source_layer = source.GetLayer(0)
        destination_layer = destination.GetLayer(0)
        intervention_field = field_config.get("intervention_field")
        default_intervention = field_config.get("default_intervention")
        effectiveness_field = field_config.get("effectiveness_field")
        raw_default = field_config.get("default_effectiveness")
        default_effectiveness = target_effectiveness(
            0.8 if raw_default is None else raw_default
        )
        units = field_config.get("effectiveness_units", "fraction")
        source_srs = source_layer.GetSpatialRef()
        if source_srs is None:
            raise ValueError("Targets must declare a source CRS.")
        source_srs = source_srs.Clone()
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        destination_srs = destination_layer.GetSpatialRef()
        if destination_srs is None:
            raise ValueError("Destination targets layer must declare a CRS.")
        destination_srs = destination_srs.Clone()
        destination_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_srs, destination_srs)

        for feature in source_layer:
            geometry = feature.GetGeometryRef()
            if (
                geometry is None
                or geometry.IsEmpty()
                or not geometry.IsValid()
                or ogr.GT_Flatten(geometry.GetGeometryType())
                not in (ogr.wkbPolygon, ogr.wkbMultiPolygon)
            ):
                raise ValueError(
                    f"Invalid geometry in target feature {feature.GetFID()}."
                )
            if intervention_field is not None:
                raw = feature.GetField(intervention_field)
                intervention = str(raw or "").strip().lower()
            else:
                intervention = default_intervention or "avoid"
            if intervention not in {"avoid", "reduce", "reverse"}:
                raise ValueError(
                    f"Invalid intervention in target feature {feature.GetFID()}: {intervention}"
                )
            raw_effectiveness = (
                feature.GetField(effectiveness_field)
                if effectiveness_field is not None
                else None
            )
            effectiveness = (
                default_effectiveness
                if raw_effectiveness is None
                else target_effectiveness(raw_effectiveness, units)
            )
            output_feature = ogr.Feature(destination_layer.GetLayerDefn())
            output_geometry = geometry.Clone()
            if output_geometry.Transform(transform) != 0:
                raise ValueError("Cannot transform target geometry.")
            output_feature.SetGeometry(output_geometry)
            output_feature.SetField("intervention", intervention)
            output_feature.SetField("effectiveness", effectiveness)
            if destination_layer.CreateFeature(output_feature) != 0:
                raise OSError("Cannot write target feature.")
            output_feature = None
        destination.FlushCache()
    finally:
        source = None
        destination = None
