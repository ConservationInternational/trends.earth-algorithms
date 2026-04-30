"""Pure-Python band name sanitization — no GDAL dependency.

Kept in a standalone module so it can be imported by both
``te_algorithms.gdal.util`` (which requires GDAL/osgeo) and
``te_algorithms.gee.util`` (which must run in GDAL-free GEE environments).
"""

import re

_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]+")


def generate_sanitized_band_names(bands):
    """Return a sanitized, unique name for every band in *bands*.

    Year and type metadata are embedded in the name so that bands covering
    different time periods or population sub-groups get distinct labels.
    """
    sanitized_names = []
    used_names = set()

    for idx, band in enumerate(bands, start=1):
        raw_name = band.name or f"band_{idx}"
        sanitized = _SANITIZE_PATTERN.sub("_", raw_name).strip("_")

        if not sanitized:
            sanitized = f"band_{idx}"

        if sanitized[0].isdigit():
            sanitized = f"b_{sanitized}"

        year = band.metadata.get("year")
        year_initial = band.metadata.get("year_initial")
        year_final = band.metadata.get("year_final")
        reporting_year_initial = band.metadata.get("reporting_year_initial")
        reporting_year_final = band.metadata.get("reporting_year_final")
        year_bl_start = band.metadata.get("year_bl_start")
        year_bl_end = band.metadata.get("year_bl_end")
        year_tg_start = band.metadata.get("year_tg_start")
        year_tg_end = band.metadata.get("year_tg_end")
        deg_year_initial = band.metadata.get("deg_year_initial")
        deg_year_final = band.metadata.get("deg_year_final")
        band_type = band.metadata.get("type")

        if year is not None:
            sanitized = f"{sanitized}_{year}"
        elif year_initial is not None and year_final is not None:
            sanitized = f"{sanitized}_{year_initial}_{year_final}"
        elif reporting_year_initial is not None and reporting_year_final is not None:
            sanitized = f"{sanitized}_{reporting_year_initial}_{reporting_year_final}"
        elif (
            year_bl_start is not None
            and year_bl_end is not None
            and year_tg_start is not None
            and year_tg_end is not None
        ):
            sanitized = (
                f"{sanitized}_{year_bl_start}_{year_bl_end}"
                f"_{year_tg_start}_{year_tg_end}"
            )
        elif deg_year_initial is not None and deg_year_final is not None:
            sanitized = f"{sanitized}_{deg_year_initial}_{deg_year_final}"

        if band_type is not None:
            sanitized = (
                f"{sanitized}_{_SANITIZE_PATTERN.sub('_', str(band_type)).strip('_')}"
            )

        base_name = sanitized
        suffix = 2
        while sanitized in used_names:
            sanitized = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(sanitized)
        sanitized_names.append(sanitized)

    return sanitized_names
