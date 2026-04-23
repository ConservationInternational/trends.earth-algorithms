"""Utility helpers for Trends.Earth openEO algorithm implementations."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend-specific normalisations
# ---------------------------------------------------------------------------

#: Per-backend metadata: nodata values, default CRS, and collection aliases.
#: Keys are substrings of the backend URL (matched case-insensitively).
BACKEND_NORMALIZATIONS: dict[str, dict[str, Any]] = {
    "openeo.vito.be": {
        # VITO Terrascope – SoilGrids collection uses native float values;
        # no explicit nodata masking is required.
        "nodata_value": None,
        "crs_default": "EPSG:4326",
        "collection_aliases": {
            # Logical name → actual Terrascope collection ID
            "SOC_BASELINE": "TERRASCOPE_WORLD_SOILGRIDS",
            "SOC_BASELINE_BAND": "SOC",
            "LAND_COVER": "ESA_CCI_LC",
            "CLIMATE_ZONES": "IPCC_CLIMATE_ZONES",
        },
        # Last year available in the land cover collection (mirrors GEE asset 1992-2022)
        "lc_last_year": 2022,
    },
}


def get_backend_normalizations(backend_url: str) -> dict[str, Any]:
    """Return the normalisation metadata for *backend_url*.

    Falls back to an empty dict when no entry matches.
    """
    url_lower = backend_url.lower()
    for key, meta in BACKEND_NORMALIZATIONS.items():
        if key in url_lower:
            return meta
    logger.warning(
        "No backend normalisation found for '%s'.  Using empty defaults.",
        backend_url,
    )
    return {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def normalise_nodata(array: Any, gee_nodata: int = -32768) -> Any:
    """Replace GEE-style nodata sentinel values with NaN in a numpy array.

    This helper is useful when post-processing GEE-origin rasters that were
    exported with a -32768 sentinel before being ingested into openEO.

    Args:
        array: A numpy array (or array-like) containing pixel data.
        gee_nodata: The sentinel value to replace (default -32768).

    Returns:
        The array with sentinel values replaced by ``float('nan')``.
    """
    import numpy as np

    arr = np.asarray(array, dtype=float)
    arr[arr == gee_nodata] = float("nan")
    return arr


def build_s3_output_path(
    execution_id: str, bucket: str, prefix: str = "outputs"
) -> str:
    """Construct an S3 URI for openEO job output.

    Args:
        execution_id: The Trends.Earth execution UUID.
        bucket: The S3 bucket name.
        prefix: The key prefix (default ``"outputs"``).

    Returns:
        An ``s3://`` URI, e.g. ``s3://mybucket/outputs/<execution_id>/``.
    """
    prefix = prefix.rstrip("/")
    return f"s3://{bucket}/{prefix}/{execution_id}/"


def save_result_to_s3(
    datacube: Any,
    execution_id: str,
    filename_prefix: str,
) -> Any:
    """Save an openEO datacube result to S3, reading destination from the environment.

    Reads ``OUTPUT_S3_BUCKET`` and ``OUTPUT_S3_PREFIX`` from environment
    variables (both required).

    Args:
        datacube: The openEO datacube to save.
        execution_id: The Trends.Earth execution UUID used to name S3 outputs.
        filename_prefix: Prefix for the output filename (e.g. ``"soc"`` →
            ``"soc_<execution_id>.tif``).

    Returns:
        The datacube with a ``save_result`` step appended, ready to be
        submitted as a batch job.

    Raises:
        RuntimeError: When ``OUTPUT_S3_BUCKET`` or ``OUTPUT_S3_PREFIX`` is not set.
    """
    import os

    bucket = os.environ.get("OUTPUT_S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "OUTPUT_S3_BUCKET environment variable is not set. "
            "Cannot save openEO results without an S3 destination."
        )
    prefix = os.environ.get("OUTPUT_S3_PREFIX")
    if not prefix:
        raise RuntimeError(
            "OUTPUT_S3_PREFIX environment variable is not set. "
            "Cannot save openEO results without an S3 destination."
        )
    s3_uri = build_s3_output_path(execution_id, bucket, prefix)
    return datacube.save_result(
        format="GTiff",
        options={
            "filename_prefix": f"{filename_prefix}_{execution_id}",
            "path": s3_uri,
        },
    )


def wait_for_job(
    connection: Any,
    job_id: str,
    poll_interval: int = 30,
    timeout: int = 3600,
) -> str:
    """Poll an openEO batch job until it finishes or times out.

    Args:
        connection: An authenticated :class:`openeo.Connection`.
        job_id: The openEO batch job ID to poll.
        poll_interval: Seconds between status polls (default 30).
        timeout: Maximum seconds to wait before raising (default 3600).

    Returns:
        The final job status string (e.g. ``"finished"`` or ``"error"``).

    Raises:
        TimeoutError: When the job hasn't completed within *timeout* seconds.
    """
    job = connection.job(job_id)
    elapsed = 0
    while elapsed < timeout:
        status = job.status()
        logger.debug("Job %s status: %s (elapsed %ds)", job_id, status, elapsed)
        if status in ("finished", "error", "canceled"):
            return status
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(
        f"openEO job {job_id!r} did not complete within {timeout}s "
        f"(last status: {job.status()!r})"
    )


def bbox_from_geojsons(geojsons: list) -> dict:
    """Return a WGS-84 bounding box dict covering all geometries in *geojsons*.

    Expected format: list of GeoJSON Feature or Geometry dicts.
    """
    lons: list[float] = []
    lats: list[float] = []
    for item in geojsons:
        geom = item.get("geometry", item)
        _collect_coords(geom.get("coordinates", []), lons, lats)

    if not lons:
        # Fallback: global extent
        return {
            "west": -180,
            "south": -90,
            "east": 180,
            "north": 90,
            "crs": "EPSG:4326",
        }

    return {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
        "crs": "EPSG:4326",
    }


def _collect_coords(coords: Any, lons: list, lats: list) -> None:
    """Recursively collect lon/lat values from GeoJSON coordinate arrays."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        lons.append(coords[0])
        lats.append(coords[1])
    else:
        for sub in coords:
            _collect_coords(sub, lons, lats)
