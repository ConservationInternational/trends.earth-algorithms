"""Preview calculations for LDN prioritization."""

from typing import Optional

import numpy as np

from ..util import sampled_raster_values


def estimate_selection(
    score_path: str,
    band_index: int,
    mode: str,
    threshold: float,
    area_km2: float,
    scored_area_km2: Optional[float],
) -> dict:
    """Estimate selection size from sampled scores; the final run remains exact."""
    sample = sampled_raster_values(score_path, band_index)
    values = sample["values"]
    if values.size == 0:
        return {"has_scores": False, "sampled": sample["sampled"]}
    result: dict = {
        "has_scores": True,
        "sampled": True,
        "scored_area_km2": scored_area_km2,
    }
    if mode == "threshold":
        fraction = float(np.count_nonzero(values >= threshold)) / values.size
        result["selected_fraction"] = fraction
        if scored_area_km2 is not None:
            result["selected_area_km2"] = fraction * scored_area_km2
    else:
        if scored_area_km2 is None or scored_area_km2 <= 0:
            result["selected_fraction"] = None
        elif area_km2 >= scored_area_km2:
            result["selected_fraction"] = 1.0
            result["selected_area_km2"] = scored_area_km2
            result["binding_score"] = None
        else:
            fraction = area_km2 / scored_area_km2
            result["selected_fraction"] = fraction
            result["selected_area_km2"] = area_km2
            result["binding_score"] = float(np.quantile(values, 1 - fraction))
    return result
