"""Regression tests for SOC handling in the land degradation progress summary.

The baseline SOC degradation band holds percent change in SOC (degraded at
<= -10%, improved at >= 10%). It must be recoded to -1/0/1 classes before it is
compared with the reporting period, otherwise most baseline pixels fall outside
the class codes and are dropped from the SOC change table and status layers.
"""

from types import SimpleNamespace

import numpy as np

from te_algorithms.gdal.land_deg import config, land_deg_progress

NODATA = config.NODATA_VALUE

# Baseline SOC percent change, one pixel per column. The last pixel is water.
SOC_DEG_PERCENT = [-25, -10, -9, -1, 0, 1, 9, 10, 30, NODATA, 0]
WATER_COLUMN = len(SOC_DEG_PERCENT) - 1

BAND_DICT = {
    "lc_baseline_bandnum": 1,
    "sdg_baseline_bandnum": 2,
    "prod5_baseline_bandnum": 3,
    "lc_deg_baseline_bandnum": 4,
    "soc_deg_baseline_bandnum": 5,
    "soc_baseline_bandnum": 6,
    "sdg_reporting_0_bandnum": 7,
    "prod5_reporting_0_bandnum": 8,
    "lc_deg_reporting_0_bandnum": 9,
    "soc_reporting_0_bandnum": 10,
}


def _run_block():
    n_cols = len(SOC_DEG_PERCENT)
    in_array = np.zeros((len(BAND_DICT), 1, n_cols), dtype=np.int16)

    def band(name):
        return in_array[BAND_DICT[name] - 1]

    band("lc_baseline_bandnum")[:] = 1
    band("lc_baseline_bandnum")[0, WATER_COLUMN] = 7
    band("prod5_baseline_bandnum")[:] = 5
    band("prod5_reporting_0_bandnum")[:] = 5
    band("soc_deg_baseline_bandnum")[0, :] = SOC_DEG_PERCENT
    # Identical stocks, so the reporting period is stable (0% change) everywhere
    band("soc_baseline_bandnum")[:] = 50
    band("soc_reporting_0_bandnum")[:] = 50

    params = SimpleNamespace(
        band_dict=BAND_DICT,
        n_reporting=1,
        nesting=SimpleNamespace(nesting={7: [7]}),
    )
    mask = np.zeros((1, n_cols), dtype=bool)
    cell_areas = np.ones((1, 1), dtype=np.float64)

    (status, change), write_arrays = land_deg_progress._process_block_status(
        params, in_array, mask, 0, 0, cell_areas
    )
    return status, change, write_arrays


def test_soc_change_crosstab_uses_recoded_baseline_classes():
    _, change, _ = _run_block()
    crosstab = change.soc_crosstabs[0]

    assert {bl for bl, _ in crosstab} <= {-1, 0, 1, NODATA}
    assert crosstab == {
        (-1, 0): 2.0,  # -25%, -10%
        (0, 0): 5.0,  # -9%, -1%, 0%, 1%, 9%
        (1, 0): 2.0,  # 10%, 30%
        (NODATA, 0): 1.0,  # baseline nodata
        (NODATA, NODATA): 1.0,  # water
    }
    assert sum(crosstab.values()) == len(SOC_DEG_PERCENT)


def test_soc_status_uses_ten_percent_thresholds():
    status, _, _ = _run_block()
    summary = status.soc_summaries[0]["all_cover_types"]

    assert summary == {-1: 2.0, 0: 5.0, 1: 2.0, NODATA: 2.0}


def test_soc_status_layer_classes_per_pixel():
    _, _, write_arrays = _run_block()
    # Write order is SDG, productivity, SOC, then LC status for each period
    soc_status = write_arrays[2]["array"]

    np.testing.assert_array_equal(
        soc_status[0],
        [3, 3, 4, 4, 4, 4, 4, 5, 5, NODATA, NODATA],
    )
