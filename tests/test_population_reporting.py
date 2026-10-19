import json
from types import SimpleNamespace

import numpy as np
import pytest
from te_algorithms.gdal import drought
from te_algorithms.gdal.land_deg import config, land_deg, land_deg_report, models
from te_algorithms.gdal.population import COUNTS, DENSITY_HA, DENSITY_KM2
from te_schemas import land_cover
from te_schemas.datafile import DataFile
from te_schemas.productivity import ProductivityMode
from te_schemas.results import Band


@pytest.fixture
def population_grid():
    counts = np.array(
        [
            [0.25, 40000.5, 1.5, 2.75, 3.125, 0],
            [7.25, -32768, np.nan, np.inf, 4.625, 9.875],
        ]
    )
    areas = np.array([[0.0009], [0.00045]])
    mask = np.zeros(counts.shape, dtype=bool)
    mask[-1, -1] = True
    return counts, areas, mask


def _population_inputs(counts, areas, units, by_sex, years):
    bands, arrays = [], []
    valid = np.isfinite(counts) & (counts != -32768)
    for pop_type, factor in (("male", 1), ("female", 2)) if by_sex else (("total", 1),):
        for offset, year in enumerate(years, start=1):
            values = counts.copy()
            values[valid] *= factor * offset
            if units in (DENSITY_HA, DENSITY_KM2):
                values[valid] /= np.broadcast_to(
                    areas * (100 if units == DENSITY_HA else 1), counts.shape
                )[valid]
            metadata = {"type": pop_type, "year": year}
            if units is not None:
                metadata["units"] = units
            bands.append(Band(config.POPULATION_BAND_NAME, metadata=metadata))
            arrays.append(values)
    return bands, arrays


def _assert_population_json(report, classes, counts, mask, by_sex, factor=1):
    valid = np.isfinite(counts) & (counts != -32768) & ~mask
    expected = {
        label: float(counts[valid & class_mask].sum()) * factor
        for label, class_mask in classes.items()
    }
    factors = {"Total population": 3 if by_sex else 1}
    if by_sex:
        factors.update({"Male population": 1, "Female population": 2})
    assert set(report) == set(factors)
    for pop_type, multiplier in factors.items():
        values = {
            item["name"]: item["population"] for item in report[pop_type]["values"]
        }
        assert values == pytest.approx(
            {label: value * multiplier for label, value in expected.items()},
            rel=1e-12,
            abs=1e-10,
        )


def _aoi():
    return SimpleNamespace(
        get_geojson=lambda: {"type": "Polygon", "coordinates": []},
        get_crs_wkt=lambda: "EPSG:4326",
    )


@pytest.mark.parametrize("units", [None, COUNTS, DENSITY_HA, DENSITY_KM2])
@pytest.mark.parametrize("by_sex", [False, True])
def test_drought_json_reports_counts_not_density(
    tmp_path, population_grid, units, by_sex
):
    counts, areas, mask = population_grid
    years = list(range(2000, 2005))
    spi = np.array([[0, -500, -1250, -1750, -2250, -32768]] * 2)
    bands = [Band(drought.SPI_BAND_NAME, metadata={"year": year}) for year in years]
    pop_bands, pop_arrays = _population_inputs(counts, areas, units, by_sex, years)
    bands += pop_bands + [
        Band(drought.JRC_BAND_NAME, metadata={}),
        Band(drought.WATER_MASK_BAND_NAME, metadata={}),
    ]
    water = np.zeros(counts.shape)
    water[1, 0] = 1
    arrays = np.array([spi] * len(years) + pop_arrays + [np.zeros(counts.shape), water])
    params = drought.DroughtSummaryParams(
        DataFile("in.vrt", bands), "out.tif", 4, "mask.tif"
    )
    summary, _ = drought._process_block(params, arrays, mask, 0, 0, areas)
    output = tmp_path / "drought.json"
    result = drought.save_reporting_json(
        output, summary, {"layer_spi_years": years}, "test", _aoi()
    )
    assert result == json.loads(output.read_text())
    classes = {
        label: spi == value
        for label, value in [
            ("Non-drought", 0),
            ("Mild drought", -500),
            ("Moderate drought", -1250),
            ("Severe drought", -1750),
            ("Extreme drought", -2250),
            ("No data", -32768),
        ]
    }
    for offset, year in enumerate(years, start=1):
        _assert_population_json(
            result["drought"]["tier_two"][str(year)],
            classes,
            counts,
            mask | (water == 1),
            by_sex,
            offset,
        )


@pytest.mark.parametrize("units", [None, COUNTS, DENSITY_HA, DENSITY_KM2])
@pytest.mark.parametrize("by_sex", [False, True])
def test_land_degradation_json_reports_counts_not_density(
    tmp_path, population_grid, units, by_sex
):
    counts, areas, mask = population_grid
    legend = land_cover.LCLegend(
        "Test",
        [land_cover.LCClass(1, "Land"), land_cover.LCClass(7, "Water")],
        land_cover.LCClass(-32768, "No data"),
    )
    nesting = land_cover.LCLegendNesting(
        legend, legend, {1: [1], 7: [7], -32768: [-32768]}
    )
    matrix = land_cover.LCTransitionDefinitionDeg(
        legend,
        "Test",
        land_cover.LCTransitionMatrixDeg(
            [
                land_cover.LCTransitionMeaningDeg(a, b, "stable")
                for a in legend.key
                for b in legend.key
            ],
            "Test",
        ),
    )
    bands = [Band("Land cover", metadata={"year": year}) for year in (2000, 2001)] + [
        Band(config.LC_DEG_BAND_NAME, metadata={}),
        Band(config.SOC_DEG_BAND_NAME, metadata={}),
        Band(config.CUSTOM_LPD_BAND_NAME, metadata={}),
    ]
    degradation = np.array([[-1, 0, 1, -32768, -1, 1]] * 2)
    pop_bands, pop_arrays = _population_inputs(counts, areas, units, by_sex, [2001])
    bands += pop_bands
    arrays = np.array(
        [
            np.ones(counts.shape),
            np.ones(counts.shape),
            degradation,
            np.zeros(counts.shape),
            np.full(counts.shape, 4),
        ]
        + pop_arrays
    )
    periods = {
        name: {"year_initial": 2000, "year_final": 2001}
        for name in ("land_cover", "soc", "productivity")
    }
    params = models.DegradationSummaryParams(
        DataFile("in.vrt", bands),
        ProductivityMode.CUSTOM_5_CLASS_LPD.value,
        "in.vrt",
        "out.tif",
        1,
        2,
        "mask.tif",
        nesting,
        matrix,
        "baseline",
        periods,
    )
    summary, _ = land_deg._process_block_summary(params, arrays, mask, 0, 0, areas)
    output = tmp_path / "land.json"
    result = land_deg_report.save_reporting_json(
        output,
        {"baseline": summary},
        None,
        None,
        {
            "periods": [
                {
                    "name": "baseline",
                    "params": {
                        "periods": periods,
                        "layer_lc_years": [2000, 2001],
                        "layer_soc_years": [],
                    },
                }
            ]
        },
        "test",
        _aoi(),
        {"baseline": {"lc_legend_nesting": nesting, "lc_trans_matrix": matrix}},
    )
    assert result == json.loads(output.read_text())
    _assert_population_json(
        result["affected_population"]["baseline"]["summary"],
        {
            label: degradation == value
            for label, value in [
                ("Degraded", -1),
                ("Stable", 0),
                ("Improved", 1),
                ("No data", -32768),
            ]
        },
        counts,
        mask,
        by_sex,
    )


@pytest.mark.parametrize("pop_type", ["total", "male", "female"])
def test_population_output_bands_are_explicitly_counts(pop_type):
    drought_band = drought._get_population_band_instance(pop_type, 2000, 2003)
    degradation_band = land_deg._get_so3_band_instance(
        pop_type, {"year_initial": 2000, "year_final": 2003}
    )
    for band in (drought_band, degradation_band):
        assert band.metadata["units"] == COUNTS
