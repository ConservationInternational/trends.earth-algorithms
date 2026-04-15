import dataclasses
from typing import Dict, List, Optional, Tuple

import marshmallow_dataclass
from te_schemas import SchemaBase, land_cover
from te_schemas.datafile import DataFile


@marshmallow_dataclass.dataclass
class SummaryTableLD(SchemaBase):
    soc_by_lc_annual_totals: List[Dict[int, float]]
    lc_annual_totals: List[Dict[int, float]]
    lc_trans_zonal_areas: List[Dict[int, float]]
    lc_trans_zonal_areas_periods: List[Dict[str, float]]
    lc_trans_prod_bizonal: Dict[Tuple[int, int], float]
    sdg_zonal_population_total: Dict[int, float]
    sdg_zonal_population_male: Dict[int, float]
    sdg_zonal_population_female: Dict[int, float]
    sdg_summary: Dict[int, float]
    prod_summary: Dict[str, Dict[int, float]]
    lc_summary: Dict[int, float]
    soc_summary: Dict[str, Dict[int, float]]

    def cast_to_cpython(self):
        # Numba compiled functions return numba types which won't pickle correctly
        # (which is needed for multiprocessing), so cast them to regular python types
        self.soc_by_lc_annual_totals = [
            dict(item) for item in self.soc_by_lc_annual_totals
        ]
        self.lc_annual_totals = [dict(item) for item in self.lc_annual_totals]
        self.lc_trans_zonal_areas = [dict(item) for item in self.lc_trans_zonal_areas]
        self.lc_trans_zonal_areas_periods = [
            dict(item) for item in self.lc_trans_zonal_areas_periods
        ]
        self.lc_trans_prod_bizonal = {
            tuple(key): value for key, value in self.lc_trans_prod_bizonal.items()
        }
        self.sdg_zonal_population_total = dict(self.sdg_zonal_population_total)
        self.sdg_zonal_population_male = dict(self.sdg_zonal_population_male)
        self.sdg_zonal_population_female = dict(self.sdg_zonal_population_female)
        self.sdg_summary = dict(self.sdg_summary)
        self.prod_summary = {
            str(key): dict(value) for key, value in self.prod_summary.items()
        }
        self.lc_summary = dict(self.lc_summary)
        self.soc_summary = {
            str(key): dict(value) for key, value in self.soc_summary.items()
        }


@marshmallow_dataclass.dataclass
class SummaryTableLDStatus(SchemaBase):
    """Records land degradation status for one or more periods"""

    sdg_summaries: List[Dict[int, float]]
    prod_summaries: List[Dict[str, Dict[int, float]]]
    lc_summaries: List[Dict[int, float]]
    soc_summaries: List[Dict[str, Dict[int, float]]]


@marshmallow_dataclass.dataclass
class SummaryTableLDChange(SchemaBase):
    """Records change in land degradation status between baseline and one or more periods"""

    sdg_crosstabs: List[Dict[tuple, float]]
    prod_crosstabs: List[Dict[tuple, float]]
    lc_crosstabs: List[Dict[tuple, float]]
    soc_crosstabs: List[Dict[tuple, float]]


@marshmallow_dataclass.dataclass
class SummaryTableLDErrorRecode(SchemaBase):
    baseline_summary: Dict[int, float]
    report_summaries: Optional[List[Dict[int, float]]] = None
    crosstabs: Optional[List] = None  # List of numpy float64[4,4] arrays


@dataclasses.dataclass()
class DegradationSummaryParams(SchemaBase):
    in_df: DataFile
    prod_mode: str
    in_file: str
    out_file: str
    model_band_number: int
    n_out_bands: int
    mask_file: str
    nesting: land_cover.LCLegendNesting
    trans_matrix: land_cover.LCTransitionDefinitionDeg
    period_name: str
    periods: dict
    error_recode: Optional[Dict] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass()
class DegradationStatusSummaryParams(SchemaBase):
    prod_mode: str
    in_file: str
    out_file: str
    band_dict: dict
    model_band_number: int
    n_out_bands: int
    n_reporting: int  # Number of periods in addition to baseline
    mask_file: str
    nesting: land_cover.LCLegendNesting


@dataclasses.dataclass()
class DegradationErrorRecodeSummaryParams(SchemaBase):
    in_file: str
    out_file: str
    band_dict: dict
    model_band_number: int
    n_out_bands: int
    mask_file: str
    trans_code_lists: tuple
    write_reporting_sdg_tifs: bool = (
        False  # Whether to write out the reporting period layers
    )
    baseline_band_num: Optional[int] = None  # Band number for baseline SDG
    report_band_nums: Optional[List[int]] = None  # Band numbers for reporting periods


@marshmallow_dataclass.dataclass
class CounterbalancingLandTypeResult(SchemaBase):
    """Gains, losses, and delta LDN for a single land type."""

    land_type_code: int
    land_type_name: str
    gains_area_sqkm: float
    losses_area_sqkm: float
    total_area_sqkm: float
    delta_ldn: float
    ldn_achieved: bool
    ldn_pct: float  # (gains-losses)/total_area*100 (% of total land area)
    status_breakdown_sqkm: Optional[Dict[int, float]] = None
    transition_breakdown_sqkm: Optional[Dict[int, float]] = None


@marshmallow_dataclass.dataclass
class SummaryTableCounterbalancing(SchemaBase):
    """Accumulated counterbalancing statistics across all land types."""

    # Per-land_type gain and loss areas
    gains_by_land_type: Dict[int, float]
    losses_by_land_type: Dict[int, float]

    # Per-land_type area breakdown by status class (7-class)
    status_breakdown: Optional[Dict[int, Dict[int, float]]] = None
    # Per-land_type baseline→period transition matrix (encoded keys)
    transition_breakdown: Optional[Dict[int, Dict[int, float]]] = None

    def cast_to_cpython(self):
        self.gains_by_land_type = dict(self.gains_by_land_type)
        self.losses_by_land_type = dict(self.losses_by_land_type)
        if self.status_breakdown is not None:
            self.status_breakdown = {
                int(k): dict(v) for k, v in self.status_breakdown.items()
            }
        if self.transition_breakdown is not None:
            self.transition_breakdown = {
                int(k): dict(v) for k, v in self.transition_breakdown.items()
            }


def accumulate_summary_table_counterbalancing(
    tables: List[SummaryTableCounterbalancing],
) -> SummaryTableCounterbalancing:
    from .. import util

    if len(tables) == 1:
        return tables[0]

    out = tables[0]

    for table in tables[1:]:
        out.gains_by_land_type = util.accumulate_dicts(
            [out.gains_by_land_type, table.gains_by_land_type]
        )
        out.losses_by_land_type = util.accumulate_dicts(
            [out.losses_by_land_type, table.losses_by_land_type]
        )
        if table.status_breakdown is not None:
            out.status_breakdown = util.accumulate_nested_dicts(
                out.status_breakdown, table.status_breakdown
            )
        if table.transition_breakdown is not None:
            out.transition_breakdown = util.accumulate_nested_dicts(
                out.transition_breakdown, table.transition_breakdown
            )

    return out


def accumulate_summarytableld(
    tables: List[SummaryTableLD],
) -> SummaryTableLD:
    from .. import util  # needed for multiprocessing

    if len(tables) == 1:
        return tables[0]

    out = tables[0]

    for table in tables[1:]:
        out.soc_by_lc_annual_totals = [
            util.accumulate_dicts([a, b])
            for a, b in zip(out.soc_by_lc_annual_totals, table.soc_by_lc_annual_totals)
        ]
        out.lc_annual_totals = [
            util.accumulate_dicts([a, b])
            for a, b in zip(out.lc_annual_totals, table.lc_annual_totals)
        ]
        out.lc_trans_zonal_areas = [
            util.accumulate_dicts([a, b])
            for a, b in zip(out.lc_trans_zonal_areas, table.lc_trans_zonal_areas)
        ]
        # A period should be listed for each object in lc_trans_zonal_areas
        assert len(out.lc_trans_zonal_areas) == len(table.lc_trans_zonal_areas_periods)
        # Periods for lc_trans_zonal_areas must be the same in both objects
        assert out.lc_trans_zonal_areas_periods == table.lc_trans_zonal_areas_periods
        out.lc_trans_prod_bizonal = util.accumulate_dicts(
            [out.lc_trans_prod_bizonal, table.lc_trans_prod_bizonal]
        )
        out.sdg_zonal_population_total = util.accumulate_dicts(
            [out.sdg_zonal_population_total, table.sdg_zonal_population_total]
        )
        out.sdg_zonal_population_male = util.accumulate_dicts(
            [out.sdg_zonal_population_male, table.sdg_zonal_population_male]
        )
        out.sdg_zonal_population_female = util.accumulate_dicts(
            [out.sdg_zonal_population_female, table.sdg_zonal_population_female]
        )
        out.sdg_summary = util.accumulate_dicts([out.sdg_summary, table.sdg_summary])
        assert set(out.prod_summary.keys()) == set(table.prod_summary.keys())
        out.prod_summary = {
            key: util.accumulate_dicts([out.prod_summary[key], table.prod_summary[key]])
            for key in out.prod_summary.keys()
        }
        assert set(out.soc_summary.keys()) == set(table.soc_summary.keys())
        out.soc_summary = {
            key: util.accumulate_dicts([out.soc_summary[key], table.soc_summary[key]])
            for key in out.soc_summary.keys()
        }
        out.lc_summary = util.accumulate_dicts([out.lc_summary, table.lc_summary])

    return out
