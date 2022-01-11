import dataclasses
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import marshmallow_dataclass
from te_schemas import land_cover
from te_schemas import SchemaBase
from te_schemas.datafile import DataFile


@marshmallow_dataclass.dataclass
class SummaryTableLD(SchemaBase):
    soc_by_lc_annual_totals: List[Dict[int, float]]
    lc_annual_totals: List[Dict[int, float]]
    lc_trans_zonal_areas: List[Dict[int, float]]
    lc_trans_zonal_areas_periods: List[Dict[str, int]]
    lc_trans_prod_bizonal: Dict[Tuple[int, int], float]
    lc_trans_zonal_soc_initial: Dict[int, float]
    lc_trans_zonal_soc_final: Dict[int, float]
    sdg_zonal_population_total: Dict[int, float]
    sdg_zonal_population_male: Dict[int, float]
    sdg_zonal_population_female: Dict[int, float]
    sdg_summary: Dict[int, float]
    prod_summary: Dict[int, float]
    soc_summary: Dict[int, float]
    lc_summary: Dict[int, float]


@marshmallow_dataclass.dataclass
class SummaryTableLDProgress(SchemaBase):
    sdg_summary: Dict[int, float]
    prod_summary: Dict[int, float]
    soc_summary: Dict[int, float]
    lc_summary: Dict[int, float]


@marshmallow_dataclass.dataclass
class SummaryTableLDErrorRecode(SchemaBase):
    sdg_summary: Dict[int, float]


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
class DegradationProgressSummaryParams(SchemaBase):
    prod_mode: str
    in_file: str
    out_file: str
    band_dict: dict
    model_band_number: int
    n_out_bands: int
    mask_file: str


@dataclasses.dataclass()
class DegradationErrorRecodeSummaryParams(SchemaBase):
    in_file: str
    out_file: str
    band_dict: dict
    model_band_number: int
    n_out_bands: int
    mask_file: str
    trans_code_lists: tuple
