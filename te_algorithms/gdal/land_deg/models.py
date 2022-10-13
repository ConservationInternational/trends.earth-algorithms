import dataclasses
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import marshmallow_dataclass
from te_schemas import land_cover
from te_schemas import SchemaBase
from te_schemas.datafile import DataFile

from .. import util_numba


@marshmallow_dataclass.dataclass
class SummaryTableLD(SchemaBase):
    soc_by_lc_annual_totals: List[Dict[int, float]]
    lc_annual_totals: List[Dict[int, float]]
    lc_trans_zonal_areas: List[Dict[int, float]]
    lc_trans_zonal_areas_periods: List[Dict[str, float]]
    lc_trans_prod_bizonal: Dict[Tuple[int, int], float]
    lc_trans_zonal_soc_initial: Dict[int, float]
    lc_trans_zonal_soc_final: Dict[int, float]
    sdg_zonal_population_total: Dict[int, float]
    sdg_zonal_population_male: Dict[int, float]
    sdg_zonal_population_female: Dict[int, float]
    sdg_summary: Dict[int, float]
    prod_summary: Dict[str, Dict[int, float]]
    soc_summary: Dict[str, Dict[int, float]]
    lc_summary: Dict[int, float]

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
        self.lc_trans_zonal_soc_initial = dict(self.lc_trans_zonal_soc_initial)
        self.lc_trans_zonal_soc_final = dict(self.lc_trans_zonal_soc_final)
        self.sdg_zonal_population_total = dict(self.sdg_zonal_population_total)
        self.sdg_zonal_population_male = dict(self.sdg_zonal_population_male)
        self.sdg_zonal_population_female = dict(self.sdg_zonal_population_female)
        self.sdg_summary = dict(self.sdg_summary)
        self.prod_summary = {
            str(key): dict(value) for key, value in self.prod_summary.items()
        }
        self.soc_summary = {
            str(key): dict(value) for key, value in self.soc_summary.items()
        }
        self.lc_summary = dict(self.lc_summary)


@marshmallow_dataclass.dataclass
class SummaryTableLDProgress(SchemaBase):
    sdg_summary: Dict[int, float]
    prod_summary: Dict[str, Dict[int, float]]
    soc_summary: Dict[str, Dict[int, float]]
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
