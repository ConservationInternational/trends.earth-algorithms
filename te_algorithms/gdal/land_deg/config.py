import numpy as np

# Ensure mask and nodata values are saved as 16 bit integers to keep numba
# happy
NODATA_VALUE = np.int16(-32768)
MASK_VALUE = np.int16(-32767)

SDG_BAND_NAME = "SDG 15.3.1 Indicator"
SDG_STATUS_BAND_NAME = "SDG 15.3.1 Indicator (status)"
ERROR_RECODE_BAND_NAME = "Error recode"
PROD_DEG_COMPARISON_BAND_NAME = "Productivity degradation (comparison)"
JRC_LPD_BAND_NAME = "Land Productivity Dynamics (from JRC)"
FAO_WOCAT_LPD_BAND_NAME = "Land Productivity Dynamics (from FAO-WOCAT)"
TE_LPD_BAND_NAME = "Land Productivity Dynamics (from Trends.Earth)"
TRAJ_BAND_NAME = "Productivity trajectory (significance)"
PERF_BAND_NAME = "Productivity performance (degradation)"
STATE_BAND_NAME = "Productivity state (degradation)"
LC_DEG_BAND_NAME = "Land cover (degradation)"
LC_DEG_COMPARISON_BAND_NAME = "Land cover degradation (comparison)"
LC_BAND_NAME = ["Land cover", "Land cover (7 class)"]
LC_TRANS_BAND_NAME = "Land cover transitions"
SOC_DEG_BAND_NAME = "Soil organic carbon (degradation)"
SOC_BAND_NAME = "Soil organic carbon"
POPULATION_BAND_NAME = "Population (number of people)"
POP_AFFECTED_BAND_NAME = "Population affected by degradation (number of people)"

PRODUCTIVITY_CLASS_KEY = {
    "Increasing": 5,
    "Stable": 4,
    "Stressed": 3,
    "Moderate decline": 2,
    "Declining": 1,
    "No data": NODATA_VALUE,
}
