import numpy as np

try:
    import numba
    from numba.pycc import CC

except ImportError:
    # Will use these as regular Python functions if numba is not present.
    class DecoratorSubstitute:
        # Make a cc.export that doesn't do anything
        def export(*args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

        # Make a numba.jit that doesn't do anything
        def jit(*args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

    cc = DecoratorSubstitute()
    numba = DecoratorSubstitute()
else:
    cc = CC("land_deg_numba")

# Ensure mask and nodata values are saved as 16 bit integers to keep numba
# happy
NODATA_VALUE = np.array([-32768], dtype=np.int16)
MASK_VALUE = np.array([-32767], dtype=np.int16)


@numba.jit(nopython=True)
@cc.export(
    "recode_indicator_errors", "i2[:,:](i2[:,:], i2[:,:], i2[:], i2[:], i2[:], i2[:])"
)
def recode_indicator_errors(x, recode, codes, deg_to, stable_to, imp_to):
    """Optimized version with numba JIT compilation enabled"""
    out = x.copy()

    # Vectorized approach: process all codes at once for each category
    for i in range(len(codes)):
        code = codes[i]
        mask = recode == code

        if deg_to[i] != NODATA_VALUE[0]:  # Use explicit comparison instead of None
            out[(x == -1) & mask] = deg_to[i]

        if stable_to[i] != NODATA_VALUE[0]:
            out[(x == 0) & mask] = stable_to[i]

        if imp_to[i] != NODATA_VALUE[0]:
            out[(x == 1) & mask] = imp_to[i]

    return out


@numba.jit(nopython=True)
@cc.export("recode_traj", "i2[:,:](i2[:,:])")
def recode_traj(x):
    # Recode trajectory into deg, stable, imp. Capture trends that are at least
    # 95% significant.
    #
    # Remember that traj is coded as:
    # -3: 99% signif decline
    # -2: 95% signif decline
    # -1: 90% signif decline
    #  0: stable
    #  1: 90% signif increase
    #  2: 95% signif increase
    #  3: 99% signif increase

    # Optimized: work directly on flattened array to avoid reshape overhead
    original_shape = x.shape
    x_flat = x.ravel()
    out = x_flat.copy()

    # Vectorized operations - more efficient than individual assignments
    # Stable: -1 to 1 (not significant at 95%)
    stable_mask = (x_flat >= -1) & (x_flat <= 1)
    out[stable_mask] = 0

    # Declining: -3 to -2 (95%+ significant decline)
    decline_mask = (x_flat >= -3) & (x_flat < -1)
    out[decline_mask] = -1

    # Improving: 2 to 3 (95%+ significant increase)
    improve_mask = (x_flat > 1) & (x_flat <= 3)
    out[improve_mask] = 1

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("recode_state", "i2[:,:](i2[:,:])")
def recode_state(x):
    # Recode state into deg, stable, imp. Note the >= -10 is so no data
    # isn't coded as degradation. More than two changes in class is defined
    # as degradation in state.

    # Optimized: work directly on flattened array
    original_shape = x.shape
    x_flat = x.ravel()
    out = x_flat.copy()

    # Vectorized operations for better performance
    # NODATA: x < -10 
    nodata_mask = x_flat < -10
    out[nodata_mask] = NODATA_VALUE[0]
    
    # Stable: -2 < x < 2
    stable_mask = (x_flat > -2) & (x_flat < 2)
    out[stable_mask] = 0

    # Declining: -10 <= x <= -2
    decline_mask = (x_flat >= -10) & (x_flat <= -2)
    out[decline_mask] = -1

    # Improving: x >= 2
    improve_mask = x_flat >= 2
    out[improve_mask] = 1

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("calc_progress_lc_deg", "i2[:,:](i2[:,:], i2[:,:])")
def calc_progress_lc_deg(initial, final):
    # First need to calculate transitions, then recode them as deg, stable,
    # improved
    #
    # -32768: no data
    shp = initial.shape
    initial = initial.ravel()
    final = final.ravel()
    out = initial.copy()

    # improvements on areas that were degraded at baseline -> stable
    out[(initial == -1) & (final == 1)] = 0
    # improvements on areas that were stable at baseline -> improved
    out[(initial == 0) & (final == 1)] = 1
    # degradation during progress -> degraded
    out[final == -1] = -1

    return np.reshape(out, shp)


@numba.jit(nopython=True)
@cc.export("calc_prod5", "i2[:,:](i2[:,:], i2[:,:], i2[:,:])")
def calc_prod5(traj, state, perf):
    # Coding of LPD (prod5)
    # 1: declining
    # 2: early signs of decline
    # 3: stable but stressed
    # 4: stable
    # 5: improving
    # -32768: no data

    original_shape = traj.shape

    # Work on flattened arrays for better performance
    traj_flat = traj.ravel()
    state_flat = state.ravel()
    perf_flat = perf.ravel()

    # Start with trajectory as base and modify
    out = traj_flat.copy()

    # Vectorized assignments - more efficient than individual conditions
    # Declining = 1
    decline_mask = traj_flat == -1
    out[decline_mask] = 1

    # Stable = 4
    stable_mask = traj_flat == 0
    out[stable_mask] = 4

    # Improving = 5
    improve_mask = traj_flat == 1
    out[improve_mask] = 5

    # Complex conditions using combined masks for efficiency
    # Stressed: stable traj + stable state + declining perf
    stressed_mask = (traj_flat == 0) & (state_flat == 0) & (perf_flat == -1)
    out[stressed_mask] = 3

    # Early signs of decline: various combinations
    early_decline_mask1 = (traj_flat == 1) & (state_flat == -1) & (perf_flat == -1)
    early_decline_mask2 = (traj_flat == 0) & (state_flat == -1) & (perf_flat == 0)
    out[early_decline_mask1 | early_decline_mask2] = 2

    # Definitive decline: stable traj + declining state + declining perf
    definitive_decline_mask = (traj_flat == 0) & (state_flat == -1) & (perf_flat == -1)
    out[definitive_decline_mask] = 1

    # Handle NODATA efficiently with single combined mask
    nodata_mask = (
        (traj_flat == NODATA_VALUE[0])
        | (perf_flat == NODATA_VALUE[0])
        | (state_flat == NODATA_VALUE[0])
    )
    out[nodata_mask] = NODATA_VALUE[0]

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("sdg_status_expanded", "i2[:,:](i2[:,:], i2[:,:])")
def sdg_status_expanded(sdg_bl, sdg_tg):
    """
    Optimized SDG Status layer following GPG addendum "expanded status matrix"
    """
    original_shape = sdg_bl.shape
    sdg_bl_flat = sdg_bl.ravel()
    sdg_tg_flat = sdg_tg.ravel()

    out = np.full(sdg_bl_flat.shape[0], NODATA_VALUE[0], dtype=np.int16)

    # Vectorized lookup using more efficient approach
    # Create combined condition masks for better performance
    out[(sdg_bl_flat == -1) & (sdg_tg_flat == -1)] = 1

    # Group similar assignments to reduce mask calculations
    condition_2 = ((sdg_bl_flat == 0) | (sdg_bl_flat == 1)) & (sdg_tg_flat == -1)
    out[condition_2] = 2

    out[(sdg_bl_flat == -1) & (sdg_tg_flat == 0)] = 3
    out[(sdg_bl_flat == 0) & (sdg_tg_flat == 0)] = 4
    out[(sdg_bl_flat == 1) & (sdg_tg_flat == 0)] = 5

    # Group conditions for value 6
    condition_6 = ((sdg_bl_flat == -1) | (sdg_bl_flat == 0)) & (sdg_tg_flat == 1)
    out[condition_6] = 6

    out[(sdg_bl_flat == 1) & (sdg_tg_flat == 1)] = 7

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("sdg_status_expanded_to_simple", "i2[:,:](i2[:,:])")
def sdg_status_expanded_to_simple(sdg_status):
    """
    Optimized conversion of expanded status matrix to simple deg/stable/not deg layer
    """
    original_shape = sdg_status.shape
    sdg_status_flat = sdg_status.ravel()

    out = sdg_status_flat.copy()

    # Vectorized assignments using efficient masks
    # Degraded: status 1, 2, 3 -> -1
    degraded_mask = (
        (sdg_status_flat == 1) | (sdg_status_flat == 2) | (sdg_status_flat == 3)
    )
    out[degraded_mask] = -1

    # Stable: status 4 -> 0
    stable_mask = sdg_status_flat == 4
    out[stable_mask] = 0

    # Improved: status 5, 6, 7 -> 1
    improved_mask = (
        (sdg_status_flat == 5) | (sdg_status_flat == 6) | (sdg_status_flat == 7)
    )
    out[improved_mask] = 1

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("prod5_to_prod3", "i2[:,:](i2[:,:])")
def prod5_to_prod3(prod5):
    """Optimized conversion from 5-class to 3-class productivity"""
    original_shape = prod5.shape
    prod5_flat = prod5.ravel()
    out = prod5_flat.copy()

    # Vectorized operations for better performance
    # Declining: classes 1 and 2 -> -1
    decline_mask = (prod5_flat == 1) | (prod5_flat == 2)
    out[decline_mask] = -1

    # Stable: classes 3 and 4 -> 0
    stable_mask = (prod5_flat == 3) | (prod5_flat == 4)
    out[stable_mask] = 0

    # Improving: class 5 -> 1
    improve_mask = prod5_flat == 5
    out[improve_mask] = 1

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export(
    "calc_lc_trans", "i4[:,:](i2[:,:], i2[:,:], i4, optional(i2[:]), optional(i2[:]))"
)
def calc_lc_trans(lc_bl, lc_tg, multiplier, recode_from=None, recode_to=None):
    """Optimized land cover transition calculation"""
    original_shape = lc_bl.shape
    lc_bl_flat = lc_bl.ravel()
    lc_tg_flat = lc_tg.ravel()

    # Apply recoding if provided - optimize with single pass
    if recode_from is not None and recode_to is not None:
        # Create lookup arrays for faster recoding
        for i in range(len(recode_from)):
            value = int(recode_from[i])
            replacement = int(recode_to[i])

            # Vectorized recoding
            bl_mask = lc_bl_flat == value
            tg_mask = lc_tg_flat == value
            lc_bl_flat[bl_mask] = replacement
            lc_tg_flat[tg_mask] = replacement

    # Calculate transitions efficiently - convert to int32 to handle large values
    a_trans_bl_tg = lc_bl_flat.astype(np.int32) * multiplier + lc_tg_flat.astype(np.int32)

    # Apply invalid data mask in single operation
    invalid_mask = (lc_bl_flat < 1) | (lc_tg_flat < 1)
    a_trans_bl_tg[invalid_mask] = NODATA_VALUE[0]

    return a_trans_bl_tg.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("recode_deg_soc", "i2[:,:](i2[:,:], i2[:,:])")
def recode_deg_soc(soc, water):
    """recode SOC change layer from percent change into a categorical map"""
    # Degradation in terms of SOC is defined as a decline of more
    # than 10% (and improving increase greater than 10%)
    shp = soc.shape
    soc = soc.ravel()
    water = water.ravel()
    out = soc.copy()
    out[(soc >= -101) & (soc <= -10)] = -1
    out[(soc > -10) & (soc < 10)] = 0
    out[soc >= 10] = 1
    out[water == 1] = NODATA_VALUE[0]  # don't count soc in water

    return np.reshape(out, shp)


@numba.jit(nopython=True)
@cc.export("calc_soc_pch", "i2[:,:](i2[:,:], i2[:,:])")
def calc_soc_pch(soc_bl, soc_tg):
    """calculate percent change in SOC from initial and final SOC"""
    # Degradation in terms of SOC is defined as a decline of more
    # than 10% (and improving increase greater than 10%)
    shp = soc_bl.shape
    soc_bl = soc_bl.ravel()
    soc_tg = soc_tg.ravel()
    soc_chg = ((soc_tg - soc_bl).astype(np.float64) / soc_bl.astype(np.float64)) * 100.0
    soc_chg[(soc_bl == NODATA_VALUE) | (soc_tg == NODATA_VALUE)] = NODATA_VALUE

    return np.reshape(soc_chg, shp)


@numba.jit(nopython=True)
@cc.export("calc_deg_soc", "i2[:,:](i2[:,:], i2[:,:], i2[:,:])")
def calc_deg_soc(soc_bl, soc_tg, water):
    """Optimized SOC degradation calculation with fewer intermediate arrays"""
    original_shape = soc_bl.shape

    # Work on flattened arrays
    soc_bl_flat = soc_bl.ravel()
    soc_tg_flat = soc_tg.ravel()
    water_flat = water.ravel()

    # Pre-allocate output array
    out = np.zeros(soc_bl_flat.shape[0], dtype=np.int16)

    # Calculate percent change efficiently - avoid intermediate array
    # Use mask to avoid division by zero
    valid_mask = (
        (soc_bl_flat != NODATA_VALUE[0])
        & (soc_tg_flat != NODATA_VALUE[0])
        & (soc_bl_flat != 0)
    )

    # Only calculate percent change for valid pixels
    for i in range(soc_bl_flat.shape[0]):
        if valid_mask[i]:
            pct_change = ((soc_tg_flat[i] - soc_bl_flat[i]) / soc_bl_flat[i]) * 100.0

            if pct_change >= -101.0 and pct_change <= -10.0:
                out[i] = -1  # Declining
            elif pct_change > -10.0 and pct_change < 10.0:
                out[i] = 0  # Stable
            elif pct_change >= 10.0:
                out[i] = 1  # Improving
        else:
            out[i] = NODATA_VALUE[0]

    # Apply water mask - water_flat is already boolean, no need to cast
    out[water_flat] = NODATA_VALUE[0]

    return out.reshape(original_shape)


@numba.jit(nopython=True)
@cc.export("calc_deg_lc", "i2[:,:](i2[:,:], i2[:,:], i2[:], i2[:], i4, i2[:], i2[:])")
def calc_deg_lc(lc_bl, lc_tg, trans_code, trans_meaning, multiplier):
    """calculate land cover degradation"""
    shp = lc_bl.shape
    trans = calc_lc_trans(lc_bl, lc_tg, multiplier)
    trans = trans.ravel()
    lc_bl = lc_bl.ravel()
    lc_tg = lc_tg.ravel()
    out = np.zeros(lc_bl.shape, dtype=np.int16)

    for code, meaning in zip(trans_code, trans_meaning):
        out[trans == code] = meaning
    out[np.logical_or(lc_bl == NODATA_VALUE, lc_tg == NODATA_VALUE)] = NODATA_VALUE

    return np.reshape(out, shp)


@numba.jit(nopython=True)
@cc.export("calc_deg_sdg", "i2[:,:](i2[:,:], i2[:,:], i2[:,:])")
def calc_deg_sdg(deg_prod3, deg_lc, deg_soc):
    """Optimized SDG degradation calculation"""
    original_shape = deg_prod3.shape

    # Work on flattened arrays for better performance
    deg_prod3_flat = deg_prod3.ravel()
    deg_lc_flat = deg_lc.ravel()
    deg_soc_flat = deg_soc.ravel()

    out = deg_prod3_flat.copy()

    # Vectorized operations for degradation logic
    # Degradation by either LC or SOC overrides productivity
    degradation_mask = (deg_lc_flat == -1) | (deg_soc_flat == -1)
    out[degradation_mask] = -1

    # Improvements by LC or SOC, but only if productivity is stable (0)
    # and no other indicator shows decline
    improvement_condition = (out == 0) & ((deg_lc_flat == 1) | (deg_soc_flat == 1))
    out[improvement_condition] = 1

    # Handle NODATA efficiently with single combined mask
    nodata_mask = (
        (deg_prod3_flat == NODATA_VALUE[0])
        | (deg_lc_flat == NODATA_VALUE[0])
        | (deg_soc_flat == NODATA_VALUE[0])
    )
    out[nodata_mask] = NODATA_VALUE[0]

    return out.reshape(original_shape)
