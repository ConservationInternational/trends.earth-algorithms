"""Numba JIT-compiled functions for LDN counterbalancing assessment.

Implements Steps 4-5 of the GPG Addendum counterbalancing procedure:
  Step 4: Calculate ΔᵢLDN = Aᵢgains − Aᵢlosses per land type
  Step 5: Assess LDN per land type (ΔᵢLDN ≥ 0 → achieved)

The 7-class expanded status matrix (from sdg_status_expanded()) encodes:
  1 = Persistent degradation   (baseline degraded, reporting degraded)
  2 = Recent degradation        (baseline stable/improved, reporting degraded)
  3 = Baseline degradation      (baseline degraded, reporting stable)
  4 = Stable                    (baseline stable, reporting stable)
  5 = Baseline improvement      (baseline improved, reporting stable)
  6 = Recent improvement        (baseline stable/degraded, reporting improved)
  7 = Persistent improvement    (baseline improved, reporting improved)

For counterbalancing:
  Gains  = status classes 6, 7  (improvement in the reporting period)
  Losses = status classes 1, 2  (degradation in the reporting period)
  Neutral = status classes 3, 4, 5 (no change in the reporting period)
"""

import numpy as np

try:
    import numba
    from numba.pycc import CC
except ImportError:
    HAVE_NUMBA = False

    class DecoratorSubstitute:
        def export(*args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

        def jit(*args, **kwargs):
            def wrapper(func):
                return func

            return wrapper

    cc = DecoratorSubstitute()
    numba = DecoratorSubstitute()
else:
    HAVE_NUMBA = True
    cc = CC("counterbalancing_numba")

NODATA_VALUE = np.array([-32768], dtype=np.int16)
MASK_VALUE = np.array([-32767], dtype=np.int16)

# Gain/loss/neutral codes for the output raster
GAIN_CODE = np.int16(1)
LOSS_CODE = np.int16(-1)
NEUTRAL_CODE = np.int16(0)


@numba.jit(nopython=True, nogil=True)
@cc.export("classify_gains_losses", "i2[:,:](i2[:,:], b1[:,:])")
def _classify_gains_losses_numba(status_7class, mask):
    """Classify each pixel as gain, loss, or neutral from 7-class status.

    Args:
        status_7class: 2D array of expanded status values (1-7).
        mask: 2D boolean array (True = outside AOI, excluded).

    Returns:
        2D int16 array: 1=gain, -1=loss, 0=neutral, NODATA for
        nodata/masked pixels.
    """
    original_shape = status_7class.shape
    status_flat = status_7class.ravel()
    mask_flat = mask.ravel()

    out = np.full(status_flat.shape[0], NODATA_VALUE[0], dtype=np.int16)

    for i in range(status_flat.shape[0]):
        if mask_flat[i]:
            continue
        val = status_flat[i]
        if val == NODATA_VALUE[0] or val == MASK_VALUE[0]:
            continue
        if val == 1 or val == 2:
            out[i] = LOSS_CODE
        elif val == 6 or val == 7:
            out[i] = GAIN_CODE
        elif val >= 3 and val <= 5:
            out[i] = NEUTRAL_CODE

    return out.reshape(original_shape)


@numba.jit(nopython=True, nogil=True)
@cc.export(
    "zonal_gains_losses",
    "Tuple((DictType(i4, f8), DictType(i4, f8)))(i2[:,:], i4[:,:], f8[:,:], b1[:,:])",
)
def _zonal_gains_losses_numba(status_7class, land_type, cell_area, mask):
    """Accumulate gain and loss areas per land type.

    Implements GPG Addendum Step 4: for each land type, sum the area of
    pixels classified as gains (status 6, 7) and losses (status 1, 2).

    Args:
        status_7class: 2D int16 array of expanded status values (1-7).
        land_type: 2D int32 array of land type class codes.
        cell_area: 2D float64 array of pixel areas (sq km).
        mask: 2D boolean mask (True = masked/excluded).

    Returns:
        Tuple of two dicts:
          gains: land_type_code → total gain area (sq km)
          losses: land_type_code → total loss area (sq km)
    """
    status_flat = status_7class.ravel().astype(np.int32)
    lt_flat = land_type.ravel().astype(np.int32)
    area_flat = cell_area.ravel().astype(np.float64)
    mask_flat = mask.ravel()

    # Mask out nodata
    status_flat[mask_flat] = np.int32(MASK_VALUE[0])

    gains = dict()
    losses = dict()

    for i in range(status_flat.shape[0]):
        s = status_flat[i]
        if s == np.int32(MASK_VALUE[0]) or s == np.int32(NODATA_VALUE[0]):
            continue
        lt = lt_flat[i]
        if lt == np.int32(NODATA_VALUE[0]):
            continue

        area = area_flat[i]

        if s == 6 or s == 7:
            if lt not in gains:
                gains[lt] = area
            else:
                gains[lt] += area
        elif s == 1 or s == 2:
            if lt not in losses:
                losses[lt] = area
            else:
                losses[lt] += area

    return gains, losses


@numba.jit(nopython=True, nogil=True)
@cc.export(
    "zonal_class_breakdown",
    "DictType(UniTuple(i4, 2), f8)(i2[:,:], i4[:,:], f8[:,:], b1[:,:])",
)
def _zonal_class_breakdown_numba(class_band, land_type, cell_area, mask):
    """Accumulate area by (land_type, class_value) for any categorical band.

    Only nodata and masked pixels are skipped.

    Args:
        class_band: 2D int16 array of class values.
        land_type: 2D int32 array of land type codes.
        cell_area: 2D float64 array of pixel areas (sq km).
        mask: 2D boolean mask.

    Returns:
        Dict of (land_type_code, class_value) → area (sq km).
    """
    class_flat = class_band.ravel().astype(np.int32)
    lt_flat = land_type.ravel().astype(np.int32)
    area_flat = cell_area.ravel().astype(np.float64)
    mask_flat = mask.ravel()

    class_flat[mask_flat] = np.int32(MASK_VALUE[0])

    breakdown = dict()

    for i in range(class_flat.shape[0]):
        c = class_flat[i]
        if c == np.int32(MASK_VALUE[0]) or c == np.int32(NODATA_VALUE[0]):
            continue
        lt = lt_flat[i]
        if lt == np.int32(NODATA_VALUE[0]):
            continue

        key = (lt, c)
        if key not in breakdown:
            breakdown[key] = area_flat[i]
        else:
            breakdown[key] += area_flat[i]

    return breakdown


# Vectorized equivalents of the loops above. Without numba those loops run as
# pure Python, which is far slower and holds the GIL


def _sum_by_key(keys, weights):
    if keys.size == 0:
        return {}
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse.ravel(), weights=weights)
    return {int(k): float(v) for k, v in zip(unique_keys, sums)}


def _classify_gains_losses_numpy(status_7class, mask):
    valid = ~mask
    s = status_7class
    out = np.full(s.shape, NODATA_VALUE[0], dtype=np.int16)
    out[valid & ((s == 1) | (s == 2))] = LOSS_CODE
    out[valid & ((s == 6) | (s == 7))] = GAIN_CODE
    out[valid & (s >= 3) & (s <= 5)] = NEUTRAL_CODE
    return out


def _zonal_gains_losses_numpy(status_7class, land_type, cell_area, mask):
    s = status_7class
    lt = land_type.astype(np.int32, copy=False)
    area = cell_area.astype(np.float64, copy=False)
    valid = ~mask & (lt != np.int32(NODATA_VALUE[0]))
    gain = valid & ((s == 6) | (s == 7))
    loss = valid & ((s == 1) | (s == 2))
    return _sum_by_key(lt[gain], area[gain]), _sum_by_key(lt[loss], area[loss])


def _zonal_class_breakdown_numpy(class_band, land_type, cell_area, mask):
    c = class_band.astype(np.int32)
    lt = land_type.astype(np.int32, copy=False)
    valid = (
        ~mask
        & (c != np.int32(MASK_VALUE[0]))
        & (c != np.int32(NODATA_VALUE[0]))
        & (lt != np.int32(NODATA_VALUE[0]))
    )
    # Pack (land_type, class) into one int64: high 32 bits = lt, low = class
    packed = (lt[valid].astype(np.int64) << 32) | (
        c[valid].astype(np.int64) & 0xFFFFFFFF
    )
    sums = _sum_by_key(packed, cell_area[valid].astype(np.float64, copy=False))
    return {
        (k >> 32, int(np.uint32(k & 0xFFFFFFFF).view(np.int32))): v
        for k, v in sums.items()
    }


if HAVE_NUMBA:
    classify_gains_losses = _classify_gains_losses_numba
    zonal_gains_losses = _zonal_gains_losses_numba
    zonal_class_breakdown = _zonal_class_breakdown_numba
else:
    classify_gains_losses = _classify_gains_losses_numpy
    zonal_gains_losses = _zonal_gains_losses_numpy
    zonal_class_breakdown = _zonal_class_breakdown_numpy


if __name__ == "__main__":
    cc.compile()
