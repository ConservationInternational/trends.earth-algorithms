import logging

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
    cc = CC("util_numba")

logger = logging.getLogger(__name__)

# Ensure mask and nodata values are saved as 16 bit integers for raster compatibility
# but use int32 in numba functions to avoid overflow issues
NODATA_VALUE = np.array([-32768], dtype=np.int16)
MASK_VALUE = np.array([-32767], dtype=np.int16)


# Calculate the area of a slice of the globe from the equator to the parallel
# at latitude f (on WGS84 ellipsoid). Based on:
# https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
@numba.jit(nopython=True)
@cc.export("slice_area", "f8(f8)")
def slice_area(f):
    a = 6378137.0  # in meters
    b = 6356752.3142  # in meters,
    e = np.sqrt(1 - pow(b / a, 2))
    zp = 1 + e * np.sin(f)
    zm = 1 - e * np.sin(f)

    return (
        np.pi
        * pow(b, 2)
        * ((2 * np.arctanh(e * np.sin(f))) / (2 * e) + np.sin(f) / (zp * zm))
    )


# Formula to calculate area of a raster cell on WGS84 ellipsoid, following
# https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
@numba.jit(nopython=True)
@cc.export("calc_cell_area", "f8(f8, f8, f8)")
def calc_cell_area(ymin, ymax, x_width):
    if ymin > ymax:
        temp = 0
        temp = ymax
        ymax = ymin
        ymin = temp
    # ymin: minimum latitude
    # ymax: maximum latitude
    # x_width: width of cell in degrees

    return (slice_area(np.deg2rad(ymax)) - slice_area(np.deg2rad(ymin))) * (
        x_width / 360.0
    )


@numba.jit(nopython=True)
@cc.export("zonal_total", "DictType(i4, f8)(i4[:,:], f8[:,:], b1[:,:])")
def zonal_total(z, d, mask):
    """
    Calculate zonal totals by summing data values within each zone.

    This function sums the values in the data array (d) for each unique zone
    identified in the zone array (z). It is typically used for calculating
    area totals where the data array contains area values (cell_areas) and
    the zone array contains class values (-1, 0, 1 for SDG indicators).

    For 7-class status maps, use zonal_status_total() instead which counts
    areas for each status class rather than summing the status values.

    Args:
        z: 2D array with zone identifiers (e.g., SDG classes -1, 0, 1)
        d: 2D array with data values to sum (e.g., cell areas in sq km)
        mask: 2D boolean mask array (True = masked/excluded pixels)

    Returns:
        Dictionary mapping zone identifier (int) to total summed value (float)

    Example:
        For SDG indicators with z=[-1,0,1] and d=[cell_areas], returns:
        {-1: total_degraded_area, 0: total_stable_area, 1: total_improved_area}
    """
    # Use int32 to avoid overflow issues with int16
    z = z.copy().ravel().astype(np.int32)  # Use int32 instead of int16
    d = d.copy().ravel().astype(np.float64)  # Ensure float64 type
    mask = mask.ravel()
    # Convert int16 constants to int32 for mask operations
    z[mask] = np.int32(MASK_VALUE[0])  # Convert to int32 for assignment
    d[d == NODATA_VALUE[0]] = 0  # Use explicit indexing and ignore nodata values
    # Use regular dict and let NumBA infer types from consistent usage
    totals = dict()

    for i in range(z.shape[0]):
        zone_key = z[i]  # int32 now to avoid overflow
        if zone_key not in totals:
            totals[zone_key] = d[i]  # Already float64 from astype above
        else:
            totals[zone_key] += d[i]

    return totals


@numba.jit(nopython=True)
@cc.export(
    "zonal_total_weighted", "DictType(i4, f8)(i4[:,:], i4[:,:], f8[:,:], b1[:,:])"
)
def zonal_total_weighted(z, d, weights, mask):
    z = z.copy().ravel().astype(np.int32)  # Use int32 instead of int16
    d = d.copy().ravel().astype(np.float64)  # Ensure float64 type
    weights = weights.ravel().astype(np.float64)  # Ensure float64 type
    mask = mask.ravel()
    z[mask] = np.int32(MASK_VALUE[0])  # Convert int16 to int32 for assignment
    d[d == NODATA_VALUE[0]] = 0  # Use explicit indexing and ignore nodata values
    # Use regular dict and let NumBA infer types from consistent usage
    totals = dict()

    for i in range(z.shape[0]):
        zone_key = z[i]  # int32 now to avoid overflow
        if zone_key not in totals:
            totals[zone_key] = (
                d[i] * weights[i]
            )  # Both already float64 from astype above
        else:
            totals[zone_key] += d[i] * weights[i]

    return totals


@numba.jit(nopython=True)
@cc.export(
    "bizonal_total", "DictType(UniTuple(i4, 2), f8)(i4[:,:], i4[:,:], f8[:,:], b1[:,:])"
)
def bizonal_total(z1, z2, d, mask):
    z1 = z1.copy().ravel().astype(np.int32)  # Use int32 instead of int16
    z2 = z2.copy().ravel().astype(np.int32)  # Use int32 instead of int16
    d = d.ravel().astype(np.float64)  # Ensure float64 type
    mask = mask.ravel()
    z1[mask] = np.int32(MASK_VALUE[0])  # Convert int16 to int32 for assignment
    z2[mask] = np.int32(MASK_VALUE[0])  # Convert int16 to int32 for assignment
    # Use regular dict and let NumBA infer types from consistent usage
    tab = dict()

    for i in range(z1.shape[0]):
        # Ensure both elements are consistently int32
        key = (z1[i], z2[i])  # Both are int32 now

        if key not in tab:
            tab[key] = d[i]  # Already float64 from astype above
        else:
            tab[key] += d[i]

    return tab


def _accumulate_dicts(z):
    out = z[0].copy()

    for d in z[1:]:
        _combine_dicts(out, d)

    return out


def _combine_dicts(z1, z2):
    out = z1

    for key in z2:
        if key in out:
            out[key] += z2[key]
        else:
            out[key] = z2[key]

    return out


# Numba compiled functions return numba types which won't pickle correctly
# (which is needed for multiprocessing), so cast them to regular python types
def cast_numba_int_dict_list_to_cpython(dict_list):
    return [cast_numba_int_dict_to_cpython(dictionary) for dictionary in dict_list]


def cast_numba_int_dict_to_cpython(dictionary):
    return {int(key): float(value) for key, value in dictionary.items()}


if __name__ == "__main__":
    cc.compile()
