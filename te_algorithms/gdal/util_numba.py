import logging

import numpy as np

logger = logging.getLogger(__name__)

# Ensure mask and nodata values are saved as 16 bit integers for raster compatibility
# but use int32 in numba functions to avoid overflow issues
NODATA_VALUE = np.array([-32768], dtype=np.int16)
MASK_VALUE = np.array([-32767], dtype=np.int16)

try:
    import numba
    from numba.pycc import CC

    cc = CC("util_numba")

    # Calculate the area of a slice of the globe from the equator to the parallel
    # at latitude f (on WGS84 ellipsoid). Based on:
    # https://gis.stackexchange.com/questions/127165/more-accurate-way-to-calculate-area-of-rasters
    @numba.jit(nopython=True, nogil=True)
    @cc.export("slice_area", "f8(f8)")
    def slice_area(f):
        a = 6378137.0  # in meters
        b = 6356752.3142  # in meters
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
    @numba.jit(nopython=True, nogil=True)
    @cc.export("calc_cell_area", "f8(f8, f8, f8)")
    def calc_cell_area(ymin, ymax, x_width):
        if ymin > ymax:
            temp = ymax
            ymax = ymin
            ymin = temp
        return (slice_area(np.deg2rad(ymax)) - slice_area(np.deg2rad(ymin))) * (
            x_width / 360.0
        )

except ImportError:

    def slice_area(f):
        a = 6378137.0  # in meters
        b = 6356752.3142  # in meters
        e = np.sqrt(1 - pow(b / a, 2))
        zp = 1 + e * np.sin(f)
        zm = 1 - e * np.sin(f)

        return (
            np.pi
            * pow(b, 2)
            * ((2 * np.arctanh(e * np.sin(f))) / (2 * e) + np.sin(f) / (zp * zm))
        )

    def calc_cell_area(ymin, ymax, x_width):
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        return (slice_area(np.deg2rad(ymax)) - slice_area(np.deg2rad(ymin))) * (
            x_width / 360.0
        )


# zonal_total, zonal_total_weighted, and bizonal_total all return dicts.
# Numba's typed.Dict boxing overhead cancels the loop speedup, so numpy
# vectorised implementations are used unconditionally regardless of whether
# numba is installed.


def zonal_total(z, d, mask):
    """
    Calculate zonal totals by summing data values within each zone.

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
    z = z.ravel().astype(np.int32)  # astype already creates a new array
    d = d.ravel().astype(np.float64)  # astype already creates a new array
    mask = mask.ravel()
    # Convert int16 constants to int32 for mask operations
    z[mask] = np.int32(MASK_VALUE[0])  # Convert to int32 for assignment
    d[d == NODATA_VALUE[0]] = 0  # Use explicit indexing and ignore nodata values
    keys, inverse = np.unique(z, return_inverse=True)
    sums = np.bincount(inverse, weights=d)

    return {int(k): float(v) for k, v in zip(keys, sums)}


def zonal_total_weighted(z, d, weights, mask):
    z = z.ravel().astype(np.int32)  # astype already creates a new array
    d = d.ravel().astype(np.float64)  # astype already creates a new array
    weights = weights.ravel().astype(np.float64)  # Ensure float64 type
    mask = mask.ravel()
    z[mask] = np.int32(MASK_VALUE[0])  # Convert int16 to int32 for assignment
    d[d == NODATA_VALUE[0]] = 0  # Use explicit indexing and ignore nodata values
    keys, inverse = np.unique(z, return_inverse=True)
    sums = np.bincount(inverse, weights=d * weights)

    return {int(k): float(v) for k, v in zip(keys, sums)}


def bizonal_total(z1, z2, d, mask):
    z1 = z1.ravel().astype(np.int64)  # astype already creates a new array
    z2 = z2.ravel().astype(np.int64)  # astype already creates a new array
    d = d.ravel().astype(np.float64)  # Ensure float64 type
    mask = mask.ravel()
    z1[mask] = np.int64(MASK_VALUE[0])  # Convert int16 to int64 for assignment
    z2[mask] = np.int64(MASK_VALUE[0])  # Convert int16 to int64 for assignment
    combined = z1 * (2**32) + (z2 + 2**31)
    keys, inverse = np.unique(combined, return_inverse=True)
    sums = np.bincount(inverse, weights=d)

    tab = {}
    for c, v in zip(keys, sums):
        z2_val = int(c % (2**32)) - 2**31
        z1_val = int((c - (z2_val + 2**31)) // (2**32))
        tab[(z1_val, z2_val)] = float(v)

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
