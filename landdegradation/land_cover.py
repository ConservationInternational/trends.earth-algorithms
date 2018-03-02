from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation import GEEIOError


def land_cover(year_baseline, year_target, geojson, trans_matrix,
               remap_matrix, EXECUTION_ID, logger):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    ## land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2015")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    ## target land cover map reclassified to IPCC 6 classes
    lc_tg_raw = lc.select('y{}'.format(year_target))
    lc_tg_remapped = lc_tg_raw.remap(remap_matrix[0], remap_matrix[1])

    ## baseline land cover map reclassified to IPCC 6 classes
    lc_bl_raw = lc.select('y{}'.format(year_baseline))
    lc_bl_remapped = lc_bl_raw.remap(remap_matrix[0], remap_matrix[1])

    ## compute transition map (first digit for baseline land cover, and second digit for target year land cover)
    lc_tr = lc_bl_remapped.multiply(10).add(lc_tg_remapped)

    ## definition of land cover transitions as degradation (-1), improvement (1), or no relevant change (0)
    lc_dg = lc_tr.remap([11, 12, 13, 14, 15, 16, 17,
                         21, 22, 23, 24, 25, 26, 27,
                         31, 32, 33, 34, 35, 36, 37,
                         41, 42, 43, 44, 45, 46, 47,
                         51, 52, 53, 54, 55, 56, 57,
                         61, 62, 63, 64, 65, 66, 67,
                         71, 72, 73, 74, 75, 76, 77],
                        trans_matrix)

    ## Remap persistence classes so they are sequential. This
    ## makes it easier to assign a clear color ramp in QGIS.
    lc_tr = lc_tr.remap([11, 12, 13, 14, 15, 16, 17,
                         21, 22, 23, 24, 25, 26, 27,
                         31, 32, 33, 34, 35, 36, 37,
                         41, 42, 43, 44, 45, 46, 47,
                         51, 52, 53, 54, 55, 56, 57,
                         61, 62, 63, 64, 65, 66, 67,
                         71, 72, 73, 74, 75, 76, 77],
                        [1, 12, 13, 14, 15, 16, 17,
                         21, 2, 23, 24, 25, 26, 27,
                         31, 32, 3, 34, 35, 36, 37,
                         41, 42, 43, 4, 45, 46, 47,
                         51, 52, 53, 54, 5, 56, 57,
                         61, 62, 63, 64, 65, 6, 67,
                         71, 72, 73, 74, 75, 76, 7])

    lc_out = lc_bl_remapped \
        .addBands(lc_tg_remapped) \
        .addBands(lc_tr) \
        .addBands(lc_dg) \
        .addBands(lc_bl_raw) \
        .addBands(lc_tg_raw)

    return lc_out


