from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation.util import TEImage
from landdegradation.schemas.schemas import BandInfo


def land_cover(year_baseline, year_target, trans_matrix,
               remap_matrix, EXECUTION_ID, logger):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    ## land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2019")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # Remap LC according to input matrix
    lc_remapped = lc.select('y{}'.format(year_baseline)).remap(remap_matrix[0], remap_matrix[1])
    for year in range(year_baseline + 1, year_target + 1):
        lc_remapped = lc_remapped.addBands(lc.select('y{}'.format(year)).remap(remap_matrix[0], remap_matrix[1]))

    ## target land cover map reclassified to IPCC 6 classes
    lc_bl = lc_remapped.select(0)

    ## baseline land cover map reclassified to IPCC 6 classes
    lc_tg = lc_remapped.select(len(lc_remapped.getInfo()['bands']) - 1)

    ## compute transition map (first digit for baseline land cover, and second digit for target year land cover)
    lc_tr = lc_bl.multiply(10).add(lc_tg)

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

    logger.debug("Setting up output.")
    out = TEImage(lc_dg.addBands(lc.select('y{}'.format(year_baseline))).addBands(lc.select('y{}'.format(year_target))).addBands(lc_tr),
                  [BandInfo("Land cover (degradation)", add_to_map=True, metadata={'year_baseline': year_baseline, 'year_target': year_target}),
                   BandInfo("Land cover (ESA classes)", metadata={'year': year_baseline}),
                   BandInfo("Land cover (ESA classes)", metadata={'year': year_target}),
                   BandInfo("Land cover transitions", add_to_map=True, metadata={'year_baseline': year_baseline, 'year_target': year_target})])

    # Return the full land cover timeseries so it is available for reporting
    logger.debug("Adding annual lc layers.")
    d_lc = []
    for year in range(year_baseline, year_target + 1):
        if (year == year_baseline) or (year == year_target):
            add_to_map = True
        else:
            add_to_map = False
        d_lc.append(BandInfo("Land cover (7 class)", add_to_map=add_to_map, metadata={'year': year}))
    out.addBands(lc_remapped, d_lc)

    out.image = out.image.unmask(-32768).int16()

    return out
