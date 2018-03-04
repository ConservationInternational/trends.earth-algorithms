from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation.util import TEImage
from landdegradation.schemas import BandInfo


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

    # Remap LC according to input matrix
    lc_remapped = lc.remap(remap_matrix[0], remap_matrix[1])

    ## target land cover map reclassified to IPCC 6 classes
    lc_tg = lc_remapped.select('y{}'.format(year_target))

    ## baseline land cover map reclassified to IPCC 6 classes
    lc_bl = lc_remapped.select('y{}'.format(year_baseline))

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

    # Return the full land cover timeseries so it is available for reporting
    lc_out_images = lc_remapped.select(ee.List.sequence(year_baseline - 1992, year_target- 1992, 1))
            
    for year in range(year_baseline, year_target + 1):
        d_lc = []
        if (year == year_baseline) or (year == year_target):
            add_to_map = True
        else:
            add_to_map = False
        d_lc.extend([BandInfo("Land cover (7 class)", add_to_map=add_to_map, metadata={'year': year})])

    out = TEImage(lc_out_images, d_lc)

    our.addBands(lc_tr.addBands(lc_dg).addBands(lc_bl_raw).addBands(lc_tg_raw),
                 [BandInfo("Land cover transitions", add_to_map=True, metadata={'year_baseline': year_baseline, 'year_target': year_target}),
                  BandInfo("Land cover degradation", add_to_map=True, metadata={'year_baseline': year_baseline, 'year_target': year_target}),
                  BandInfo("Land cover (ESA classes)", metadata={'year': year_baseline}),
                  BandInfo("Land cover (ESA classes)", metadata={'year': year_target})])

    out.image = out.image.unmask(-32768).int16()

    return out
