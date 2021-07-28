from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation.util import TEImage
from te_schemas.schemas import BandInfo


def land_cover(year_baseline, year_target, trans_matrix,
               nesting, EXECUTION_ID, logger):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    # Land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2020")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # Remap LC according to input matrix
    lc_remapped = lc.select('y{}'.format(year_baseline)).remap(nesting.get_list()[0], nesting.get_list()[1])
    for year in range(year_baseline + 1, year_target + 1):
        lc_remapped = lc_remapped.addBands(lc.select('y{}'.format(year)).remap(nesting.get_list()[0], nesting.get_list()[1]))

    # Target land cover map reclassified to IPCC 6 classes
    lc_bl = lc_remapped.select(0)

    # baseline land cover map reclassified to IPCC 6 classes
    lc_tg = lc_remapped.select(len(lc_remapped.getInfo()['bands']) - 1)

    # compute transition map (first digit for baseline land cover, and second 
    # digit for target year land cover)
    lc_tr = lc_bl.multiply(trans_matrix.get_multiplier()).add(lc_tg)

    # definition of land cover transitions as degradation (-1), improvement 
    # (1), or no relevant change (0)
    lc_dg = lc_tr.remap(trans_matrix.get_list()[0], trans_matrix.get_list()[1])

    # Remap persistence classes so they are sequential. This
    # makes it easier to assign a clear color ramp in QGIS.
    lc_tr = lc_tr.remap(trans_matrix.get_persistence_list()[0], 
                        trans_matrix.get_persistence_list()[1])

    logger.debug("Setting up output.")
    out = TEImage(
            lc_dg.addBands(lc.select('y{}'.format(year_baseline))).addBands(lc.select('y{}'.format(year_target))).addBands(lc_tr),
                  [BandInfo("Land cover (degradation)", add_to_map=True,
                            metadata={'year_baseline': year_baseline,
                                      'year_target': year_target,
                                      'trans_matrix': trans_matrix.dumps(),
                                      'nesting': nesting.dumps()}),
                   BandInfo("Land cover (ESA classes)",
                            metadata={'year': year_baseline}),
                   BandInfo("Land cover (ESA classes)",
                            metadata={'year': year_target}),
                   BandInfo("Land cover transitions", add_to_map=True,
                            metadata={'year_baseline': year_baseline,
                                      'year_target': year_target,
                                      'nesting': nesting.dumps()})])

    # Return the full land cover timeseries so it is available for reporting
    logger.debug("Adding annual lc layers.")
    d_lc = []
    for year in range(year_baseline, year_target + 1):
        if (year == year_baseline) or (year == year_target):
            add_to_map = True
        else:
            add_to_map = False
        d_lc.append(BandInfo("Land cover (7 class)",
            add_to_map=add_to_map,
            metadata={'year': year,
                      'nesting': nesting.dumps()}))
    out.addBands(lc_remapped, d_lc)

    out.image = out.image.unmask(-32768).int16()

    logger.debug("Leaving land_cover function.")

    return out
