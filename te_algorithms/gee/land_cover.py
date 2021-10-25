from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from .util import TEImage
from te_schemas.schemas import BandInfo


def land_cover(
    year_initial,
    year_final,
    trans_matrix,
    nesting,
    additional_years,  # allows including years of lc outside of period
    logger
):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    # Land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2020")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # Initial and final land cover map reclassified to IPCC classes
    lc_bl = lc.select('y{}'.format(year_initial)).remap(
        nesting.get_list()[0], nesting.get_list()[1]
    )
    lc_tg = lc.select('y{}'.format(year_final)).remap(
        nesting.get_list()[0], nesting.get_list()[1]
    )

    # compute transition map (first digit for baseline land cover, and second 
    # digit for target year land cover)
    lc_tr = lc_bl.multiply(trans_matrix.get_multiplier()).add(lc_tg)

    # definition of land cover transitions as degradation (-1), improvement 
    # (1), or no relevant change (0)
    lc_dg = lc_tr.remap(
        trans_matrix.get_list()[0], trans_matrix.get_list()[1]
    ).rename('Land_cover_degradation')

    # Remap persistence classes so they are sequential. This
    # makes it easier to assign a clear color ramp in QGIS.
    lc_tr = lc_tr.remap(
        trans_matrix.get_persistence_list()[0],
        trans_matrix.get_persistence_list()[1]
    ).rename(f'Land_cover_transitions_{year_initial}-{year_final}')

    logger.debug("Setting up output.")
    lc_baseline_esa = lc.select('y{}'.format(year_initial)).rename(
            f'Land_cover_{year_initial}')
    lc_target_esa = lc.select('y{}'.format(year_final)).rename(
            f'Land_cover_{year_final}')
    out = TEImage(
        lc_dg.addBands(lc_baseline_esa).addBands(lc_target_esa).addBands(lc_tr),
        [
            BandInfo(
                "Land cover (degradation)",
                add_to_map=True,
                metadata={'year_initial': year_initial,
                          'year_final': year_final,
                          'trans_matrix': trans_matrix.dumps(),
                          'nesting': nesting.dumps()}
            ),
            BandInfo(
                "Land cover (ESA classes)",
                metadata={'year': year_initial}
            ),
            BandInfo(
                "Land cover (ESA classes)",
                metadata={'year': year_final}
            ),
            BandInfo(
                "Land cover transitions",
                add_to_map=True,
                metadata={'year_initial': year_initial,
                          'year_final': year_final,
                          'nesting': nesting.dumps()}
                )
        ]
    )

    # Return the full land cover timeseries so it is available for reporting
    logger.debug("Adding annual lc layers.")
    years = [*range(year_initial, year_final + 1)] + additional_years
    years = list(set(years))
    lc_remapped = lc.select('y{}'.format(years[0])).remap(
        nesting.get_list()[0], nesting.get_list()[1]
    )
    d_lc = [BandInfo(
        "Land cover (7 class)",
        add_to_map=True,  # because this is initial year
        metadata={'year': years[0],
                  'nesting': nesting.dumps()}
    )]
    for year in years[1:]:
        lc_remapped = lc_remapped.addBands(
            lc.select('y{}'.format(year)).remap(
                nesting.get_list()[0], nesting.get_list()[1]
            )
        )
        if year == year_final:
            add_to_map = True
        else:
            add_to_map = False
        d_lc.append(BandInfo("Land cover (7 class)",
                    add_to_map=add_to_map,
                    metadata={'year': year,
                              'nesting': nesting.dumps()}))
    lc_remapped = lc_remapped.rename([
        f'Land_cover_{year}' for year in years
    ])
    out.addBands(lc_remapped, d_lc)

    out.image = out.image.unmask(-32768).int16()

    logger.debug("Leaving land_cover function.")

    return out
