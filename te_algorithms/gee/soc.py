from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from .util import TEImage
from te_schemas.schemas import BandInfo


def soc(
    year_initial,
    year_final,
    fl,
    trans_matrix,
    nesting,
    dl_annual_lc,
    logger
):
    """Calculate SOC indicator."""
    logger.debug("Entering soc function.")

    # soc
    soc = ee.Image("users/geflanddegradation/toolbox_datasets/soc_sgrid_30cm")
    soc_t0 = soc.updateMask(soc.neq(-32768))

    # land cover - note it needs to be reprojected to match soc so that it can 
    # be output to cloud storage in the same stack
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2020") \
            .select(ee.List.sequence(year_initial - 1992, year_final - 1992, 1)) \
            .reproject(crs=soc.projection())
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    if fl == 'per pixel':
        # Setup a raster of climate regimes to use for coding Fl automatically
        climate = ee.Image("users/geflanddegradation/toolbox_datasets/ipcc_climate_zones") \
            .remap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                   [0, 2, 1, 2, 1, 2, 1, 2, 1, 5, 4, 4, 3])
        clim_fl = climate.remap([0, 1, 2, 3, 4, 5],
                                [0, 0.8, 0.69, 0.58, 0.48, 0.64])
    # create empty stacks to store annual land cover maps
    stack_lc  = ee.Image().select()

    # create empty stacks to store annual soc maps
    stack_soc = ee.Image().select()

    # loop through all the years in the period of analysis to compute changes in SOC
    soc_t0 = 2000
    for k in range(year_final - soc_t0):
        # land cover map reclassified to UNCCD 7 classes (1: forest, 2: 
        # grassland, 3: cropland, 4: wetland, 5: artifitial, 6: bare, 7: water)
        lc_t0 = lc.select(k).remap(nesting.get_list()[0], nesting.get_list()[1])

        lc_t1 = lc.select(k + 1).remap(nesting.get_list()[0], nesting.get_list()[1])

        if (k == 0):
            # compute transition map (first digit for baseline land cover, and 
            # second digit for target year land cover) 
            lc_tr = lc_t0.multiply(trans_matrix.get_multiplier()).add(lc_t1)
          
            # compute raster to register years since transition
            tr_time = ee.Image(2).where(lc_t0.neq(lc_t1), 1)
        else:
            # Update time since last transition. Add 1 if land cover remains 
            # constant, and reset to 1 if land cover changed.
            tr_time = tr_time.where(lc_t0.eq(lc_t1), tr_time.add(ee.Image(1))) \
                .where(lc_t0.neq(lc_t1), ee.Image(1))
                               
            # compute transition map (first digit for baseline land cover, and 
            # second digit for target year land cover), but only update where 
            # changes actually ocurred.
            lc_tr_temp = lc_t0.multiply(10).add(lc_t1)
            lc_tr = lc_tr.where(lc_t0.neq(lc_t1), lc_tr_temp)

        # stock change factor for land use - note the 99 and -99 will be 
        # recoded using the chosen Fl option
        lc_tr_fl_0 = lc_tr.remap([11, 12, 13, 14, 15, 16, 17,
                                  21, 22, 23, 24, 25, 26, 27,
                                  31, 32, 33, 34, 35, 36, 37,
                                  41, 42, 43, 44, 45, 46, 47,
                                  51, 52, 53, 54, 55, 56, 57,
                                  61, 62, 63, 64, 65, 66, 67,
                                  71, 72, 73, 74, 75, 76, 77],
                                 [1,     1,    99,        1, 0.1, 0.1, 1,
                                  1,     1,    99,        1, 0.1, 0.1, 1,
                                  -99, -99,     1, 1 / 0.71, 0.1, 0.1, 1,
                                  1,      1, 0.71,        1, 0.1, 0.1, 1,
                                  2,      2,    2,        2,   1,   1, 1,
                                  2,      2,    2,        2,   1,   1, 1,
                                  1,      1,    1,        1,   1,   1, 1])

        if fl == 'per pixel':
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), clim_fl)\
                                 .where(lc_tr_fl_0.eq(-99), ee.Image(1).divide(clim_fl))
        else:
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), fl)\
                                 .where(lc_tr_fl_0.eq(-99), ee.Image(1).divide(fl))

        # stock change factor for management regime
        lc_tr_fm = lc_tr.remap([11, 12, 13, 14, 15, 16, 17,
                                21, 22, 23, 24, 25, 26, 27,
                                31, 32, 33, 34, 35, 36, 37,
                                41, 42, 43, 44, 45, 46, 47,
                                51, 52, 53, 54, 55, 56, 57,
                                61, 62, 63, 64, 65, 66, 67,
                                71, 72, 73, 74, 75, 76, 77],
                               [1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1])

        # stock change factor for input of organic matter
        lc_tr_fo = lc_tr.remap([11, 12, 13, 14, 15, 16, 17,
                                21, 22, 23, 24, 25, 26, 27,
                                31, 32, 33, 34, 35, 36, 37,
                                41, 42, 43, 44, 45, 46, 47,
                                51, 52, 53, 54, 55, 56, 57,
                                61, 62, 63, 64, 65, 66, 67,
                                71, 72, 73, 74, 75, 76, 77],
                               [1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1])

        if (k == 0):
            soc_chg = (soc_t0.subtract((soc_t0.multiply(lc_tr_fl).multiply(lc_tr_fm).multiply(lc_tr_fo)))).divide(20)
          
            # compute final SOC stock for the period
            soc_t1 = soc_t0.subtract(soc_chg)
            
            # add to land cover and soc to stacks from both dates for the first 
            # period
            stack_lc = stack_lc.addBands(lc_t0).addBands(lc_t1)
            stack_soc = stack_soc.addBands(soc_t0).addBands(soc_t1)

        else:
            # compute annual change in soc (updates from previous period based 
            # on transition and time <20 years)
            soc_chg = (soc_chg.where(lc_t0.neq(lc_t1),
                                     (stack_soc.select(k).subtract(stack_soc.select(k)
                                                                   .multiply(lc_tr_fl)
                                                                   .multiply(lc_tr_fm) 
                                                                   .multiply(lc_tr_fo))).divide(20))
                       .where(tr_time.gt(20), 0))
          
            # compute final SOC for the period
            socn = stack_soc.select(k).subtract(soc_chg)
          
            # add land cover and soc to stacks only for the last year in the 
            # period
            stack_lc = stack_lc.addBands(lc_t1)
            stack_soc = stack_soc.addBands(socn)

    # compute soc percent change for the analysis period
    soc_pch = (
        (
            (
                stack_soc.select(year_final - soc_t0)
                .subtract(year_initial - soc_t0),
            ).divide(year_initial - soc_t0)
        ).multiply(100)
    ).rename(
        f'Percent_SOC_increase_{year_initial}-{year_final}'
    )

    logger.debug("Setting up output.")
    out = TEImage(soc_pch,
                  [BandInfo("Soil organic carbon (degradation)", add_to_map=True,
                   metadata={'year_initial': year_initial,
                             'year_final': year_final,
                             'trans_matrix': trans_matrix.dumps(),
                             'nesting': nesting.dumps()})])

    logger.debug("Adding annual SOC layers.")
    # Output annual SOC layers
    d_soc = []
    soc_stack_out = stack_soc.select(year_initial)
    years = [*range(year_initial, year_final + 1)]
    for year in years[1:]:
        soc_stack_out = soc_stack_out.addBands(stack_soc.select(year))
        if (year == year_initial) or (year == year_final):
            add_to_map = True
        else:
            add_to_map = False
        d_soc.append(
            BandInfo(
                "Soil organic carbon",
                add_to_map=add_to_map,
                metadata={'year': year}
            )
        )
    soc_stack_out = soc_stack_out.rename([
        f'SOC_{year}' for year in years
    ])
    out.addBands(soc_stack_out, d_soc)

    if dl_annual_lc:
        logger.debug("Adding all annual LC layers.")
        d_lc = []
        for year in range(year_initial, year_final + 1):
            d_lc.append(BandInfo("Land cover (7 class)",
                                 metadata={'year': year,
                                           'nesting': nesting.dumps()}))
        stack_lc = stack_lc.rename([
            f'Land_cover_{year}'
            for year in range(year_initial, year_final + 1)
        ])
        out.addBands(stack_lc, d_lc)
    else:
        logger.debug("Adding initial and final LC layers.")
        lc_initial = stack_lc.select(0).rename(f'Land_cover_{year_initial}')
        lc_final = stack_lc.select(
            len(stack_lc.getInfo()['bands']) - 1
        ).rename(f'Land_cover_{year_final}')
        out.addBands(
            lc_initial.addBands(lc_final),
            [
                BandInfo(
                    "Land cover (7 class)",
                    metadata={'year': year_initial,
                              'nesting': nesting.dumps()}
                ),
                BandInfo(
                    "Land cover (7 class)",
                    metadata={'year': year_final,
                              'nesting': nesting.dumps()}
                )
            ]
        )

    out.image = out.image.unmask(-32768).int16()

    return out
