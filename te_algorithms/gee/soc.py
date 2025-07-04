import ee
from te_schemas.schemas import BandInfo

from ..common.soc import trans_factors_for_custom_legend
from .util import TEImage

# stock change factor for management regime
# fmt: off
SOC_CHANGE_FACTOR_FOR_MANAGEMENT = (
    [
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47,
        51, 52, 53, 54, 55, 56, 57,
        61, 62, 63, 64, 65, 66, 67,
        71, 72, 73, 74, 75, 76, 77,
    ],
    [
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ],
)

# stock change factor for input of organic matter
SOC_CHANGE_FACTOR_FOR_ORGANIC_MATTER = (
    [
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47,
        51, 52, 53, 54, 55, 56, 57,
        61, 62, 63, 64, 65, 66, 67,
        71, 72, 73, 74, 75, 76, 77,
    ],
    [
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ],
)

# stock change factor for land use - note the 99 and -99 will be
# recoded using the chosen Fl option
SOC_CHANGE_FACTOR_FOR_LAND_USE = (
    [
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27,
        31, 32, 33, 34, 35, 36, 37,
        41, 42, 43, 44, 45, 46, 47,
        51, 52, 53, 54, 55, 56, 57,
        61, 62, 63, 64, 65, 66, 67,
        71, 72, 73, 74, 75, 76, 77,
    ],
    [
        1, 1, 99, 1, 0.1, 0.1, 1,
        1, 1, 99, 1, 0.1, 0.1, 1,
        -99, -99, 1, 1 / 0.71, 0.1, 0.1, 1,
        1, 1, 0.71, 1, 0.1, 0.1, 1,
        2, 2, 2, 2, 1, 1, 1,
        2, 2, 2, 2, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ],
)
# fmt: on


def _select_lc(soc_t0_year, lc_band0_year, year_final, fake_data, logger):
    """
    Get the indices for the land cover bands.

    This function is used to get the indices for the land cover bands in the
    land cover image stack. It allows for repeating the final land cover layer
    if fake_data is set to True and the year_final is outside of the range of
    the land cover image stack.
    """
    if not fake_data:
        img = ee.Image(
            "users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022"
        ).select(
            list(range(soc_t0_year - lc_band0_year, year_final - lc_band0_year + 1))
        )
    else:
        lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022")
        img = lc.select(soc_t0_year - lc_band0_year)
        for index in range(
            soc_t0_year - lc_band0_year + 1, year_final - lc_band0_year + 1
        ):
            # Can only go up to index 30 as there are 31 bands (1992-2022), and zero-based indexing
            if index <= 30 or not fake_data:
                img = img.addBands(lc.select(index))
            else:
                logger.warn(
                    f"Could not select year {lc_band0_year + index} from land cover asset "
                    "for SOC calculations. Returning data from 2022."
                )
                img = img.addBands(lc.select(30).rename(f"y{index + lc_band0_year}"))
        logger.info(f"_select_lc img band names are: {img.bandNames().getInfo()}")
    return img


def soc(
    year_initial,
    year_final,
    fl,
    esa_to_custom_nesting,  # defines how ESA nests to custom classes
    ipcc_nesting,  # defines how custom classes nest to IPCC
    annual_lc,
    annual_soc,
    logger,
    fake_data=False,  # return data from closest available year if year is outside of range
):
    """Calculate SOC indicator."""
    logger.debug("Entering soc function.")

    # soc
    soc = ee.Image("users/geflanddegradation/toolbox_datasets/soc_sgrid_30cm")
    soc_t0 = soc.updateMask(soc.neq(-32768))

    # Reference all SOC calculations to the year 2000
    soc_t0_year = 2000

    # First band in the LC layer stack is 1992
    lc_band0_year = 1992

    # land cover - note it needs to be reprojected to match soc so that it can
    # be output to cloud storage in the same stack
    lc = _select_lc(
        soc_t0_year, lc_band0_year, year_final, fake_data, logger
    ).reproject(crs=soc.projection())
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    if fl == "per pixel":
        # Setup a raster of climate regimes to use for coding Fl automatically
        climate = ee.Image(
            "users/geflanddegradation/toolbox_datasets/ipcc_climate_zones"
        ).remap(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 2, 1, 2, 1, 2, 1, 2, 1, 5, 4, 4, 3],
        )  # yapf: disable
        clim_fl = climate.remap(
            [0, 1, 2, 3, 4, 5], [0, 0.8, 0.69, 0.58, 0.48, 0.64]
        )  # yapf: disable
    # create empty stacks to store annual land cover maps
    stack_lc = ee.Image().select()

    # create empty stacks to store annual soc maps
    stack_soc = ee.Image().select()

    # loop through all the years in the period of analysis to compute changes in SOC

    class_codes = sorted([c.code for c in esa_to_custom_nesting.parent.key])
    class_positions = [*range(1, len(class_codes) + 1)]
    for k in range(year_final - soc_t0_year):
        logger.info(
            f"Comparing lc band {k} with band {k + 1} for years {soc_t0_year + k} and {soc_t0_year + k + 1}."
        )
        # land cover map reclassified to custom classes (1: forest, 2:
        # grassland, 3: cropland, 4: wetland, 5: artifitial, 6: bare, 7: water)
        lc_t0_orig_coding = lc.select(k).remap(
            esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
        )
        lc_t0 = lc_t0_orig_coding.remap(class_codes, class_positions)

        lc_t1_orig_coding = lc.select(k + 1).remap(
            esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
        )
        lc_t1 = lc_t1_orig_coding.remap(class_codes, class_positions)

        if k == 0:
            # compute transition map (first digit for baseline land cover, and
            # second digit for target year land cover)
            lc_tr = lc_t0.multiply(esa_to_custom_nesting.parent.get_multiplier()).add(
                lc_t1
            )

            # compute raster to register years since transition
            tr_time = ee.Image(2).where(lc_t0.neq(lc_t1), 1)
        else:
            # Update time since last transition. Add 1 if land cover remains
            # constant, and reset to 1 if land cover changed.
            tr_time = tr_time.where(lc_t0.eq(lc_t1), tr_time.add(ee.Image(1))).where(
                lc_t0.neq(lc_t1), ee.Image(1)
            )

            # compute transition map (first digit for baseline land cover, and
            # second digit for target year land cover), but only update where
            # changes actually ocurred.
            lc_tr_temp = lc_t0.multiply(10).add(lc_t1)
            lc_tr = lc_tr.where(lc_t0.neq(lc_t1), lc_tr_temp)

        # Convert lc_tr_fl_0 (defined against IPCC legend)
        soc_change_factor_for_land_use = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_LAND_USE, ipcc_nesting
        )
        lc_tr_fl_0 = lc_tr.remap(*soc_change_factor_for_land_use)

        if fl == "per pixel":
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), clim_fl).where(
                lc_tr_fl_0.eq(-99), ee.Image(1).divide(clim_fl)
            )
        else:
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), fl).where(
                lc_tr_fl_0.eq(-99), ee.Image(1).divide(fl)
            )

        soc_change_factor_for_management = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_MANAGEMENT, ipcc_nesting
        )
        lc_tr_fm = lc_tr.remap(*soc_change_factor_for_management)

        soc_change_factor_for_organic_matter = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_ORGANIC_MATTER, ipcc_nesting
        )
        lc_tr_fo = lc_tr.remap(*soc_change_factor_for_organic_matter)

        if k == 0:
            soc_chg = (
                soc_t0.subtract(
                    soc_t0.multiply(lc_tr_fl).multiply(lc_tr_fm).multiply(lc_tr_fo)
                )
            ).divide(20)

            # compute final SOC stock for the period
            soc_t1 = soc_t0.subtract(soc_chg)

            # add to land cover and soc to stacks from both dates for the first
            # period
            stack_lc = stack_lc.addBands(lc_t0_orig_coding).addBands(lc_t1_orig_coding)
            stack_soc = stack_soc.addBands(soc_t0).addBands(soc_t1)

        else:
            # compute annual change in soc (updates from previous period based
            # on transition and time <20 years)
            soc_chg = soc_chg.where(
                lc_t0.neq(lc_t1),
                (
                    stack_soc.select(k).subtract(
                        stack_soc.select(k)
                        .multiply(lc_tr_fl)
                        .multiply(lc_tr_fm)
                        .multiply(lc_tr_fo)
                    )
                ).divide(20),
            ).where(tr_time.gt(20), 0)

            # compute final SOC for the period
            socn = stack_soc.select(k).subtract(soc_chg)

            # add land cover and soc to stacks only for the last year in the
            # period
            stack_lc = stack_lc.addBands(lc_t1_orig_coding)
            stack_soc = stack_soc.addBands(socn)

    # compute soc percent change for the analysis period
    logger.debug(
        f"lc band names are: {lc.bandNames().getInfo()}\n"
        f"lc band length is: {len(lc.bandNames().getInfo())}\n"
        f"stack_lc band names are: {stack_lc.bandNames().getInfo()}\n"
        f"stack_lc band length is: {len(stack_lc.bandNames().getInfo())}\n"
        f"stack_soc band names are: {stack_soc.bandNames().getInfo()}\n"
        f"stack_soc band length is: {len(stack_soc.bandNames().getInfo())}\n"
        f"year_final is: {year_final}, and soc_t0_year is: {soc_t0_year}"
    )
    soc_pch = (
        (
            (
                stack_soc.select(year_final - soc_t0_year).subtract(
                    stack_soc.select(year_initial - soc_t0_year)
                )
            ).divide(stack_soc.select(year_initial - soc_t0_year))
        ).multiply(100)
    ).rename(f"Percent_SOC_increase_{year_initial}-{year_final}")

    logger.debug("Setting up output.")
    out = TEImage(
        soc_pch,
        [
            BandInfo(
                "Soil organic carbon (degradation)",
                add_to_map=True,
                metadata={
                    "year_initial": year_initial,
                    "year_final": year_final,
                    "nesting": ipcc_nesting.dumps(),
                },
            )
        ],
    )

    # Setup a list of the years included
    years = [*range(year_initial, year_final + 1)]

    # Output annual SOC layers
    d_soc = []

    if annual_soc:
        logger.debug(f"Adding annual SOC layers for years {years}.")
        for year in years:
            if year == years[0]:
                soc_stack_out = stack_soc.select(year - soc_t0_year)
            else:
                soc_stack_out = soc_stack_out.addBands(
                    stack_soc.select(year - soc_t0_year)
                )

            if (year == year_initial) or (year == year_final):
                add_to_map = True
            else:
                add_to_map = False
            d_soc.append(
                BandInfo(
                    "Soil organic carbon",
                    add_to_map=add_to_map,
                    metadata={"year": year},
                )
            )
        soc_stack_out = soc_stack_out.rename([f"SOC_{year}" for year in years])
        out.addBands(soc_stack_out, d_soc)
    else:
        logger.debug("Adding initial and final SOC layers.")
        soc_initial = stack_soc.select(year_initial - soc_t0_year).rename(
            f"SOC_{year_initial}"
        )
        soc_final = stack_soc.select(year_final - soc_t0_year).rename(
            f"SOC_{year_final}"
        )
        out.addBands(
            soc_initial.addBands(soc_final),
            [
                BandInfo(
                    "Soil organic carbon",
                    add_to_map=True,
                    metadata={"year": year_initial},
                ),
                BandInfo(
                    "Soil organic carbon",
                    add_to_map=True,
                    metadata={"year": year_final},
                ),
            ],
        )

    if annual_lc:
        logger.debug("Adding all annual LC layers.")
        d_lc = []

        for year in years:
            if year == years[0]:
                lc_stack_out = stack_lc.select(year - soc_t0_year)
            else:
                lc_stack_out = lc_stack_out.addBands(
                    stack_lc.select(year - soc_t0_year)
                )
            d_lc.append(
                BandInfo(
                    "Land cover",
                    metadata={"year": year, "nesting": ipcc_nesting.dumps()},
                )
            )
        lc_stack_out = lc_stack_out.rename([f"Land_cover_{year}" for year in years])
        out.addBands(lc_stack_out, d_lc)
    else:
        logger.debug("Adding initial and final LC layers.")
        lc_initial = stack_lc.select(year_initial - soc_t0_year).rename(
            f"Land_cover_{year_initial}"
        )
        lc_final = stack_lc.select(year_final - soc_t0_year).rename(
            f"Land_cover_{year_final}"
        )
        out.addBands(
            lc_initial.addBands(lc_final),
            [
                BandInfo(
                    "Land cover",
                    metadata={"year": year_initial, "nesting": ipcc_nesting.dumps()},
                ),
                BandInfo(
                    "Land cover",
                    metadata={"year": year_final, "nesting": ipcc_nesting.dumps()},
                ),
            ],
        )

    out.image = out.image.unmask(-32768).int16()

    return out


def soc_deg(
    year_initial,
    year_final,
    fl,
    esa_to_custom_nesting,  # defines how ESA nests to custom classes
    ipcc_nesting,  # defines how custom classes nest to IPCC
    logger,
    fake_data=False,  # return data from closest available year if year is outside of range
):
    """Calculate SOC indicator."""
    logger.debug("Entering soc function.")

    # soc
    soc = ee.Image("users/geflanddegradation/toolbox_datasets/soc_sgrid_30cm")
    soc_t0 = soc.updateMask(soc.neq(-32768))

    # Reference all SOC calculations to the year 2000
    soc_t0_year = 2000

    # First band in the LC layer stack is 1992
    lc_band0_year = 1992

    # land cover - note it needs to be reprojected to match soc so that it can
    # be output to cloud storage in the same stack
    lc = _select_lc(
        soc_t0_year, lc_band0_year, year_final, fake_data, logger
    ).reproject(crs=soc.projection())
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    if fl == "per pixel":
        # Setup a raster of climate regimes to use for coding Fl automatically
        climate = ee.Image(
            "users/geflanddegradation/toolbox_datasets/ipcc_climate_zones"
        ).remap(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 2, 1, 2, 1, 2, 1, 2, 1, 5, 4, 4, 3],
        )  # yapf: disable
        clim_fl = climate.remap(
            [0, 1, 2, 3, 4, 5], [0, 0.8, 0.69, 0.58, 0.48, 0.64]
        )  # yapf: disable
    # create empty stacks to store annual land cover maps
    stack_lc = ee.Image().select()

    # create empty stacks to store annual soc maps
    stack_soc = ee.Image().select()

    # loop through all the years in the period of analysis to compute changes in SOC

    class_codes = sorted([c.code for c in esa_to_custom_nesting.parent.key])
    class_positions = [*range(1, len(class_codes) + 1)]
    for k in range(year_final - soc_t0_year):
        logger.info(
            f"Comparing lc band {k} with band {k + 1} for years {soc_t0_year + k} and {soc_t0_year + k + 1}."
        )
        # land cover map reclassified to custom classes (1: forest, 2:
        # grassland, 3: cropland, 4: wetland, 5: artifitial, 6: bare, 7: water)
        lc_t0_orig_coding = lc.select(k).remap(
            esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
        )
        lc_t0 = lc_t0_orig_coding.remap(class_codes, class_positions)

        lc_t1_orig_coding = lc.select(k + 1).remap(
            esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
        )
        lc_t1 = lc_t1_orig_coding.remap(class_codes, class_positions)

        if k == 0:
            # compute transition map (first digit for baseline land cover, and
            # second digit for target year land cover)
            lc_tr = lc_t0.multiply(esa_to_custom_nesting.parent.get_multiplier()).add(
                lc_t1
            )

            # compute raster to register years since transition
            tr_time = ee.Image(2).where(lc_t0.neq(lc_t1), 1)
        else:
            # Update time since last transition. Add 1 if land cover remains
            # constant, and reset to 1 if land cover changed.
            tr_time = tr_time.where(lc_t0.eq(lc_t1), tr_time.add(ee.Image(1))).where(
                lc_t0.neq(lc_t1), ee.Image(1)
            )

            # compute transition map (first digit for baseline land cover, and
            # second digit for target year land cover), but only update where
            # changes actually ocurred.
            lc_tr_temp = lc_t0.multiply(10).add(lc_t1)
            lc_tr = lc_tr.where(lc_t0.neq(lc_t1), lc_tr_temp)

        # Convert lc_tr_fl_0 (defined against IPCC legend)
        soc_change_factor_for_land_use = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_LAND_USE, ipcc_nesting
        )
        lc_tr_fl_0 = lc_tr.remap(*soc_change_factor_for_land_use)

        if fl == "per pixel":
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), clim_fl).where(
                lc_tr_fl_0.eq(-99), ee.Image(1).divide(clim_fl)
            )
        else:
            lc_tr_fl = lc_tr_fl_0.where(lc_tr_fl_0.eq(99), fl).where(
                lc_tr_fl_0.eq(-99), ee.Image(1).divide(fl)
            )

        soc_change_factor_for_management = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_MANAGEMENT, ipcc_nesting
        )
        lc_tr_fm = lc_tr.remap(*soc_change_factor_for_management)

        soc_change_factor_for_organic_matter = trans_factors_for_custom_legend(
            SOC_CHANGE_FACTOR_FOR_ORGANIC_MATTER, ipcc_nesting
        )
        lc_tr_fo = lc_tr.remap(*soc_change_factor_for_organic_matter)

        if k == 0:
            soc_chg = (
                soc_t0.subtract(
                    soc_t0.multiply(lc_tr_fl).multiply(lc_tr_fm).multiply(lc_tr_fo)
                )
            ).divide(20)

            # compute final SOC stock for the period
            soc_t1 = soc_t0.subtract(soc_chg)

            # add to land cover and soc to stacks from both dates for the first
            # period
            stack_lc = stack_lc.addBands(lc_t0_orig_coding).addBands(lc_t1_orig_coding)
            stack_soc = stack_soc.addBands(soc_t0).addBands(soc_t1)

        else:
            # compute annual change in soc (updates from previous period based
            # on transition and time <20 years)
            soc_chg = soc_chg.where(
                lc_t0.neq(lc_t1),
                (
                    stack_soc.select(k).subtract(
                        stack_soc.select(k)
                        .multiply(lc_tr_fl)
                        .multiply(lc_tr_fm)
                        .multiply(lc_tr_fo)
                    )
                ).divide(20),
            ).where(tr_time.gt(20), 0)

            # compute final SOC for the period
            socn = stack_soc.select(k).subtract(soc_chg)

            # add land cover and soc to stacks only for the last year in the
            # period
            stack_lc = stack_lc.addBands(lc_t1_orig_coding)
            stack_soc = stack_soc.addBands(socn)

    # compute soc percent change for the analysis period
    logger.debug(
        f"lc band names are: {lc.bandNames().getInfo()}\n"
        f"lc band length is: {len(lc.bandNames().getInfo())}\n"
        f"stack_lc band names are: {stack_lc.bandNames().getInfo()}\n"
        f"stack_lc band length is: {len(stack_lc.bandNames().getInfo())}\n"
        f"stack_soc band names are: {stack_soc.bandNames().getInfo()}\n"
        f"stack_soc band length is: {len(stack_soc.bandNames().getInfo())}\n"
        f"year_final is: {year_final}, and soc_t0_year is: {soc_t0_year}"
    )
    soc_pch = (
        (
            (
                stack_soc.select(year_final - soc_t0_year).subtract(
                    stack_soc.select(year_initial - soc_t0_year)
                )
            ).divide(stack_soc.select(year_initial - soc_t0_year))
        ).multiply(100)
    ).rename(f"Percent_SOC_increase_{year_initial}-{year_final}")

    return soc_pch.unmask(-32768).int16()
