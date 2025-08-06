import ee
from ee.ee_exception import EEException
from te_schemas.schemas import BandInfo

from .util import TEImage

LAND_COVER_INITIAL_YEAR = 1992
LAND_COVER_FINAL_YEAR = 2022


def _select_lc(lc, year, logger, fake_data=False):
    try:
        image = lc.select("y{}".format(year))
        image.getInfo()
    except EEException:
        if year < LAND_COVER_INITIAL_YEAR and fake_data:
            image = lc.select(f"y{LAND_COVER_INITIAL_YEAR}")
            logger.warning(
                f"Could not select year {year} from land cover asset. Returning data from {LAND_COVER_INITIAL_YEAR}."
            )
        elif year > LAND_COVER_FINAL_YEAR and fake_data:
            image = lc.select(f"y{LAND_COVER_FINAL_YEAR}")
            logger.warning(
                f"Could not select year {year} from land cover asset. Returning data from {LAND_COVER_FINAL_YEAR}."
            )
    return image


def land_cover(
    year_initial,
    year_final,
    trans_matrix,
    esa_to_custom_nesting,  # defines how ESA nests to custom classes
    ipcc_nesting,  # defines how custom classes nest to IPCC
    additional_years,  # allows including years of lc outside of period
    logger,
    annual_lc=False,
    fake_data=False,  # return data from closest available year if year is outside of range
):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    # Land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    lc_initial = _select_lc(lc, year_initial, logger)
    lc_final = _select_lc(lc, year_final, logger)

    # Transition codes are based on the class code indices (i.e. their order when
    # sorted by class code) - not the class codes themselves. So need to reclass
    # the land cover used for the transition calculations from the raw class codes
    # to the positional indices of those class codes. And before doing that, need to
    # reclassified initial and final layers to the IPCC (or custom) classes.
    class_codes = sorted([c.code for c in esa_to_custom_nesting.parent.key])
    class_positions = [*range(1, len(class_codes) + 1)]
    lc_bl = lc_initial.remap(
        esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
    ).remap(class_codes, class_positions)
    lc_tg = lc_final.remap(
        esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
    ).remap(class_codes, class_positions)

    # compute transition map (first digit for baseline land cover, and second
    # digit for target year land cover)
    lc_tr = lc_bl.multiply(esa_to_custom_nesting.parent.get_multiplier()).add(lc_tg)

    # definition of land cover transitions as degradation (-1), improvement
    # (1), or no relevant change (0)
    lc_dg = lc_tr.remap(trans_matrix.get_list()[0], trans_matrix.get_list()[1]).rename(
        "Land_cover_degradation"
    )

    # Remap persistence classes so they are sequential. This
    # makes it easier to assign a clear color ramp in QGIS.
    lc_tr = lc_tr.remap(
        trans_matrix.get_persistence_list()[0], trans_matrix.get_persistence_list()[1]
    ).rename(f"Land_cover_transitions_{year_initial}-{year_final}")

    logger.debug("Setting up output.")
    lc_baseline_esa = lc_initial.rename(f"Land_cover_{year_initial}")
    lc_target_esa = lc_final.rename(f"Land_cover_{year_final}")
    out = TEImage(
        lc_dg.addBands(lc_baseline_esa).addBands(lc_target_esa).addBands(lc_tr),
        [
            BandInfo(
                "Land cover (degradation)",
                add_to_map=True,
                metadata={
                    "year_initial": year_initial,
                    "year_final": year_final,
                    "trans_matrix": trans_matrix.dumps(),
                    "nesting": ipcc_nesting.dumps(),
                },
            ),
            BandInfo(
                "Land cover (ESA classes)",
                metadata={
                    "year": year_initial,
                    "nesting": esa_to_custom_nesting.dumps(),
                },
            ),
            BandInfo(
                "Land cover (ESA classes)",
                metadata={"year": year_final, "nesting": esa_to_custom_nesting.dumps()},
            ),
            BandInfo(
                "Land cover transitions",
                add_to_map=True,
                metadata={
                    "year_initial": year_initial,
                    "year_final": year_final,
                    "nesting": ipcc_nesting.dumps(),
                },
            ),
        ],
    )

    filtered_years = []
    for year in additional_years:
        if year == year_initial or year == year_final:
            logger.warning(
                f"Year {year} is already included as initial or final year. Skipping."
            )
        elif year < LAND_COVER_INITIAL_YEAR or year > LAND_COVER_FINAL_YEAR:
            if fake_data:
                filtered_years.append(year)
                logger.warning(
                    f"Year {year} is outside of available land cover data range "
                    f"({LAND_COVER_INITIAL_YEAR}-{LAND_COVER_FINAL_YEAR}) "
                    "Including it anyway because fake_data=True."
                )
            else:
                logger.warning(
                    f"Year {year} is outside of available land cover data range. Skipping."
                )
        else:
            filtered_years.append(year)
    additional_years = filtered_years

    if annual_lc:
        years = [*range(year_initial, year_final + 1)] + additional_years
    else:
        years = [year_initial, year_final] + additional_years
    logger.debug(f"Adding lc layers for {years}")
    years = list(set(years))
    lc_remapped = _select_lc(lc, years[0], logger).remap(
        esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
    )
    d_lc = [
        BandInfo(
            "Land cover",
            add_to_map=True,  # because this is initial year
            metadata={"year": years[0], "nesting": ipcc_nesting.dumps()},
        )
    ]
    for year in years[1:]:
        lc_remapped = lc_remapped.addBands(
            _select_lc(lc, year, logger).remap(
                esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
            )
        )
        if year == year_final:
            add_to_map = True
        else:
            add_to_map = False
        d_lc.append(
            BandInfo(
                "Land cover",
                add_to_map=add_to_map,
                metadata={"year": year, "nesting": ipcc_nesting.dumps()},
            )
        )
    lc_remapped = lc_remapped.rename([f"Land_cover_{year}" for year in years])
    out.addBands(lc_remapped, d_lc)

    out.image = out.image.unmask(-32768).int16()

    logger.debug("Leaving land_cover function.")

    return out


def land_cover_deg(
    year_initial,
    year_final,
    trans_matrix,
    esa_to_custom_nesting,  # defines how ESA nests to custom classes
    ipcc_nesting,  # defines how custom classes nest to IPCC
    logger,
    fake_data=False,  # return data from closest available year if year is outside of range
):
    """
    Calculate land cover indicator.
    """
    logger.debug("Entering land_cover function.")

    # Land cover
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    lc_initial = _select_lc(lc, year_initial, logger)
    lc_final = _select_lc(lc, year_final, logger)

    # Transition codes are based on the class code indices (i.e. their order when
    # sorted by class code) - not the class codes themselves. So need to reclass
    # the land cover used for the transition calculations from the raw class codes
    # to the positional indices of those class codes. And before doing that, need to
    # reclassified initial and final layers to the IPCC (or custom) classes.
    class_codes = sorted([c.code for c in esa_to_custom_nesting.parent.key])
    class_positions = [*range(1, len(class_codes) + 1)]
    lc_bl = lc_initial.remap(
        esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
    ).remap(class_codes, class_positions)
    lc_tg = lc_final.remap(
        esa_to_custom_nesting.get_list()[0], esa_to_custom_nesting.get_list()[1]
    ).remap(class_codes, class_positions)

    # compute transition map (first digit for baseline land cover, and second
    # digit for target year land cover)
    lc_tr = lc_bl.multiply(esa_to_custom_nesting.parent.get_multiplier()).add(lc_tg)

    # definition of land cover transitions as degradation (-1), improvement
    # (1), or no relevant change (0)
    lc_dg = lc_tr.remap(trans_matrix.get_list()[0], trans_matrix.get_list()[1]).rename(
        "Land_cover_degradation"
    )

    logger.debug("Leaving land_cover_deg function.")
    return lc_dg.unmask(-32768).int16()
