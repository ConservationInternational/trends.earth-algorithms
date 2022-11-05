import ee
from te_schemas.schemas import BandInfo

from .util import TEImage


def tc(
    fc_threshold, year_initial, year_final, method, biomass_data, EXECUTION_ID, logger
):
    """
    Calculate total carbon (in belowground and aboveground biomass).
    """
    logger.debug("Entering tc function.")

    ##############################################
    # DATASETS
    # Import Hansen global forest dataset
    hansen = ee.Image("UMD/hansen/global_forest_change_2019_v1_7")

    # Aboveground Live Woody Biomass per Hectare (Mg/Ha)
    if biomass_data == "woodshole":
        agb = (
            ee.ImageCollection(
                "users/geflanddegradation/toolbox_datasets/forest_agb_30m_gfw"
            )
            .mosaic()
            .unmask(0)
        )
    elif biomass_data == "geocarbon":
        agb = ee.Image(
            "users/geflanddegradation/toolbox_datasets/forest_agb_1km_geocarbon"
        )
    else:
        agb = None
    # All datasets will be reprojected to Hansen resolution
    agb = agb.reproject(crs=hansen.projection())

    # JRC Global Surface Water Mapping Layers, v1.0 (>50% occurrence)
    water = ee.Image("JRC/GSW1_0/GlobalSurfaceWater").select("occurrence")
    water = water.reproject(crs=hansen.projection())

    # reclass to 1.broadleaf, 2.conifer, 3.mixed, 4.savanna
    f_type = ee.Image(
        "users/geflanddegradation/toolbox_datasets/esa_forest_expanded_2015"
    ).remap(
        [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110],
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
    )
    f_type = f_type.reproject(crs=hansen.projection())

    # IPCC climate zones reclassified as from http://eusoils.jrc.ec.europa.eu/projects/RenewableEnergy/
    # 0-No data, 1-Warm Temperate Moist, 2-k Temperate Dry, 3-Cool Temperate
    # Moist, 4-Cool Temperate Dry, 5-Polar Moist,
    # 6-Polar Dry, 7-Boreal Moist, 8-Boreal Dry, 9-Tropical Montane, 10-Tropical Wet, 11-Tropical Moist, 12-Tropical Dry) to
    # 0: no data, 1:trop/sub moist, 2: trop/sub dry, 3: temperate)
    climate = ee.Image(
        "users/geflanddegradation/toolbox_datasets/ipcc_climate_zones"
    ).remap(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 2],
    )
    climate = climate.reproject(crs=hansen.projection())

    # Root to shoot ratio methods
    if method == "ipcc":
        rs_ratio = (
            ee.Image(-32768)
            # low biomass wet tropical forest
            .where(climate.eq(1).And(agb.lte(125)), 0.42)
            # high biomass wet tropical forest
            .where(climate.eq(1).And(agb.gte(125)), 0.24)
            # dry tropical forest
            .where(climate.eq(2), 0.27)
            # low biomass temperate conifer forest
            .where(climate.eq(3).And(f_type.eq(2).And(agb.lte(50))), 0.46)
            # mid biomass temperate conifer forest
            .where(
                climate.eq(3).And(f_type.eq(2).And(agb.gte(50)).And(agb.lte(150))), 0.32
            )
            # high biomass temperate conifer forest
            .where(climate.eq(3).And(f_type.eq(2).And(agb.lte(150))), 0.23)
            # low biomass temperate broadleaf forest
            .where(climate.eq(3).And(f_type.eq(1).And(agb.lte(75))), 0.43)
            # low biomass temperate broadleaf forest
            .where(
                climate.eq(3).And(f_type.eq(1).And(agb.gte(75)).And(agb.lte(150))), 0.26
            )
            # low biomass temperate broadleaf forest
            .where(climate.eq(3).And(f_type.eq(1).And(agb.lte(150))), 0.24)
            # low biomass temperate mixed forest
            .where(climate.eq(3).And(f_type.eq(1).And(agb.lte(75))), (0.46 + 0.43) / 2)
            # low biomass temperate mixed forest
            .where(
                climate.eq(3).And(f_type.eq(1).And(agb.gte(75)).And(agb.lte(150))),
                (0.32 + 0.26) / 2,
            )
            # low biomass temperate mixed forest
            .where(climate.eq(3).And(f_type.eq(1).And(agb.lte(150))), (0.23 + 0.24) / 2)
            # savanas regardless of climate
            .where(f_type.eq(4), 2.8)
        )
        bgb = agb.multiply(rs_ratio)
    elif method == "mokany":
        # calculate average above and below ground biomass
        # BGB (t ha-1) Citation Mokany et al. 2006 = (0.489)*(AGB)^(0.89)
        # Mokany used a linear regression of root biomass to shoot biomass for forest
        # and woodland and found that BGB(y) is ~ 0.489 of AGB(x).  However,
        # applying a power (0.89) to the shoot data resulted in an improved
        # model for relating root biomass (y) to shoot biomass (x):
        # y = 0:489 x0:890
        bgb = agb.expression("0.489 * BIO**(0.89)", {"BIO": agb})
        rs_ratio = bgb.divide(agb)
    else:
        raise

    # Calculate Total biomass (t/ha) then convert to carbon equilavent (*0.5) to get Total Carbon (t ha-1) = (AGB+BGB)*0.5
    tbcarbon = agb.expression("(bgb + abg) * 0.5", {"bgb": bgb, "abg": agb})

    # convert Total Carbon to Total Carbon dioxide tCO2/ha
    # One ton of carbon equals 44/12 = 11/3 = 3.67 tons of carbon dioxide
    # teco2 = agb.expression('totalcarbon * 3.67 ', {'totalcarbon': tbcarbon})

    ##############################################/
    # define forest cover at the starting date
    fc_str = (
        hansen.select("treecover2000")
        .gte(fc_threshold)
        .multiply(
            hansen.select("lossyear")
            .unmask(0)
            .lte(0)
            .add(hansen.select("lossyear").unmask(0).gt(year_initial - 2000))
        )
    )

    # Create three band layer clipped to study area
    # Band 1: forest layer for initial year (0) and year loss coded as numbers (e.g. 1 = 2001)
    # Band 2: root to shoot ratio
    # Band 3: total carbon stocks (tons of C per ha)
    output = (
        fc_str.multiply(
            hansen.select("lossyear")
            .unmask(0)
            .lte(year_final - 2000)
            .multiply(hansen.select("lossyear").unmask(0))
        )
        .where(fc_str.eq(0), -1)
        .where(water.gte(50), -2)
        .unmask(-32768)
        .addBands((rs_ratio.multiply(100)).multiply(fc_str))
        .unmask(-32768)
        .addBands((tbcarbon.multiply(10)).multiply(fc_str))
        .unmask(-32768)
    )
    output = output.reproject(crs=hansen.projection())

    logger.debug("Setting up output.")
    out = TEImage(
        output.int16(),
        [
            BandInfo(
                "Forest loss",
                add_to_map=True,
                metadata={
                    "year_initial": year_initial,
                    "year_final": year_final,
                    "ramp_min": year_initial - 2000 + 1,
                    "ramp_max": year_final - 2000,
                    "threshold": fc_threshold,
                },
            ),
            BandInfo("Root/shoot ratio", add_to_map=False, metadata={"method": method}),
            BandInfo(
                "Total carbon",
                add_to_map=True,
                metadata={
                    "year_initial": year_initial,
                    "year_final": year_final,
                    "method": method,
                    "threshold": fc_threshold,
                },
            ),
        ],
    )
    return out
