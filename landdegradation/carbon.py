from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation.util import TEImage
from landdegradation.schemas.schemas import BandInfo


def tc(fc_threshold, year_start, year_end, method, biomass_data, EXECUTION_ID, 
       logger):
    """
    Calculate total carbon (in belowground and aboveground biomass).
    """
    logger.debug("Entering tc function.")

    ##############################################
    # DATASETS
    # Import Hansen global forest dataset
    hansen = ee.Image('UMD/hansen/global_forest_change_2016_v1_4')

    #Import biomass dataset: WHRC is Megagrams of Aboveground Live Woody Biomass per Hectare (Mg/Ha)
    agb = ee.Image("users/geflanddegradation/toolbox_datasets/biomass_30m")

    # reclass to 1.broadleaf, 2.conifer, 3.mixed, 4.savanna
    f_type = ee.Image("users/geflanddegradation/toolbox_datasets/esa_forest_expanded_2015") \
        .remap([50,60,61,62,70,71,72,80,81,82,90,100,110],
               [ 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3,  3,  3])

    # IPCC climate zones reclassified as from http://eusoils.jrc.ec.europa.eu/projects/RenewableEnergy/
    # 0-No data, 1-Warm Temperate Moist, 2-Warm Temperate Dry, 3-Cool Temperate Moist, 4-Cool Temperate Dry, 5-Polar Moist,
    # 6-Polar Dry, 7-Boreal Moist, 8-Boreal Dry, 9-Tropical Montane, 10-Tropical Wet, 11-Tropical Moist, 12-Tropical Dry) to
    # 0: no data, 1:trop/sub moist, 2: trop/sub dry, 3: temperate)
    climate = ee.Image("users/geflanddegradation/toolbox_datasets/ipcc_climate_zones") \
        .remap([0,1,2,3,4,5,6,7,8,9,10,11,12],
               [0,1,2,3,3,3,3,3,3,1, 1, 1, 2])

    # Root to shoot ratio methods
    if method == 'ipcc':
        rs_ratio = (ee.Image(-32768)
            # low biomass wet tropical forest
            .where(climate.eq(1).and(agb.lte(125)), 0.42)
            # high biomass wet tropical forest
            .where(climate.eq(1).and(agb.gte(125)), 0.24)
            # dry tropical forest
            .where(climate.eq(2), 0.27)
            # low biomass temperate conifer forest
            .where(climate.eq(3).and(f_type.eq(2).and(agb.lte(50))), 0.46)
            # mid biomass temperate conifer forest
            .where(climate.eq(3).and(f_type.eq(2).and(agb.gte(50)).and(agb.lte(150))), 0.32)
            # high biomass temperate conifer forest
            .where(climate.eq(3).and(f_type.eq(2).and(agb.lte(150))), 0.23)
            # low biomass temperate broadleaf forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.lte(75))), 0.43)
            # low biomass temperate broadleaf forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.gte(75)).and(agb.lte(150))), 0.26)
            # low biomass temperate broadleaf forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.lte(150))), 0.24)
            # low biomass temperate mixed forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.lte(75))), (0.46+0.43)/2)
            # low biomass temperate mixed forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.gte(75)).and(agb.lte(150))), (0.32+0.26)/2)
            # low biomass temperate mixed forest
            .where(climate.eq(3).and(f_type.eq(1).and(agb.lte(150))), (0.23+0.24)/2)
            # savanas regardless of climate
            .where(f_type.eq(4), 2.8))
        bgb = agb.multiply(rs_ratio)
    elif (method == 'mokany'):
        # calculate average above and below ground biomass
        # BGB (t ha-1) Citation Mokany et al. 2006 = (0.489)*(AGB)^(0.89)
        # Mokany used a linear regression of root biomass to shoot biomass for forest 
        # and woodland and found that BGB(y) is ~ 0.489 of AGB(x).  However, 
        # applying a power (0.89) to the shoot data resulted in an improved 
        # model for relating root biomass (y) to shoot biomass (x):
        # y = 0:489 x0:890
        bgb = agb.expression('0.489 * BIO**(0.89)', {'BIO': agb})
        rs_ratio = bgb.divide(agb)
    else:
        raise

    # Calculate Total biomass (t/ha) then convert to carbon equilavent (*0.5) to get Total Carbon (t ha-1) = (AGB+BGB)*0.5
    tbcarbon = agb.expression('(bgb + abg ) * 0.5 ', {'bgb': bgb,'abg': agb})

    # convert Total Carbon to Total Carbon dioxide tCO2/ha 
    # One ton of carbon equals 44/12 = 11/3 = 3.67 tons of carbon dioxide
    # teco2 = agb.expression('totalcarbon * 3.67 ', {'totalcarbon': tbcarbon})

    ##############################################/
    # define forest cover at the starting date
    fc_str = ee.Image(1).updateMask(hansen.select('treecover2000').gte(fc_threshold)) \
        .updateMask(hansen.select('lossyear').where(hansen.select('lossyear').eq(0), 9999).gte(year_start-2000+1))

    # Create three band layer clipped to study area
    # Band 1: forest layer for initial year (0) and year loss coded as numbers (e.g. 1 = 2001)
    # Band 2: root to shoot ratio 
    # Band 3: total carbon stocks (tons of C per ha)
    output = fc_str.multiply(hansen.select('lossyear')).unmask(-32768) \
        .addBands(rs_ratio.multiply(100)).updateMask(fc_str.eq(1)).unmask(-32768) \
        .addBands(tbcarbon.multiply(10)).updateMask(fc_str.eq(1)).unmask(-32768)

    logger.debug("Setting up output.")
    out = TEImage(output.int16(),
                  [BandInfo("Forest loss", add_to_map=True, metadata={'year_start': year_start,
                                                                      'year_end': year_end,
                                                                      'threshold': fc_threshold}),
                   BandInfo("Root/shoot ratio", add_to_map=False, metadata={'method': method}),
                   BandInfo("Total carbon", add_to_map=True, metadata={'year_start': year_start,
                                                                       'year_end': year_end,
                                                                       'method': method,
                                                                       'threshold': fc_threshold})])
    return out
