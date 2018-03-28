from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation import stats, GEEIOError
from landdegradation.util import TEImage
from landdegradation.schemas.schemas import BandInfo


def ndvi_trend(year_start, year_end, ndvi_1yr, logger):
    """Calculate temporal NDVI analysis.
    Calculates the trend of temporal NDVI using NDVI data from the
    MODIS Collection 6 MOD13Q1 dataset. Areas where changes are not significant
    are masked out using a Mann-Kendall test.
    Args:
        year_start: The starting year (to define the period the trend is
            calculated over).
        year_end: The ending year (to define the period the trend is
            calculated over).
    Returns:
        Output of google earth engine task.
    """
    logger.debug("Entering ndvi_trend function.")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi_img = ndvi_stack.select('y' + str(k)).addBands(ee.Image(k).float()).rename(['ndvi', 'year'])
            img_coll = img_coll.add(ndvi_img)
        return ee.ImageCollection(img_coll)

    ## Apply function to compute NDVI annual integrals from 15d observed NDVI data
    ndvi_1yr_coll = f_img_coll(ndvi_1yr)

    ## Compute linear trend function to predict ndvi based on year (ndvi trend)
    lf_trend = ndvi_1yr_coll.select(['year', 'ndvi']).reduce(ee.Reducer.linearFit())

    ## Compute Kendall statistics
    mk_trend = stats.mann_kendall(ndvi_1yr_coll.select('ndvi'))

    return (lf_trend, mk_trend)


def p_restrend(year_start, year_end, ndvi_1yr, climate_1yr, logger):
    logger.debug("Entering p_restrend function.")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi_img = ndvi_stack.select('y{}'.format(k))\
                .addBands(climate_1yr.select('y{}'.format(k)))\
                .rename(['ndvi', 'clim']).set({'year': k})
            img_coll = img_coll.add(ndvi_img)
        return ee.ImageCollection(img_coll)

    ## Function to predict NDVI from climate
    first = ee.List([])

    def f_ndvi_clim_p(image, list):
        ndvi = lf_clim_ndvi.select('offset').add((lf_clim_ndvi.select('scale').multiply(image))).set({'year': image.get('year')})
        return ee.List(list).add(ndvi)

    ## Function to compute residuals (ndvi obs - ndvi pred)
    def f_ndvi_clim_r_img(year):
        ndvi_o = ndvi_1yr_coll.filter(ee.Filter.eq('year', year)).select('ndvi').median()
        ndvi_p = ndvi_1yr_p.filter(ee.Filter.eq('year', year)).median()
        ndvi_r = ee.Image(year).float().addBands(ndvi_o.subtract(ndvi_p))
        return ndvi_r.rename(['year', 'ndvi_res'])

    # Function to compute differences between observed and predicted NDVI and compilation in an image collection
    def stack(year_start, year_end):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi = ndvi_1yr_o.filter(ee.Filter.eq('year', k)).select('ndvi').median()
            clim = clim_1yr_o.filter(ee.Filter.eq('year', k)).select('ndvi').median()
            img = ndvi.addBands(clim.addBands(ee.Image(k).float())).rename(['ndvi', 'clim', 'year']).set({'year': k})
            img_coll = img_coll.add(img)
        return ee.ImageCollection(img_coll)

    ## Function create image collection of residuals
    def f_ndvi_clim_r_coll(year_start, year_end):
        res_list = ee.List([])
        #for(i = year_start i <= year_end i += 1):
        for i in range(year_start, year_end + 1):
            res_image = f_ndvi_clim_r_img(i)
            res_list = res_list.add(res_image)
        return ee.ImageCollection(res_list)

    ## Apply function to create image collection of ndvi and climate
    ndvi_1yr_coll = f_img_coll(ndvi_1yr)

    ## Compute linear trend function to predict ndvi based on climate (independent are followed by dependent var
    lf_clim_ndvi = ndvi_1yr_coll.select(['clim', 'ndvi']).reduce(ee.Reducer.linearFit())

    ## Apply function to  predict NDVI based on climate
    ndvi_1yr_p = ee.ImageCollection(ee.List(ndvi_1yr_coll.select('clim').iterate(f_ndvi_clim_p, first)))

    ## Apply function to compute NDVI annual residuals
    ndvi_1yr_r = f_ndvi_clim_r_coll(year_start, year_end)

    ## Fit a linear regression to the NDVI residuals
    lf_trend = ndvi_1yr_r.select(['year', 'ndvi_res']).reduce(ee.Reducer.linearFit())

    ## Compute Kendall statistics
    mk_trend = stats.mann_kendall(ndvi_1yr_r.select('ndvi_res'))

    return (lf_trend, mk_trend)


def s_restrend(year_start, year_end, ndvi_1yr, climate_1yr, logger):
    #TODO: Copy this code over
    logger.debug("Entering s_restrend function.")


def ue_trend(year_start, year_end, ndvi_1yr, climate_1yr, logger):
    # Convert the climate layer to meters (for precip) so that RUE layer can be
    # scaled correctly
    # TODO: Need to handle scaling for ET for WUE
    climate_1yr = climate_1yr.divide(1000)
    logger.debug("Entering ue_trend function.")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi_img = ndvi_stack.select('y{}'.format(k)).divide(climate_1yr.select('y{}'.format(k)))\
                .addBands(ee.Image(k).float())\
                .rename(['ue', 'year']).set({'year': k})
            img_coll = img_coll.add(ndvi_img)
        return ee.ImageCollection(img_coll)

    ## Apply function to compute ue and store as a collection
    ue_1yr_coll = f_img_coll(ndvi_1yr)

    ## Compute linear trend function to predict ndvi based on year (ndvi trend)
    lf_trend = ue_1yr_coll.select(['year', 'ue']).reduce(ee.Reducer.linearFit())

    ## Compute Kendall statistics
    mk_trend = stats.mann_kendall(ue_1yr_coll.select('ue'))

    return (lf_trend, mk_trend)


def productivity_trajectory(year_start, year_end, method, ndvi_gee_dataset,
                            climate_gee_dataset, logger):
    logger.debug("Entering productivity_trajectory function.")

    climate_1yr = ee.Image(climate_gee_dataset)
    climate_1yr = climate_1yr.where(climate_1yr.eq(9999), -32768)
    climate_1yr = climate_1yr.updateMask(climate_1yr.neq(-32768))

    if climate_gee_dataset == None and method != 'ndvi_trend':
        raise GEEIOError("Must specify a climate dataset")

    ndvi_dataset = ee.Image(ndvi_gee_dataset)
    ndvi_dataset = ndvi_dataset.where(ndvi_dataset.eq(9999), -32768)
    ndvi_dataset = ndvi_dataset.updateMask(ndvi_dataset.neq(-32768))

    # Run the selected algorithm
    if method == 'ndvi_trend':
        lf_trend, mk_trend = ndvi_trend(year_start, year_end, ndvi_dataset, logger)
    elif method == 'p_restrend':
        lf_trend, mk_trend = p_restrend(year_start, year_end, ndvi_dataset, climate_1yr, logger)
        if climate_1yr == None:
            climate_1yr = precp_gpcc
    elif method == 's_restrend':
        #TODO: need to code this
        raise GEEIOError("s_restrend method not yet supported")
    elif method == 'ue':
        lf_trend, mk_trend = ue_trend(year_start, year_end, ndvi_dataset, climate_1yr, logger)
    else:
        raise GEEIOError("Unrecognized method '{}'".format(method))

    # Define Kendall parameter values for a significance of 0.05
    period = year_end - year_start + 1
    kendall90 = stats.get_kendall_coef(period - 4, 90)
    kendall95 = stats.get_kendall_coef(period - 4, 95)
    kendall99 = stats.get_kendall_coef(period - 4, 99)

    # create final degradation output layer: 9999 is no data, 0 is not
    # degraded, -3 is degraded (pvalue < 0.1), -2 is degraded (pvalue < 0.05),
    # -3 is degraded (pvalue < 0.01), 3 is improving (pvalue < 0.1), 2 is
    # improving (pvalue < 0.05), 3 is improving (pvalue < 0.01)
    signif = ee.Image(-32768) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall90)), 1) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall95)), 2) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall99)), 3) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall90)), -1) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall95)), -2) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall99)), -3) \
        .where(mk_trend.abs().lte(kendall90), 0)

    return TEImage(lf_trend.select('scale').addBands(signif).rename(['slope', 'signif']).unmask(-32768).int16(),
                   [BandInfo("Productivity trajectory (trend)", add_to_map=True, metadata={'year_start': year_start, 'year_end': year_end}),
                    BandInfo("Productivity trajectory (significance)", add_to_map=True, metadata={'year_start': year_start, 'year_end': year_end})])


def productivity_performance(year_start, year_end, ndvi_gee_dataset, geojson,
                             EXECUTION_ID, logger):
    logger.debug("Entering productivity_performance function.")

    ndvi_1yr = ee.Image(ndvi_gee_dataset)
    ndvi_1yr = ndvi_1yr.where(ndvi_1yr.eq(9999), -32768)
    ndvi_1yr = ndvi_1yr.updateMask(ndvi_1yr.neq(-32768))

    # land cover data from esa cci
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2015")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # global agroecological zones from IIASA
    soil_tax_usda = ee.Image("users/geflanddegradation/toolbox_datasets/soil_tax_usda_sgrid")

    # Make sure the bounding box of the poly is used, and not the geodesic 
    # version, for the clipping
    poly = ee.Geometry(geojson, opt_geodesic=False)

    # compute mean ndvi for the period
    ndvi_avg = ndvi_1yr.select(ee.List(['y{}'.format(i) for i in range(year_start, year_end + 1)])) \
        .reduce(ee.Reducer.mean()).rename(['ndvi']).clip(poly)

    # Handle case of year_start that isn't included in the CCI data
    if year_start > 2015:
        lc_year_start = 2015
    elif year_start < 1992:
        lc_year_start = 1992
    else:
        lc_year_start = year_start
    # reclassify lc to ipcc classes
    lc_t0 = lc.select('y{}'.format(lc_year_start)) \
        .remap([10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 160, 170, 110, 130, 180, 190, 120, 121, 122, 140, 150, 151, 152, 153, 200, 201, 202, 210], 
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36])

    # create a binary mask.
    mask = ndvi_avg.neq(0)

    # define modis projection attributes
    modis_proj = ee.Image("users/geflanddegradation/toolbox_datasets/ndvi_modis_2001_2016").projection()

    # reproject land cover, soil_tax_usda and avhrr to modis resolution
    lc_proj = lc_t0.reproject(crs=modis_proj)
    soil_tax_usda_proj = soil_tax_usda.reproject(crs=modis_proj)
    ndvi_avg_proj = ndvi_avg.reproject(crs=modis_proj)

    # define unit of analysis as the intersect of soil_tax_usda and land cover
    units = soil_tax_usda_proj.multiply(100).add(lc_proj)

    # create a 2 band raster to compute 90th percentile per unit (analysis restricted by mask and study area)
    ndvi_id = ndvi_avg_proj.addBands(units).updateMask(mask)

    # compute 90th percentile by unit
    perc90 = ndvi_id.reduceRegion(reducer=ee.Reducer.percentile([90]).
                                  group(groupField=1, groupName='code'),
                                  geometry=poly,
                                  scale=ee.Number(modis_proj.nominalScale()).getInfo(),
                                  maxPixels=1e15)

    # Extract the cluster IDs and the 90th percentile
    groups = ee.List(perc90.get("groups"))
    ids = groups.map(lambda d: ee.Dictionary(d).get('code'))
    perc = groups.map(lambda d: ee.Dictionary(d).get('p90'))

    # remap the units raster using their 90th percentile value
    raster_perc = units.remap(ids, perc)

    # compute the ration of observed ndvi to 90th for that class
    obs_ratio = ndvi_avg_proj.divide(raster_perc)

    # aggregate obs_ratio to original NDVI data resolution (for modis this step does not change anything)
    obs_ratio_2 = obs_ratio.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=2000) \
        .reproject(crs=ndvi_1yr.projection())

    # create final degradation output layer (9999 is background), 0 is not
    # degreaded, -1 is degraded
    lp_perf_deg = ee.Image(-32768).where(obs_ratio_2.gte(0.5), 0) \
        .where(obs_ratio_2.lte(0.5), -1)

    return TEImage(lp_perf_deg.addBands(obs_ratio_2.multiply(10000)).addBands(units).unmask(-32768).int16(),
                   [BandInfo("Productivity performance (degradation)", True, {'year_start': year_start, 'year_end': year_end}),
                    BandInfo("Productivity performance (ratio)", metadata={'year_start': year_start, 'year_end': year_end}),
                    BandInfo("Productivity performance (units)", metadata={'year_start': year_start})])


def productivity_state(year_bl_start, year_bl_end,
                       year_tg_start, year_tg_end,
                       ndvi_gee_dataset, EXECUTION_ID, logger):
    logger.debug("Entering productivity_state function.")

    ndvi_1yr = ee.Image(ndvi_gee_dataset)

    # compute min and max of annual ndvi for the baseline period
    bl_ndvi_range = ndvi_1yr.select(ee.List(['y{}'.format(i) for i in range(year_bl_start, year_bl_end + 1)])) \
        .reduce(ee.Reducer.percentile([0, 100]))

    # add two bands to the time series: one 5% lower than min and one 5% higher than max
    bl_ndvi_ext = ndvi_1yr.select(ee.List(['y{}'.format(i) for i in range(year_tg_start, year_tg_end + 1)])) \
        .addBands(bl_ndvi_range.select('p0').subtract((bl_ndvi_range.select('p100').subtract(bl_ndvi_range.select('p0'))).multiply(0.05)))\
        .addBands(bl_ndvi_range.select('p100').add((bl_ndvi_range.select('p100').subtract(bl_ndvi_range.select('p0'))).multiply(0.05)))

    # compute percentiles of annual ndvi for the extended baseline period
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bl_ndvi_perc = bl_ndvi_ext.reduce(ee.Reducer.percentile(percentiles))

    # compute mean ndvi for the baseline and target period period
    bl_ndvi_mean = ndvi_1yr.select(ee.List(['y{}'.format(i) for i in range(year_bl_start, year_bl_end + 1)])) \
        .reduce(ee.Reducer.mean()).rename(['ndvi'])
    tg_ndvi_mean = ndvi_1yr.select(ee.List(['y{}'.format(i) for i in range(year_tg_start, year_tg_end + 1)])) \
        .reduce(ee.Reducer.mean()).rename(['ndvi'])

    # reclassify mean ndvi for baseline period based on the percentiles
    bl_classes = ee.Image(-32768) \
        .where(bl_ndvi_mean.lte(bl_ndvi_perc.select('p10')), 1) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p10')), 2) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p20')), 3) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p30')), 4) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p40')), 5) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p50')), 6) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p60')), 7) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p70')), 8) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p80')), 9) \
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select('p90')), 10)

    # reclassify mean ndvi for target period based on the percentiles
    tg_classes = ee.Image(-32768) \
        .where(tg_ndvi_mean.lte(bl_ndvi_perc.select('p10')), 1) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p10')), 2) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p20')), 3) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p30')), 4) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p40')), 5) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p50')), 6) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p60')), 7) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p70')), 8) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p80')), 9) \
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select('p90')), 10)

    # difference between start and end clusters >= 2 means improvement (<= -2 
    # is degradation)
    classes_chg = tg_classes.subtract(bl_classes)

    band_infos = [BandInfo("Productivity state (degradation)", add_to_map=True,
                        metadata={'year_bl_start': year_bl_start, 'year_bl_end': year_bl_end, 'year_tg_start': year_tg_start, 'year_tg_end': year_tg_end}),
                  BandInfo("Productivity state classes", metadata={'year_start': year_bl_start, 'year_end': year_bl_end}),
                  BandInfo("Productivity state classes", metadata={'year_start': year_tg_start, 'year_end': year_tg_end}),
                  BandInfo("Productivity state NDVI mean", metadata={'year_start': year_bl_start, 'year_end': year_bl_end}),
                  BandInfo("Productivity state NDVI mean", metadata={'year_start': year_tg_start, 'year_end': year_tg_end})]
    pct_band_infos = [BandInfo("Productivity state percentile {}".format(pct), metadata={'year_start': year_tg_start, 'year_end': year_tg_end}) for pct in percentiles]
    band_infos.extend(pct_band_infos)
    bl_ndvi_band_infos = [BandInfo("Productivity state bl_ndvi_ext") for n in range(14)]
    band_infos.extend(bl_ndvi_band_infos)
    return TEImage(classes_chg.addBands(bl_classes).addBands(tg_classes).addBands(bl_ndvi_mean).addBands(tg_ndvi_mean).addBands(bl_ndvi_perc).addBands(bl_ndvi_ext).int16(), band_infos)
