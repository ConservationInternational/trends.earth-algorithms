from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import json

import ee

from landdegradation import stats
from landdegradation import util
from landdegradation import GEEIOError

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
    mk_trend  = stats.mann_kendall(ndvi_1yr_coll.select('ndvi'))

    return (lf_trend, mk_trend)

def p_restrend(year_start, year_end, ndvi_1yr, climate_1yr, logger):
    logger.debug("Entering p_restrend function.")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi_img = ndvi_stack.select('y{}'.format(k))\
                .addBands(climate_1yr.select('y{}'.format(k)))\
                .rename(['ndvi','clim']).set({'year': k})
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
        return ndvi_r.rename(['year','ndvi_res'])

    # Function to compute differences between observed and predicted NDVI and compilation in an image collection
    def stack(year_start, year_end):
        img_coll = ee.List([])
        for k in range(year_start, year_end + 1):
            ndvi = ndvi_1yr_o.filter(ee.Filter.eq('year', k)).select('ndvi').median()
            clim = clim_1yr_o.filter(ee.Filter.eq('year', k)).select('ndvi').median()
            img = ndvi.addBands(clim.addBands(ee.Image(k).float())).rename(['ndvi','clim','year']).set({'year': k})
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
    ndvi_1yr_r  = f_ndvi_clim_r_coll(year_start,year_end)

    ## Fit a linear regression to the NDVI residuals
    lf_trend = ndvi_1yr_r.select(['year', 'ndvi_res']).reduce(ee.Reducer.linearFit())

    ## Compute Kendall statistics
    mk_trend  = stats.mann_kendall(ndvi_1yr_r.select('ndvi_res'))

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
                                .rename(['ue','year']).set({'year': k})
            img_coll = img_coll.add(ndvi_img)
        return ee.ImageCollection(img_coll)

    ## Apply function to compute ue and store as a collection
    ue_1yr_coll = f_img_coll(ndvi_1yr)

    ## Compute linear trend function to predict ndvi based on year (ndvi trend)
    lf_trend = ue_1yr_coll.select(['year', 'ue']).reduce(ee.Reducer.linearFit())

    ## Compute Kendall statistics
    mk_trend  = stats.mann_kendall(ue_1yr_coll.select('ue'))

    return (lf_trend, mk_trend)

def productivity_trajectory(year_start, year_end, method, ndvi_gee_dataset, 
        climate_gee_dataset, logger):
    logger.debug("Entering productivity_trajectory function.")

    climate_1yr = ee.Image(climate_gee_dataset)

    if climate_gee_dataset == None and method != 'ndvi_trend':
        raise GEEIOError("Must specify a climate dataset")

    ndvi_dataset = ee.Image(ndvi_gee_dataset)

    # Run the selected algorithm
    if method == 'ndvi_trend':
        lf_trend, mk_trend = ndvi_trend(year_start, year_end, ndvi_dataset, logger)
    elif method == 'p_restrend':
        lf_trend, mk_trend = p_restrend(year_start, year_end, ndvi_dataset, climate_1yr, logger)
        if climate_1yr == None: climate_1yr = precp_gpcc
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
    kendall95 = stats.get_kendall_coef(period - 4, 90)
    kendall99 = stats.get_kendall_coef(period - 4, 90)

    # create final degradation output layer: 9997 is no data, 0 is not 
    # degraded, -3 is degraded (pvalue < 0.1), -2 is degraded (pvalue < 0.05), 
    # -3 is degraded (pvalue < 0.01), 3 is improving (pvalue < 0.1), 2 is 
    # improving (pvalue < 0.05), 3 is improving (pvalue < 0.01), 9998 is water, 
    # and 9999 is urban
    attri = ee.Image(9997) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall90)), 1) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall95)), 2) \
        .where(lf_trend.select('scale').gt(0).And(mk_trend.abs().gte(kendall99)), 3) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall90)), -1) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall95)), -2) \
        .where(lf_trend.select('scale').lt(0).And(mk_trend.abs().gte(kendall99)), -3)

    output = lf_trend.select('scale').unmask(9997) \
        .addBands(attri).rename(['slope','attri'])

    return output
