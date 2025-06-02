import ee
from te_schemas.schemas import BandInfo

from . import GEEIOError, stats
from .util import TEImage


def linear_trend(ndvi_series, logger):
    logger.debug("Entering linear_trend function")
    lf_trend = ndvi_series.select(["year", "ndvi"]).reduce(ee.Reducer.linearFit())
    mk_trend = stats.mann_kendall(ndvi_series.select("ndvi"))
    return (lf_trend, mk_trend)


def ndvi_trend(year_initial, year_final, ndvi_1yr, logger):
    logger.debug("Entering p_restrend function")
    return linear_trend(ndvi(year_initial, year_final, ndvi_1yr, logger), logger)


def ndvi(year_initial, year_final, ndvi_1yr, logger):
    """Calculate temporal NDVI analysis.
    Calculates the trend of temporal NDVI using NDVI data from the
    MODIS Collection 6 MOD13Q1 dataset. Areas where changes are not significant
    are masked out using a Mann-Kendall test.
    Args:
        year_initial: The starting year (to define the period the trend is
            calculated over).
        year_final: The ending year (to define the period the trend is
            calculated over).
    Returns:
        Output of google earth engine task.
    """
    logger.debug("Entering ndvi_trend function.")

    img_coll = ee.List([])

    for k in range(year_initial, year_final + 1):
        ndvi_img = (
            ndvi_1yr.select("y" + str(k))
            .addBands(ee.Image(k).float())
            .rename(["ndvi", "year"])
        )
        img_coll = img_coll.add(ndvi_img)

    return ee.ImageCollection(img_coll)


def p_restrend(year_initial, year_final, ndvi_1yr, climate_1yr, logger):
    logger.debug("Entering p_restrend function")
    return linear_trend(
        p_residuals(year_initial, year_final, ndvi_1yr, climate_1yr, logger), logger
    )


def p_residuals(year_initial, year_final, ndvi_1yr, climate_1yr, logger):
    logger.debug("Entering p_residuals function")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])

        for k in range(year_initial, year_final + 1):
            ndvi_img = (
                ndvi_stack.select(f"y{k}")
                .addBands(climate_1yr.select(f"y{k}"))
                .rename(["ndvi", "clim"])
                .set({"year": k})
            )
            img_coll = img_coll.add(ndvi_img)

        return ee.ImageCollection(img_coll)

    # Function to predict NDVI from climate
    first = ee.List([])

    def f_ndvi_clim_p(image, list):
        ndvi = (
            lf_clim_ndvi.select("offset")
            .add(lf_clim_ndvi.select("scale").multiply(image))
            .set({"year": image.get("year")})
        )

        return ee.List(list).add(ndvi)

    # Function to compute residuals (ndvi obs - ndvi pred)
    def f_ndvi_clim_r_img(year):
        ndvi_o = (
            ndvi_1yr_coll.filter(ee.Filter.eq("year", year)).select("ndvi").median()
        )
        ndvi_p = ndvi_1yr_p.filter(ee.Filter.eq("year", year)).median()
        ndvi_r = ee.Image(year).float().addBands(ndvi_o.subtract(ndvi_p))

        return ndvi_r.rename(["year", "ndvi"])

    # Function create image collection of residuals
    def f_ndvi_clim_r_coll(year_initial, year_final):
        res_list = ee.List([])

        for i in range(year_initial, year_final + 1):
            res_image = f_ndvi_clim_r_img(i)
            res_list = res_list.add(res_image)

        return ee.ImageCollection(res_list)

    # Apply function to create image collection of ndvi and climate
    ndvi_1yr_coll = f_img_coll(ndvi_1yr)

    # Compute linear trend function to predict ndvi based on climate (independent are followed by dependent var
    lf_clim_ndvi = ndvi_1yr_coll.select(["clim", "ndvi"]).reduce(ee.Reducer.linearFit())

    # Apply function to  predict NDVI based on climate
    ndvi_1yr_p = ee.ImageCollection(
        ee.List(ndvi_1yr_coll.select("clim").iterate(f_ndvi_clim_p, first))
    )

    # Apply function to compute NDVI annual residuals
    ndvi_1yr_r = f_ndvi_clim_r_coll(year_initial, year_final)

    return ndvi_1yr_r


def ue_trend(year_initial, year_final, ndvi_1yr, climate_1yr, logger):
    return linear_trend(
        ue(year_initial, year_final, ndvi_1yr, climate_1yr, logger), logger
    )


def ue(year_initial, year_final, ndvi_1yr, climate_1yr, logger):
    # Convert the climate layer to meters (for precip) so that RUE layer can be
    # scaled correctly
    # TODO: Need to handle scaling for ET for WUE
    climate_1yr = climate_1yr.divide(1000)
    logger.debug("Entering ue_trend function.")

    def f_img_coll(ndvi_stack):
        img_coll = ee.List([])

        for k in range(year_initial, year_final + 1):
            ndvi_img = (
                ndvi_stack.select(f"y{k}")
                .divide(climate_1yr.select(f"y{k}"))
                .addBands(ee.Image(k).float())
                .rename(["ndvi", "year"])
                .set({"year": k})
            )
            img_coll = img_coll.add(ndvi_img)

        return ee.ImageCollection(img_coll)

    return f_img_coll(ndvi_1yr)


def productivity_series(
    year_initial, year_final, method, prod_asset, climate_asset, logger
):
    logger.debug("Entering productivity_trajectory function.")

    if climate_asset:
        climate_1yr = ee.Image(climate_asset)
        climate_1yr = climate_1yr.where(climate_1yr.eq(9999), -32768)
        climate_1yr = climate_1yr.updateMask(climate_1yr.neq(-32768))
    elif method != "ndvi_trend":
        raise GEEIOError("Must specify a climate dataset")

    ndvi_dataset = ee.Image(prod_asset)
    ndvi_dataset = ndvi_dataset.where(ndvi_dataset.eq(9999), -32768)
    ndvi_dataset = ndvi_dataset.updateMask(ndvi_dataset.neq(-32768))

    if method == "ndvi_trend":
        return ndvi(year_initial, year_final, ndvi_dataset, logger)
    elif method == "p_restrend":
        return p_residuals(year_initial, year_final, ndvi_dataset, climate_1yr, logger)
    elif method == "ue":
        return ue(year_initial, year_final, ndvi_dataset, climate_1yr, logger)
    else:
        raise GEEIOError(f"Unrecognized method '{method}'")


def productivity_trajectory(
    year_initial, year_final, method, prod_asset, climate_asset, logger
):
    logger.debug("Entering productivity_trajectory function.")

    climate_1yr = ee.Image(climate_asset)
    climate_1yr = climate_1yr.where(climate_1yr.eq(9999), -32768)
    climate_1yr = climate_1yr.updateMask(climate_1yr.neq(-32768))

    if climate_asset is None and method != "ndvi_trend":
        raise GEEIOError("Must specify a climate dataset")

    ndvi_dataset = ee.Image(prod_asset)
    ndvi_dataset = ndvi_dataset.where(ndvi_dataset.eq(9999), -32768)
    ndvi_dataset = ndvi_dataset.updateMask(ndvi_dataset.neq(-32768))

    if method == "ndvi_trend":
        lf_trend, mk_trend = ndvi_trend(year_initial, year_final, ndvi_dataset, logger)
    elif method == "p_restrend":
        lf_trend, mk_trend = p_restrend(
            year_initial, year_final, ndvi_dataset, climate_1yr, logger
        )

    elif method == "ue":
        lf_trend, mk_trend = ue_trend(
            year_initial, year_final, ndvi_dataset, climate_1yr, logger
        )
    else:
        raise GEEIOError(f"Unrecognized method '{method}'")

    # Define Kendall parameter values for a significance of 0.05
    period = year_final - year_initial + 1
    kendall90 = stats.get_kendall_coef(period, 90)
    kendall95 = stats.get_kendall_coef(period, 95)
    kendall99 = stats.get_kendall_coef(period, 99)

    # Create final productivity trajectory output layer. Positive values are
    # significant increase, negative values are significant decrease.
    signif = (
        ee.Image(-32768)
        .where(lf_trend.select("scale").gt(0).And(mk_trend.abs().gte(kendall90)), 1)
        .where(lf_trend.select("scale").gt(0).And(mk_trend.abs().gte(kendall95)), 2)
        .where(lf_trend.select("scale").gt(0).And(mk_trend.abs().gte(kendall99)), 3)
        .where(lf_trend.select("scale").lt(0).And(mk_trend.abs().gte(kendall90)), -1)
        .where(lf_trend.select("scale").lt(0).And(mk_trend.abs().gte(kendall95)), -2)
        .where(lf_trend.select("scale").lt(0).And(mk_trend.abs().gte(kendall99)), -3)
        .where(mk_trend.abs().lte(kendall90), 0)
        .where(lf_trend.select("scale").abs().lte(10), 0)
    )
    trend = lf_trend.select("scale").rename("Productivity_trend")
    signif = signif.rename("Productivity_significance")
    mk_trend = mk_trend.rename("Productivity_ann_mean")

    return TEImage(
        trend.addBands(signif).addBands(mk_trend).unmask(-32768).int16(),
        [
            BandInfo(
                "Productivity trajectory (trend)",
                metadata={"year_initial": year_initial, "year_final": year_final},
            ),
            BandInfo(
                "Productivity trajectory (significance)",
                add_to_map=True,
                metadata={"year_initial": year_initial, "year_final": year_final},
            ),
            BandInfo(
                "Mean annual NDVI integral",
                metadata={"year_initial": year_initial, "year_final": year_final},
            ),
        ],
    )


def productivity_performance(year_initial, year_final, prod_asset, geojson, logger):
    logger.debug("Entering productivity_performance function.")

    ndvi_1yr = ee.Image(prod_asset)
    ndvi_1yr = ndvi_1yr.where(ndvi_1yr.eq(9999), -32768)
    ndvi_1yr = ndvi_1yr.updateMask(ndvi_1yr.neq(-32768))

    # land cover data from esa cci
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # global agroecological zones from IIASA
    soil_tax_usda = ee.Image(
        "users/geflanddegradation/toolbox_datasets/soil_tax_usda_sgrid"
    )

    # Make sure the bounding box of the poly is used, and not the geodesic
    # version, for the clipping
    poly = ee.Geometry(geojson, opt_geodesic=False)

    # compute mean ndvi for the period
    ndvi_avg = (
        ndvi_1yr.select(ee.List([f"y{i}" for i in range(year_initial, year_final + 1)]))
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
        .clip(poly)
    )

    # Handle case of year_initial that isn't included in the CCI data

    if year_initial > 2020:
        lc_year_initial = 2020
    elif year_initial < 1992:
        lc_year_initial = 1992
    else:
        lc_year_initial = year_initial
    # reclassify lc to ipcc classes
    lc_t0 = lc.select(f"y{lc_year_initial}").remap(
        [
            10,
            11,
            12,
            20,
            30,
            40,
            50,
            60,
            61,
            62,
            70,
            71,
            72,
            80,
            81,
            82,
            90,
            100,
            160,
            170,
            110,
            130,
            180,
            190,
            120,
            121,
            122,
            140,
            150,
            151,
            152,
            153,
            200,
            201,
            202,
            210,
        ],
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
        ],
    )

    # create a binary mask.
    mask = ndvi_avg.neq(0)

    # define modis projection attributes
    modis_proj = ee.Image(
        "users/geflanddegradation/toolbox_datasets/ndvi_modis_2001_2023"
    ).projection()

    # reproject land cover, soil_tax_usda and avhrr to modis resolution
    lc_proj = lc_t0.reproject(crs=modis_proj)
    soil_tax_usda_proj = soil_tax_usda.reproject(crs=modis_proj)
    ndvi_avg_proj = ndvi_avg.reproject(crs=modis_proj)

    # define unit of analysis as the intersect of soil_tax_usda and land cover
    units = soil_tax_usda_proj.multiply(100).add(lc_proj)

    # create a 2 band raster to compute 90th percentile per unit (analysis restricted by mask and study area)
    ndvi_id = ndvi_avg_proj.addBands(units).updateMask(mask)

    # compute 90th percentile by unit
    perc90 = ndvi_id.reduceRegion(
        reducer=ee.Reducer.percentile([90]).group(groupField=1, groupName="code"),
        geometry=poly,
        scale=ee.Number(modis_proj.nominalScale()).getInfo(),
        maxPixels=1e15,
    )

    # Extract the cluster IDs and the 90th percentile
    groups = ee.List(perc90.get("groups"))
    ids = groups.map(lambda d: ee.Dictionary(d).get("code"))
    perc = groups.map(lambda d: ee.Dictionary(d).get("p90"))

    if len(ids.getInfo()) > 0:
        # remap the units raster using their 90th percentile value
        raster_perc = units.remap(ids, perc)

        # compute the ration of observed ndvi to 90th for that class
        obs_ratio = ndvi_avg_proj.divide(raster_perc)

        # aggregate obs_ratio to original NDVI data resolution (for modis this step does not change anything)
        obs_ratio_2 = obs_ratio.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=2000
        ).reproject(crs=ndvi_1yr.projection())
        prod_perf_ratio = obs_ratio_2.multiply(10000).rename(
            "Productivity_performance_ratio"
        )
    else:
        logger.debug("unable to calculate ids - setting performance to -32768")
        obs_ratio_2 = ee.Image(-32768)
        prod_perf_ratio = ee.Image(-32768).rename("Productivity_performance_ratio")

    # create final degradation output layer (-32768 is background), 0 is not
    # degraded, -1 is degraded
    lp_perf_deg = (
        ee.Image(-32768)
        .where(obs_ratio_2.gte(0.5), 0)
        .where(obs_ratio_2.lte(0.5), -1)
        .where(obs_ratio_2.eq(-32768), -32768)
    )

    lp_perf_deg = lp_perf_deg.rename("Productivity_performance_degradation")
    units = units.rename("Productivity_performance_units")

    return TEImage(
        lp_perf_deg.addBands(prod_perf_ratio).addBands(units).unmask(-32768).int16(),
        [
            BandInfo(
                "Productivity performance (degradation)",
                add_to_map=True,
                metadata={"year_initial": year_initial, "year_final": year_final},
            ),
            BandInfo(
                "Productivity performance (ratio)",
                metadata={"year_initial": year_initial, "year_final": year_final},
            ),
            BandInfo(
                "Productivity performance (units)",
                metadata={"year_initial": year_initial},
            ),
        ],
    )


def productivity_state(
    year_bl_start, year_bl_end, year_tg_start, year_tg_end, prod_asset, logger
):
    logger.debug("Entering productivity_state function.")

    ndvi_1yr = ee.Image(prod_asset)

    # compute min and max of annual ndvi for the baseline period
    bl_ndvi_range = ndvi_1yr.select(
        ee.List([f"y{i}" for i in range(year_bl_start, year_bl_end + 1)])
    ).reduce(ee.Reducer.percentile([0, 100]))

    # add two bands to the time series: one 5% lower than min and one 5% higher
    # than max
    bl_ndvi_ext = (
        ndvi_1yr.select(
            ee.List([f"y{i}" for i in range(year_bl_start, year_bl_end + 1)])
        )
        .addBands(
            bl_ndvi_range.select("p0").subtract(
                (
                    bl_ndvi_range.select("p100").subtract(bl_ndvi_range.select("p0"))
                ).multiply(0.05)
            )
        )
        .addBands(
            bl_ndvi_range.select("p100").add(
                (
                    bl_ndvi_range.select("p100").subtract(bl_ndvi_range.select("p0"))
                ).multiply(0.05)
            )
        )
    )

    # compute percentiles of annual ndvi for the extended baseline period
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bl_ndvi_perc = bl_ndvi_ext.reduce(ee.Reducer.percentile(percentiles))

    # compute mean ndvi for the baseline and target period period
    bl_ndvi_mean = (
        ndvi_1yr.select(
            ee.List([f"y{i}" for i in range(year_bl_start, year_bl_end + 1)])
        )
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
    )
    tg_ndvi_mean = (
        ndvi_1yr.select(
            ee.List([f"y{i}" for i in range(year_tg_start, year_tg_end + 1)])
        )
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
    )

    # reclassify mean ndvi for baseline period based on the percentiles
    bl_classes = (
        ee.Image(-32768)
        .where(bl_ndvi_mean.lte(bl_ndvi_perc.select("p10")), 1)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p10")), 2)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p20")), 3)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p30")), 4)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p40")), 5)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p50")), 6)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p60")), 7)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p70")), 8)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p80")), 9)
        .where(bl_ndvi_mean.gt(bl_ndvi_perc.select("p90")), 10)
    )

    # reclassify mean ndvi for target period based on the percentiles
    tg_classes = (
        ee.Image(-32768)
        .where(tg_ndvi_mean.lte(bl_ndvi_perc.select("p10")), 1)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p10")), 2)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p20")), 3)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p30")), 4)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p40")), 5)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p50")), 6)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p60")), 7)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p70")), 8)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p80")), 9)
        .where(tg_ndvi_mean.gt(bl_ndvi_perc.select("p90")), 10)
    )

    # difference between start and end clusters >= 2 means improvement (<= -2
    # is degradation)
    classes_chg = tg_classes.subtract(bl_classes).where(
        bl_ndvi_mean.subtract(tg_ndvi_mean).abs().lte(100), 0
    )

    classes_chg = classes_chg.rename("Productivity_state_degradation")
    bl_classes = bl_classes.rename(
        f"Productivity_state_classes_{year_bl_start}-{year_bl_end}"
    )
    tg_classes = tg_classes.rename(
        f"Productivity_state_classes_{year_tg_start}-{year_tg_end}"
    )
    bl_ndvi_mean = bl_ndvi_mean.rename(
        f"Productivity_state_NDVI_mean_{year_bl_start}-{year_bl_end}"
    )
    tg_ndvi_mean = tg_ndvi_mean.rename(
        f"Productivity_state_NDVI_mean_{year_tg_start}-{year_tg_end}"
    )
    band_infos = [
        BandInfo(
            "Productivity state (degradation)",
            add_to_map=True,
            metadata={
                "year_bl_start": year_bl_start,
                "year_bl_end": year_bl_end,
                "year_tg_start": year_tg_start,
                "year_tg_end": year_tg_end,
            },
        ),
        BandInfo(
            "Productivity state classes",
            metadata={"year_initial": year_bl_start, "year_final": year_bl_end},
        ),
        BandInfo(
            "Productivity state classes",
            metadata={"year_initial": year_tg_start, "year_final": year_tg_end},
        ),
        BandInfo(
            "Productivity state NDVI mean",
            metadata={"year_initial": year_bl_start, "year_final": year_bl_end},
        ),
        BandInfo(
            "Productivity state NDVI mean",
            metadata={"year_initial": year_tg_start, "year_final": year_tg_end},
        ),
    ]

    return TEImage(
        classes_chg.addBands(bl_classes)
        .addBands(tg_classes)
        .addBands(bl_ndvi_mean)
        .addBands(tg_ndvi_mean)
        .int16(),
        band_infos,
    )


def productivity_faowocat(
    low_biomass=0.4,
    medium_biomass=0.55,
    high_biomass=0.7,
    years_interval=15,
    modis_mode="MannKendal + MTID",
    prod_asset=None,
    logger=None,
):
    logger.debug("Entering productivity_faowocat function.")

    if prod_asset is None:
        raise GEEIOError("Must specify a prod_asset")

    ndvi_dataset = ee.Image(prod_asset)
    ndvi_dataset = ndvi_dataset.where(ndvi_dataset.eq(9999), -32768)
    ndvi_dataset = ndvi_dataset.updateMask(ndvi_dataset.neq(-32768))

    year_start = 2001
    year_end = year_start + years_interval - 1

    if modis_mode is None:
        modis_mode = "MannKendal + MTID"
    modis_mode = modis_mode.strip()
    if modis_mode not in ("MannKendal", "MannKendal + MTID"):
        logger.warning("Unknown modis_mode '%s' – falling back to 'MannKendal + MTID'", modis_mode)
        modis_mode = "MannKendal + MTID"

    years = list(range(year_start, year_end + 1))

    def _add_img(y, lst):
        img = (ndvi_dataset.select(f"y{y}")
               .rename("NDVI")
               .set({"year": y}))
        return ee.List(lst).add(img)

    annual_ic = ee.ImageCollection(ee.List(years).iterate(_add_img, []))

    lf_trend, mk_trend = ndvi_trend(year_start, year_end, ndvi_dataset, logger)
    period = year_end - year_start + 1
    kendall_s = stats.get_kendall_coef(period, 95)

    trend_3cat = (ee.Image(0)
                  .where(lf_trend.select("scale").lt(0).And(mk_trend.abs().gte(kendall_s)), 1)
                  .where(mk_trend.abs().lt(kendall_s), 2)
                  .where(lf_trend.select("scale").gt(0).And(mk_trend.abs().gte(kendall_s)), 3))

    def _mtid(ic):
        last_mean = (ic.filter(ee.Filter.gt("year", year_end - 3))
                     .select("NDVI").mean())
        diffs = ic.map(lambda im: last_mean.subtract(im.select("NDVI")))
        return ee.ImageCollection(diffs).sum()

    mtid = _mtid(annual_ic)
    mtid_code = (ee.Image(0).where(mtid.lte(0), 1).where(mtid.gt(0), 2))

    if modis_mode == "MannKendal + MTID":
        steadiness = (
            ee.Image(0)
            .where(trend_3cat.eq(1).And(mtid_code.eq(1)), 1)
            .where(trend_3cat.eq(1).And(mtid_code.eq(2)), 2)
            .where(trend_3cat.eq(2).And(mtid_code.eq(1)), 2)
            .where(trend_3cat.eq(2).And(mtid_code.eq(2)), 3)
            .where(trend_3cat.eq(3).And(mtid_code.eq(1)), 3)
            .where(trend_3cat.eq(3).And(mtid_code.eq(2)), 4)
        )
    else:
        steadiness = (
            ee.Image(0)
            .where(trend_3cat.eq(1), 1)
            .where(trend_3cat.eq(2).And(mk_trend.lt(0)), 2)
            .where(trend_3cat.eq(2).And(mk_trend.gt(0)), 3)
            .where(trend_3cat.eq(3), 4)
        )

    init_mean = (annual_ic.filter(ee.Filter.lte("year", year_start + 2))
                 .select("NDVI").mean().divide(10000))
    init_biomass = (
        ee.Image(0)
        .where(init_mean.lte(low_biomass), 1)
        .where(init_mean.gt(low_biomass).And(init_mean.lte(high_biomass)), 2)
        .where(init_mean.gt(high_biomass), 3)
    )

    baseline_end = year_start + max(15, years_interval) - 1
    baseline_ic = annual_ic.filter(ee.Filter.lte("year", baseline_end)).select("NDVI")
    pct = baseline_ic.reduce(
        ee.Reducer.percentile([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))

    def _pct_class(mean_img):
        return (
            ee.Image(-32768)
            .where(mean_img.lte(pct.select("NDVI_p10")), 1)
            .where(mean_img.gt(pct.select("NDVI_p10")), 2)
            .where(mean_img.gt(pct.select("NDVI_p20")), 3)
            .where(mean_img.gt(pct.select("NDVI_p30")), 4)
            .where(mean_img.gt(pct.select("NDVI_p40")), 5)
            .where(mean_img.gt(pct.select("NDVI_p50")), 6)
            .where(mean_img.gt(pct.select("NDVI_p60")), 7)
            .where(mean_img.gt(pct.select("NDVI_p70")), 8)
            .where(mean_img.gt(pct.select("NDVI_p80")), 9)
            .where(mean_img.gt(pct.select("NDVI_p90")), 10)
        )

    t1_mean = (annual_ic.filter(ee.Filter.lte("year", year_start + 3))
               .select("NDVI").mean())
    t2_mean = (annual_ic.filter(ee.Filter.gte("year", year_end - 3))
               .select("NDVI").mean())
    t1_class = _pct_class(t1_mean)
    t2_class = _pct_class(t2_mean)

    diff_class = t2_class.subtract(t1_class)
    eme_state = (ee.Image(0)
                 .where(diff_class.lte(-2), 1)
                 .where(diff_class.gt(-2).And(diff_class.lt(2)), 2)
                 .where(diff_class.gte(2), 3))

    semi = ee.Image().expression(
        '(a==1&&b==1&&c==1)?1:'
        '(a==1&&b==2&&c==1)?2:'
        '(a==1&&b==3&&c==1)?3:'
        '(a==1&&b==1&&c==2)?4:'
        '(a==1&&b==2&&c==2)?5:'
        '(a==1&&b==3&&c==2)?6:'
        '(a==1&&b==1&&c==3)?7:'
        '(a==1&&b==2&&c==3)?8:'
        '(a==1&&b==3&&c==3)?9:'
        '(a==2&&b==1&&c==1)?10:'
        '(a==2&&b==2&&c==1)?11:'
        '(a==2&&b==3&&c==1)?12:'
        '(a==2&&b==1&&c==2)?13:'
        '(a==2&&b==2&&c==2)?14:'
        '(a==2&&b==3&&c==2)?15:'
        '(a==2&&b==1&&c==3)?16:'
        '(a==2&&b==2&&c==3)?17:'
        '(a==2&&b==3&&c==3)?18:'
        '(a==3&&b==1&&c==1)?19:'
        '(a==3&&b==2&&c==1)?20:'
        '(a==3&&b==3&&c==1)?21:'
        '(a==3&&b==1&&c==2)?22:'
        '(a==3&&b==2&&c==2)?23:'
        '(a==3&&b==3&&c==2)?24:'
        '(a==3&&b==1&&c==3)?25:'
        '(a==3&&b==2&&c==3)?26:'
        '(a==3&&b==3&&c==3)?27:'
        '(a==4&&b==1&&c==1)?28:'
        '(a==4&&b==2&&c==1)?29:'
        '(a==4&&b==3&&c==1)?30:'
        '(a==4&&b==1&&c==2)?31:'
        '(a==4&&b==2&&c==2)?32:'
        '(a==4&&b==3&&c==2)?33:'
        '(a==4&&b==1&&c==3)?34:'
        '(a==4&&b==2&&c==3)?35:'
        '(a==4&&b==3&&c==3)?36:99',
        {'a': steadiness, 'b': init_biomass, 'c': eme_state})

    final_lpd = (ee.Image(0)
                 .where(semi.lte(8), 1)
                 .where(semi.gt(8).And(semi.lte(14)), 2)
                 .where(semi.gt(14).And(semi.lte(22)), 3)
                 .where(semi.gt(22).And(semi.lte(32)), 4)
                 .where(semi.gt(32).And(semi.lte(36)), 5)
                 .rename("LPD"))

    return TEImage(
        final_lpd.unmask(-32768).int16(),
        [BandInfo("Land Productivity Dynamics (FAO-WOCAT)",
                  add_to_map=True,
                  metadata={"year_initial": year_start,
                            "year_final": year_end})],
    )
