import re

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


def productivity_performance(
    year_initial, year_final, prod_asset, all_geojsons, logger
):
    """
    Calculate productivity performance using unified percentiles across all geojson areas.

    Args:
        year_initial: Starting year for the analysis period
        year_final: Ending year for the analysis period
        prod_asset: Asset ID for the productivity dataset
        all_geojsons: List of geojson dictionaries for unified percentile calculation
        logger: Logger instance

    Returns:
        TEImage object with global productivity performance results
    """
    logger.debug(f"Performance period: {year_initial}-{year_final}")
    logger.debug(f"Productivity asset: {prod_asset}")
    logger.debug(f"Using unified percentiles across {len(all_geojsons)} geojson areas")

    if not all_geojsons or len(all_geojsons) == 0:
        raise ValueError("all_geojsons must contain at least one geojson")

    ndvi_1yr = ee.Image(prod_asset)
    ndvi_1yr = ndvi_1yr.where(ndvi_1yr.eq(9999), -32768)
    ndvi_1yr = ndvi_1yr.updateMask(ndvi_1yr.neq(-32768))

    # land cover data from esa cci
    logger.debug("Loading land cover data from ESA CCI...")
    lc = ee.Image("users/geflanddegradation/toolbox_datasets/lcov_esacc_1992_2022")
    lc = lc.where(lc.eq(9999), -32768)
    lc = lc.updateMask(lc.neq(-32768))

    # global agroecological zones from IIASA
    logger.debug("Loading soil taxonomy data...")
    soil_tax_usda = ee.Image(
        "users/geflanddegradation/toolbox_datasets/soil_tax_usda_sgrid"
    )

    # Create unified geometry for percentile calculation across all areas
    logger.debug("Creating unified geometry from all geojsons...")
    individual_polys = [ee.Geometry(gj, None, False) for gj in all_geojsons]

    try:
        if len(individual_polys) == 1:
            unified_poly = individual_polys[0]
            logger.debug("Single geometry - no union needed")
        else:
            # Use MultiPolygon constructor instead of GeometryCollection
            unified_poly = ee.Geometry.MultiPolygon(individual_polys).dissolve(
                maxError=1000
            )
            logger.debug("Unified geometry (dissolve) created successfully")
    except Exception as e:
        logger.warning(f"Failed to create MultiPolygon geometry: {e}")
        # Fallback to progressive union
        unified_poly = individual_polys[0]
        for poly in individual_polys[1:]:
            unified_poly = unified_poly.union(poly, maxError=1000)
        logger.debug("Using progressive union fallback for unified geometry")

    # compute mean ndvi for the period globally (not clipped)
    logger.debug(f"Computing global mean NDVI for period {year_initial}-{year_final}")
    ndvi_avg_global = (
        ndvi_1yr.select(ee.List([f"y{i}" for i in range(year_initial, year_final + 1)]))
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
    )

    # compute mean ndvi for the unified area for percentile calculation
    ndvi_avg_unified = ndvi_avg_global.clip(unified_poly)

    # Handle case of year_initial that isn't included in the CCI data

    if year_initial > 2020:
        lc_year_initial = 2020
    elif year_initial < 1992:
        lc_year_initial = 1992
    else:
        lc_year_initial = year_initial

    # remap ESA CCI land cover classes to simplified IPCC classes
    # ESA CCI land cover class remapping to IPCC classes
    # fmt: off
    esa_cci_classes = [
        10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100,
        160, 170, 110, 130, 180, 190, 120, 121, 122, 140, 150, 151, 152, 153,
        200, 201, 202, 210
    ]

    recode_classes = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36
    ]
    # fmt: on

    lc_t0 = lc.select(f"y{lc_year_initial}").remap(esa_cci_classes, recode_classes)

    # define modis projection attributes
    modis_proj = ee.Image(
        "users/geflanddegradation/toolbox_datasets/ndvi_modis_2001_2024"
    ).projection()

    # reproject land cover, soil_tax_usda and unified ndvi to modis resolution
    lc_proj = lc_t0.reproject(crs=modis_proj)
    soil_tax_usda_proj = soil_tax_usda.reproject(crs=modis_proj)
    ndvi_avg_unified_proj = ndvi_avg_unified.reproject(crs=modis_proj)

    # define unit of analysis as the intersect of soil_tax_usda and land cover (globally)
    global_units = soil_tax_usda_proj.multiply(100).add(lc_proj)

    # Use unified area for percentile calculation
    percentile_poly = unified_poly
    percentile_ndvi = ndvi_avg_unified_proj
    percentile_mask = percentile_ndvi.neq(0).And(percentile_ndvi.gt(-32768))

    # create a 2 band raster to compute 90th percentile per unit (analysis restricted by mask and study area)
    logger.debug("Preparing data for percentile calculation...")
    # Clip global_units to the percentile area to reduce computation
    global_units_clipped = global_units.clip(percentile_poly)
    ndvi_id = percentile_ndvi.addBands(global_units_clipped).updateMask(percentile_mask)

    logger.debug("Starting 90th percentile calculation by land cover/soil units")
    modis_scale = modis_proj.nominalScale().getInfo()
    logger.debug(f"MODIS projection scale: {modis_scale:,} meters")

    # Smart subsampling for very large areas
    scale = modis_scale
    subsample_factor = 1
    use_subsampling = False
    final_pixels = 0  # Initialize final_pixels

    try:
        # Get approximate area for subsampling decisions - avoid expensive getInfo() when possible
        # Use a more conservative approach for area estimation to avoid multiple GEE calls
        area_sq_km = percentile_poly.area(maxError=10000).divide(1000000)

        try:
            area_value = area_sq_km.getInfo()
            logger.debug(f"Processing area for percentiles: {area_value:,.2f} sq km")
            estimated_pixels = area_value * 1000000 / (scale * scale)
        except Exception:
            logger.debug("Could not get exact area - using conservative large estimate")
            estimated_pixels = 50_000_000
            area_value = estimated_pixels * (scale * scale) / 1000000

        logger.debug(f"Estimated pixels to process: {estimated_pixels:,.0f}")

        # Use smart subsampling for very large areas
        target_pixels = 25_000_000

        # Trigger subsampling only for large areas
        if estimated_pixels >= 50_000_000:
            calculated_factor = int((estimated_pixels / target_pixels) ** 0.5)
            max_scale_factor = 20
            subsample_factor = max(2, min(calculated_factor, max_scale_factor))
            use_subsampling = True
            final_pixels = estimated_pixels / (subsample_factor**2)
            logger.info(
                f"Very large area detected ({estimated_pixels:,.0f} pixels) - using subsampling factor {subsample_factor}"
            )
            logger.info(
                f"This will reduce computation from {estimated_pixels:,.0f} to ~{final_pixels:,.0f} pixels"
            )
            logger.info(f"Final resolution: {scale * subsample_factor:,.0f}m")

            if calculated_factor > max_scale_factor:
                logger.warning(
                    f"Calculated scale factor ({calculated_factor}) exceeds maximum ({max_scale_factor})"
                )
        else:
            final_pixels = estimated_pixels
            logger.debug(
                f"Area ({estimated_pixels:,.0f} pixels) within normal processing range - no subsampling needed"
            )

        if estimated_pixels > 1e12:  # 1 trillion pixels (reduced threshold)
            logger.warning(
                f"Extremely large area detected - {estimated_pixels:,.0f} pixels may still cause GEE timeout even with maximum subsampling"
            )
    except Exception as e:
        logger.debug(f"Could not estimate processing area: {e}")
        final_pixels = 10_000_000  # Default assumption for fallback

    # Apply subsampling if needed
    if use_subsampling:
        logger.debug(f"Applying systematic subsampling with factor {subsample_factor}")
        final_scale = scale * subsample_factor
    else:
        final_scale = scale

    # compute 90th percentile by unit
    logger.debug("Executing reduceRegion operation for 90th percentiles...")
    # Calculate percentiles grouped by land cover/soil unit
    percentile_reducer = ee.Reducer.percentile([90]).group(
        groupField=1, groupName="code"
    )
    try:
        # Execute reduceRegion with consistent settings
        perc90_results = ndvi_id.reduceRegion(
            reducer=percentile_reducer,
            geometry=percentile_poly,
            scale=final_scale,
            maxPixels=1e10,  # 10 billion pixels - consistent limit
            bestEffort=True,  # Allow GEE to optimize computation
        )

        logger.debug("Percentile calculation completed successfully")

    except Exception as e:
        logger.error(f"reduceRegion operation failed: {e}")
        # Try one more time with even more aggressive settings
        try:
            logger.debug("Retrying with maximum memory optimization...")

            perc90_results = ndvi_id.reduceRegion(
                reducer=percentile_reducer,
                geometry=percentile_poly,
                scale=final_scale * 2,  # Double the scale to reduce pixels further
                maxPixels=1e9,  # Much smaller maxPixels
                bestEffort=True,
            )
            logger.debug("Retry successful with reduced resolution")
        except Exception as retry_error:
            logger.error(f"Retry also failed: {retry_error}")
            raise

    perc90 = {"groups": perc90_results.get("groups")}

    # Extract the cluster IDs and the 90th percentile (server-side only)
    logger.debug("Extracting cluster IDs and percentiles from results (server-side)...")
    groups = ee.List(perc90.get("groups"))
    ids = groups.map(lambda d: ee.Dictionary(d).get("code"))
    perc = groups.map(lambda d: ee.Dictionary(d).get("p90"))

    logger.debug("Processing clusters to create global performance raster...")

    # Apply the percentiles globally (not clipped to individual tiles)
    logger.debug("Applying percentiles to global NDVI data...")
    global_ndvi_proj = ndvi_avg_global.reproject(crs=modis_proj)
    global_units = soil_tax_usda_proj.multiply(100).add(lc_proj)

    # Remap the units raster using their 90th percentile value (calculated from unified area)
    # Provide a default value and then mask it out to avoid using unmapped units
    raster_perc = global_units.remap(ids, perc, -32768)
    # Mask out missing (-32768) and zero percentiles to avoid divide-by-zero
    raster_perc = raster_perc.updateMask(raster_perc.neq(-32768)).updateMask(
        raster_perc.neq(0)
    )

    # Compute the ratio of observed NDVI to 90th for that class
    obs_ratio = global_ndvi_proj.divide(raster_perc)

    # Aggregate obs_ratio to original NDVI data resolution (for MODIS this step may be a no-op)
    logger.debug("Aggregating results to original NDVI resolution...")
    obs_ratio_2 = obs_ratio.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=2000
    ).reproject(crs=ndvi_1yr.projection())

    prod_perf_ratio = obs_ratio_2.multiply(10000).rename(
        "Productivity_performance_ratio"
    )
    logger.debug("Productivity performance calculation completed successfully")

    # create final degradation output layer (-32768 is background), 0 is not
    # degraded, -1 is degraded
    lp_perf_deg = (
        ee.Image(-32768)
        .where(obs_ratio_2.gte(0.5), 0)
        .where(obs_ratio_2.lte(0.5), -1)
        .where(obs_ratio_2.eq(-32768), -32768)
    )

    lp_perf_deg = lp_perf_deg.rename("Productivity_performance_degradation")
    global_units = global_units.rename("Productivity_performance_units")

    return TEImage(
        lp_perf_deg.addBands(prod_perf_ratio)
        .addBands(global_units)
        .unmask(-32768)
        .int16(),
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
    high_biomass=0.7,
    years_interval=3,
    modis_mode="MannKendal + MTID",
    year_initial=2001,
    year_final=2015,
    prod_asset=None,
    logger=None,
):
    """Compute FAO‑WOCAT Land Productivity Dynamics (LPD) and the Productivity
    State diagram in one call.

    Notes:
      * `year_final` is REQUIRED for the FAO‑WOCAT dynamics period.
      * `years_interval` (default 3) controls the baseline/target windows used
        for the Productivity State calculation.
    """
    logger.debug("Entering productivity_faowocat function.")

    if prod_asset is None:
        raise GEEIOError("Must specify a prod_asset")

    if year_final is None:
        raise GEEIOError("Must specify 'year_final' for FAO-WOCAT dynamics")

    ndvi_dataset = ee.Image(prod_asset)
    ndvi_dataset = ndvi_dataset.where(ndvi_dataset.eq(9999), -32768)
    ndvi_dataset = ndvi_dataset.updateMask(ndvi_dataset.neq(-32768))

    # Ensure we have annual bands yYYYY for the full [year_initial, year_final]
    band_names = ndvi_dataset.bandNames().getInfo()
    has_annual = any(re.match(r"^y\d{4}$", bn) for bn in band_names)

    def _annual_imgs_mean(year):
        """Compute mean NDVI for *year* and rename to 'yYYYY'. If the source
        data are already annual, just pick that band; if they are daily/monthly
        (e.g. 'dYYYY_MM_DD'), compute the yearly mean. If nothing exists for
        that year, return a fully masked band named 'yYYYY'."""
        if has_annual:
            try:
                return ndvi_dataset.select(f"y{year}").rename(f"y{year}")
            except Exception:
                return (
                    ee.Image.constant(-32768).rename(f"y{year}").updateMask(ee.Image(0))
                )
        else:
            year_bands = [bn for bn in band_names if bn.startswith(f"d{year}_")]
            if not year_bands:
                return (
                    ee.Image.constant(-32768).rename(f"y{year}").updateMask(ee.Image(0))
                )
            return (
                ndvi_dataset.select(year_bands)
                .reduce(ee.Reducer.mean())
                .rename(f"y{year}")
            )

    annual_years = list(range(year_initial, year_final + 1))
    if not has_annual or any(f"y{y}" not in band_names for y in annual_years):
        ndvi_dataset = ee.Image.cat([_annual_imgs_mean(y) for y in annual_years])

    year_end = year_final

    def _annual_mean(ic_img, years):
        images = []
        # Supports both yYYYY and dYYYY_* inputs; after the conversion above,
        bnames = ic_img.bandNames().getInfo()
        for year in years:
            y_bands = [b for b in bnames if (f"d{year}_" in b)]
            if y_bands:
                img = (
                    ic_img.select(y_bands).reduce(ee.Reducer.mean()).rename(f"y{year}")
                )
            else:
                try:
                    img = ic_img.select(f"y{year}").rename(f"y{year}")
                except Exception:
                    img = (
                        ee.Image.constant(-32768)
                        .rename(f"y{year}")
                        .updateMask(ee.Image(0))
                    )
            images.append(img.set({"year": year}))
        return ee.ImageCollection(images)

    years = list(range(year_initial, year_end + 1))
    annual_ic = _annual_mean(ndvi_dataset, years).map(
        lambda img: img.rename("NDVI").set({"year": img.get("year")})
    )

    # Trend + MK significance
    lf_trend, mk_trend = ndvi_trend(year_initial, year_end, ndvi_dataset, logger)
    period = year_end - year_initial + 1
    kendall_s = stats.get_kendall_coef(period, 95)

    trend_3cat = (
        ee.Image(0)
        .where(lf_trend.select("scale").lt(0).And(mk_trend.abs().gte(kendall_s)), 1)
        .where(mk_trend.abs().lt(kendall_s), 2)
        .where(lf_trend.select("scale").gt(0).And(mk_trend.abs().gte(kendall_s)), 3)
    )

    def _mtid(ic):
        last_mean = ic.filter(ee.Filter.gt("year", year_end - 3)).select("NDVI").mean()
        diffs = ic.map(lambda im: last_mean.subtract(im.select("NDVI")))
        return ee.ImageCollection(diffs).sum()

    mtid = _mtid(annual_ic)
    mtid_code = ee.Image(0).where(mtid.lte(0), 1).where(mtid.gt(0), 2)

    if modis_mode is None:
        modis_mode = "MannKendal + MTID"
    modis_mode = modis_mode.strip()
    if modis_mode not in ("MannKendal", "MannKendal + MTID"):
        logger.warning(
            "Unknown modis_mode '%s' – falling back to 'MannKendal + MTID'", modis_mode
        )
        modis_mode = "MannKendal + MTID"

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

    init_mean = (
        annual_ic.filter(ee.Filter.lte("year", year_initial + 2))
        .select("NDVI")
        .mean()
        .divide(10000)
    )
    init_biomass = (
        ee.Image(0)
        .where(init_mean.lte(low_biomass), 1)
        .where(init_mean.gt(low_biomass).And(init_mean.lte(high_biomass)), 2)
        .where(init_mean.gt(high_biomass), 3)
    )

    baseline_end = year_end
    baseline_ic = annual_ic.filter(ee.Filter.lte("year", baseline_end)).select("NDVI")
    pct = baseline_ic.reduce(
        ee.Reducer.percentile([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    )

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

    t1_mean = (
        annual_ic.filter(ee.Filter.lte("year", year_initial + 3)).select("NDVI").mean()
    )
    t2_mean = (
        annual_ic.filter(ee.Filter.gte("year", year_end - 3)).select("NDVI").mean()
    )
    t1_class = _pct_class(t1_mean)
    t2_class = _pct_class(t2_mean)

    diff_class = t2_class.subtract(t1_class)
    eme_state = (
        ee.Image(0)
        .where(diff_class.lte(-2), 1)
        .where(diff_class.gt(-2).And(diff_class.lt(2)), 2)
        .where(diff_class.gte(2), 3)
    )

    semi = ee.Image().expression(
        "(a==1&&b==1&&c==1)?1:"
        "(a==1&&b==2&&c==1)?2:"
        "(a==1&&b==3&&c==1)?3:"
        "(a==1&&b==1&&c==2)?4:"
        "(a==1&&b==2&&c==2)?5:"
        "(a==1&&b==3&&c==2)?6:"
        "(a==1&&b==1&&c==3)?7:"
        "(a==1&&b==2&&c==3)?8:"
        "(a==1&&b==3&&c==3)?9:"
        "(a==2&&b==1&&c==1)?10:"
        "(a==2&&b==2&&c==1)?11:"
        "(a==2&&b==3&&c==1)?12:"
        "(a==2&&b==1&&c==2)?13:"
        "(a==2&&b==2&&c==2)?14:"
        "(a==2&&b==3&&c==2)?15:"
        "(a==2&&b==1&&c==3)?16:"
        "(a==2&&b==2&&c==3)?17:"
        "(a==2&&b==3&&c==3)?18:"
        "(a==3&&b==1&&c==1)?19:"
        "(a==3&&b==2&&c==1)?20:"
        "(a==3&&b==3&&c==1)?21:"
        "(a==3&&b==1&&c==2)?22:"
        "(a==3&&b==2&&c==2)?23:"
        "(a==3&&b==3&&c==2)?24:"
        "(a==3&&b==1&&c==3)?25:"
        "(a==3&&b==2&&c==3)?26:"
        "(a==3&&b==3&&c==3)?27:"
        "(a==4&&b==1&&c==1)?28:"
        "(a==4&&b==2&&c==1)?29:"
        "(a==4&&b==3&&c==1)?30:"
        "(a==4&&b==1&&c==2)?31:"
        "(a==4&&b==2&&c==2)?32:"
        "(a==4&&b==3&&c==2)?33:"
        "(a==4&&b==1&&c==3)?34:"
        "(a==4&&b==2&&c==3)?35:"
        "(a==4&&b==3&&c==3)?36:99",
        {"a": steadiness, "b": init_biomass, "c": eme_state},
    )

    final_lpd = (
        ee.Image(0)
        .where(semi.lte(8), 1)
        .where(semi.gt(8).And(semi.lte(14)), 2)
        .where(semi.gt(14).And(semi.lte(21)), 3)
        .where(semi.gt(21).And(semi.lte(30)), 4)
        .where(semi.gt(30).And(semi.lte(36)), 5)
        .rename("LPD")
    )

    win = max(1, int(years_interval))

    bl_start = year_initial
    bl_end = max(year_initial, year_final - win)

    tg_end = year_final
    tg_start = max(year_initial, year_final - win)

    ndvi_1yr = ndvi_dataset

    bl_ndvi_range = ndvi_1yr.select(
        ee.List([f"y{i}" for i in range(bl_start, bl_end + 1)])
    ).reduce(ee.Reducer.percentile([0, 100]))

    bl_ndvi_ext = (
        ndvi_1yr.select(ee.List([f"y{i}" for i in range(bl_start, bl_end + 1)]))
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

    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bl_ndvi_perc = bl_ndvi_ext.reduce(ee.Reducer.percentile(percentiles))

    bl_ndvi_mean = (
        ndvi_1yr.select(ee.List([f"y{i}" for i in range(bl_start, bl_end + 1)]))
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
    )
    tg_ndvi_mean = (
        ndvi_1yr.select(ee.List([f"y{i}" for i in range(tg_start, tg_end + 1)]))
        .reduce(ee.Reducer.mean())
        .rename(["ndvi"])
    )

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

    classes_chg = tg_classes.subtract(bl_classes).where(
        bl_ndvi_mean.subtract(tg_ndvi_mean).abs().lte(100), 0
    )

    classes_chg = classes_chg.rename("Productivity_state_degradation")
    bl_classes = bl_classes.rename(f"Productivity_state_classes_{bl_start}-{bl_end}")
    tg_classes = tg_classes.rename(f"Productivity_state_classes_{tg_start}-{tg_end}")
    bl_ndvi_mean = bl_ndvi_mean.rename(
        f"Productivity_state_NDVI_mean_{bl_start}-{bl_end}"
    )
    tg_ndvi_mean = tg_ndvi_mean.rename(
        f"Productivity_state_NDVI_mean_{tg_start}-{tg_end}"
    )

    signif_band_lpd = (
        ee.Image(-32768)
        .where(final_lpd.eq(1), -2)
        .where(final_lpd.eq(2), -2)
        .where(final_lpd.eq(3), 1)
        .where(final_lpd.eq(4), 1)
        .where(final_lpd.eq(5), 3)
        .rename("Productivity_significance")
    )

    trend_band = (
        ee.Image(-32768)
        .where(final_lpd.eq(1).Or(final_lpd.eq(2)), -1)
        .where(final_lpd.eq(3), 0)
        .where(final_lpd.eq(4).Or(final_lpd.eq(5)), 1)
        .rename("Productivity_trend")
    )

    out_img = (
        trend_band.addBands(signif_band_lpd)
        .addBands(final_lpd)
        .addBands(classes_chg)
        .addBands(bl_classes)
        .addBands(tg_classes)
        .addBands(bl_ndvi_mean)
        .addBands(tg_ndvi_mean)
        .unmask(-32768)
        .int16()
    )
    band_infos = [
        BandInfo(
            "Productivity trajectory (trend)",
            add_to_map=True,
            metadata={"year_initial": year_initial, "year_final": year_end},
        ),
        BandInfo(
            "Productivity trajectory (significance)",
            add_to_map=True,
            metadata={"year_initial": year_initial, "year_final": year_end},
        ),
        BandInfo(
            "Land Productivity Dynamics (from FAO-WOCAT)",
            add_to_map=True,
            metadata={"year_initial": year_initial, "year_final": year_end},
        ),
        BandInfo(
            "Productivity state (degradation)",
            add_to_map=True,
            metadata={
                "year_bl_start": bl_start,
                "year_bl_end": bl_end,
                "year_tg_start": tg_start,
                "year_tg_end": tg_end,
            },
        ),
        BandInfo(
            "Productivity state classes",
            metadata={"year_initial": bl_start, "year_final": bl_end},
        ),
        BandInfo(
            "Productivity state classes",
            metadata={"year_initial": tg_start, "year_final": tg_end},
        ),
        BandInfo(
            "Productivity state NDVI mean",
            metadata={"year_initial": bl_start, "year_final": bl_end},
        ),
        BandInfo(
            "Productivity state NDVI mean",
            metadata={"year_initial": tg_start, "year_final": tg_end},
        ),
    ]

    return TEImage(out_img, band_infos)
