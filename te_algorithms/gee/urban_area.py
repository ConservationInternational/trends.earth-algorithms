import ee
from te_schemas.schemas import BandInfo

from .util import TEImage


def urban_area(geojson, un_adju, EXECUTION_ID, logger):
    """
    Calculate urban area.
    """

    logger.debug("Entering urban_area function.")

    aoi = ee.Geometry(geojson)

    # Read asset with the time series of urban extent
    urban_series = ee.Image(
        "users/geflanddegradation/toolbox_datasets/urban_series"
    ).int32()

    # Load population data from GPWv4: Gridded Population of the World Version 4, People/km2 (not UN adjusted)
    if un_adju:
        pop_densi = ee.ImageCollection("CIESIN/GPWv4/population-density")
    else:
        pop_densi = ee.ImageCollection("CIESIN/GPWv4/unwpp-adjusted-population-density")

    # the .multiply(100).int32() is to make the rasters integers, with units for density of people/km2 * 100

    # Select population datasets for each year
    pop2000 = (
        pop_densi.filter(ee.Filter.eq("system:index", "2000"))
        .mean()
        .multiply(100)
        .int32()
    )
    pop2005 = (
        pop_densi.filter(ee.Filter.eq("system:index", "2005"))
        .mean()
        .multiply(100)
        .int32()
    )
    pop2010 = (
        pop_densi.filter(ee.Filter.eq("system:index", "2010"))
        .mean()
        .multiply(100)
        .int32()
    )
    pop2015 = (
        pop_densi.filter(ee.Filter.eq("system:index", "2015"))
        .mean()
        .multiply(100)
        .int32()
    )

    # Use urban extent to mask population density data
    urb_pop2000 = pop2000.updateMask(urban_series.eq(1))
    urb_pop2005 = pop2005.updateMask(urban_series.gte(1).And(urban_series.lte(2)))
    urb_pop2010 = pop2010.updateMask(urban_series.gte(1).And(urban_series.lte(3)))
    urb_pop2015 = pop2015.updateMask(urban_series.gte(1).And(urban_series.lte(4)))

    # Compute mean population density per year
    urb_pop2000m = urb_pop2000.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e12
    )
    urb_pop2005m = urb_pop2005.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e12
    )
    urb_pop2010m = urb_pop2010.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e12
    )
    urb_pop2015m = urb_pop2015.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e12
    )

    # Compute urban area per year
    pixel_area = urban_series.updateMask(urban_series.eq(1)).multiply(
        ee.Image.pixelArea()
    )

    urb_are2000 = (
        urban_series.updateMask(urban_series.eq(1))
        .multiply(ee.Image.pixelArea())
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e12)
    )
    urb_are2005 = (
        urban_series.updateMask(urban_series.gte(1).And(urban_series.lte(2)))
        .multiply(ee.Image.pixelArea())
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e12)
    )
    urb_are2010 = (
        urban_series.updateMask(urban_series.gte(1).And(urban_series.lte(3)))
        .multiply(ee.Image.pixelArea())
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e12)
    )
    urb_are2015 = (
        urban_series.updateMask(urban_series.gte(1).And(urban_series.lte(4)))
        .multiply(ee.Image.pixelArea())
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e12)
    )

    # Make a dictionary to contain results
    result_table = ee.Dictionary(
        {
            "pop_dens2000": urb_pop2000m.get("population-density"),
            "pop_dens2005": urb_pop2005m.get("population-density"),
            "pop_dens2010": urb_pop2010m.get("population-density"),
            "pop_dens2015": urb_pop2015m.get("population-density"),
            "urb_area2000": urb_are2000.get("classification"),
            "urb_area2005": urb_are2005.get("classification"),
            "urb_area2010": urb_are2010.get("classification"),
            "urb_area2015": urb_are2015.get("classification"),
        }
    )

    # # Export the FeatureCollection.
    # Export.table.toDrive({
    #   collection: ee.FeatureCollection([ee.Feature(null,result_table)]),
    #   description: "export_urban_extent_table",
    #   folder: 'sdg1131',
    #   fileFormat: 'CSV'})
    #
    # Export raster
    result_raster = (
        urban_series.addBands(urb_pop2000)
        .addBands(urb_pop2005)
        .addBands(urb_pop2010)
        .addBands(urb_pop2015)
    )

    logger.debug("Setting up output.")
    out = TEImage(
        result_raster.clip(aoi),
        [
            BandInfo(
                "Urban series",
                add_to_map=True,
                metadata={"years": [2000, 2005, 2010, 2015]},
            ),
            BandInfo("Population", metadata={"year": 2000}),
            BandInfo("Population", metadata={"year": 2005}),
            BandInfo("Population", metadata={"year": 2010}),
            BandInfo("Population", add_to_map=True, metadata={"year": 2015}),
        ],
    )

    # out.image = out.image.unmask(-32768).int16()

    return out
