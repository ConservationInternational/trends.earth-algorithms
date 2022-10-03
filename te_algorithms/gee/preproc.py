import ee


def modis_ndvi_annual_integral(year_initial, year_final):
    """Calculate annual trend of integrated NDVI.

    Calculates the trend of annual integrated NDVI using NDVI data from the
    MODIS Collection 6 MOD13Q1 dataset. Areas where changes are not significant
    are masked out using a Mann-Kendall test.

    Args:
        year_initial: The starting year (to define the period the trend is
            calculated over).
        year_final: The ending year (to define the period the trend is
            calculated over).

    Returns:
        Google Earth Engine image collection
    """

    # Load a MODIS NDVI collection 6 MODIS/MOD13Q1
    modis_16d_o = ee.ImageCollection("MODIS/006/MOD13Q1")

    # Function to mask pixels based on quality flags
    def qa_filter(img):
        mask = img.select("SummaryQA")
        mask = mask.where(img.select("SummaryQA").eq(-1), 0)
        mask = mask.where(img.select("SummaryQA").eq(0), 1)
        mask = mask.where(img.select("SummaryQA").eq(1), 1)
        mask = mask.where(img.select("SummaryQA").eq(2), 0)
        mask = mask.where(img.select("SummaryQA").eq(3), 0)
        masked = img.select("NDVI").updateMask(mask)
        return masked

    # Function to integrate observed NDVI datasets at the annual level
    def int_16d_1yr_o(ndvi_coll):
        img_coll = ee.List([])
        for k in range(year_initial, year_final):
            ndvi_img = (
                ndvi_coll.select("NDVI")
                .filterDate("{}-01-01".format(k), "{}-12-31".format(k))
                .reduce(ee.Reducer.mean())
                .multiply(0.0001)
            )
            img = (
                ndvi_img.addBands(ee.Image(k).float())
                .rename(["ndvi", "year"])
                .set({"year": k})
            )
            img_coll = img_coll.add(img)
        return ee.ImageCollection(img_coll)

    # Filter modis collection using the quality filter
    modis_16d_o = modis_16d_o.map(qa_filter)

    # Apply function to compute NDVI annual integrals from 15d observed NDVI data
    return int_16d_1yr_o(modis_16d_o)
