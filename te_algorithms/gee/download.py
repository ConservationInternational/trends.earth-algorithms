import ee
from te_schemas.results import Band
from te_schemas.results import DataType
from te_schemas.schemas import BandInfo

from .util import GEEImage
from .util import TEImage
from .util import TEImageV2
from te_algorithms.gee.util import teimage_v1_to_teimage_v2


def _download_default(
    asset,
    name,
    temporal_resolution,
    year_initial,
    year_final,
):
    """
    Default function used to download data if no other function is provided for an asset
    """
    in_img = ee.Image(asset)

    out = in_img
    band_info = [
        BandInfo(name, add_to_map=True, metadata=in_img.getInfo()["properties"])
    ]
    n_bands = len(in_img.getInfo()["bands"])

    if n_bands > 1:
        band_info.extend(
            [BandInfo(name, add_to_map=False, metadata=in_img.getInfo()["properties"])]
            * (n_bands - 1)
        )

    return teimage_v1_to_teimage_v2(TEImage(out, band_info))


def _download_worldpop(asset, name, temporal_resolution, year_initial, year_final):
    """Download WorldPop data"""
    if year_initial is None:
        year_initial = 2000
    if year_final is None:
        year_final = 2020
    out = TEImageV2(
        {
            DataType.FLOAT32: GEEImage(
                **_get_population(year_initial, asset, add_to_map=True)
            )
        }
    )
    for year in range(year_initial + 1, year_final + 1):
        # Be inclusive of final year (+1) above, and recognize that initial
        # year was already added
        add_to_map = bool(((year - year_initial) % 4) == 0)
        out.add_image(**_get_population(year, asset, add_to_map))

    return out


def _get_population(year, asset, add_to_map=False):
    """Return WorldPop population data for a given year"""
    wp = ee.ImageCollection(asset).filterDate(f"{year}-01-01", f"{int(year) + 1}-01-01")

    wp = (
        wp.select("male")
        .toBands()
        .rename(f"Population_{year}_male")
        .addBands(wp.select("female").toBands().rename(f"Population_{year}_female"))
    )

    return {
        "image": wp,
        "bands": [
            Band(
                "Population (number of people)",
                metadata={
                    "year": year,
                    "type": "male",
                },
                add_to_map=add_to_map,
            ),
            Band(
                "Population (number of people)",
                metadata={
                    "year": year,
                    "type": "female",
                },
                add_to_map=add_to_map,
            ),
        ],
        "datatype": DataType.FLOAT32,
    }


download_functions = {"Gridded Population Count (gender breakdown)": _download_worldpop}


def download(asset, name, temporal_resolution, year_initial, year_final, logger):
    """
    Download dataset from GEE assets.
    """
    logger.debug("Entering download function.")

    # if temporal_resolution != "one time":
    #     assert (year_initial and year_final), "start year or end year not defined"
    #     out = in_img.select('y{}'.format(year_initial))
    #     band_info = [BandInfo(name, add_to_map=True, metadata={'year': year_initial})]
    #     for y in range(year_initial + 1, year_final + 1):
    #         out.addBands(in_img.select('y{}'.format(year_initial)))
    #         band_info.append(BandInfo(name, metadata={'year': year_initial}))
    # else:
    #     out = in_img
    #     band_info = [BandInfo(name, add_to_map=True)]
    if name in download_functions:
        return download_functions[name](
            asset, name, temporal_resolution, year_initial, year_final
        )
    else:
        return _download_default(
            asset, name, temporal_resolution, year_initial, year_final
        )
