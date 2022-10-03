from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from .util import TEImage
from te_schemas.schemas import BandInfo


def download(asset, name, temporal_resolution, year_initial, year_final, logger):
    """
    Download dataset from GEE assets.
    """
    logger.debug("Entering download function.")

    in_img = ee.Image(asset)

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

    return TEImage(out, band_info)
