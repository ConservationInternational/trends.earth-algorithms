from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ee

from landdegradation.util import TEImage
from landdegradation.schemas import BandInfo

def download(asset, name, temporal_resolution, start_year=None, end_year=None, 
             EXECUTION_ID, logger):
    """
    Download dataset from GEE assets.
    """
    logger.debug("Entering download function.")

    in_img = ee.Image(asset)

    if temporal_resolution != "one time":
        assert start_year, end_year, "start year or end year not defined"
        out = in_img.select('y{}'.format(start_year))
        band_info = [BandInfo(name, metadata={'year': start_year})
        for y in range(start_year + 1, end_year + 1)
            out.addBands(in_img.select('y{}'.format(start_year)))
            band_info.append(BandInfo(name, metadata={'year': start_year}))
    else:
        out = in_img
        band_info = [BandInfo(name)]

    return TEImage(out, band_info)
