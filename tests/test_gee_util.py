import json
import os
from pathlib import Path
from typing import List

import marshmallow_dataclass
import pytest
from te_schemas.results import RasterResults

from te_algorithms.gee.util import GEEImage, TEImageV2


@marshmallow_dataclass.dataclass
class ee_image_mock:
    bands: List[int]

    def select(self, bands: List[int]):
        self.bands = [item for item in self.bands if item in bands]
        return self

    def getInfo(self):
        return {"bands": [{"id": band} for band in self.bands]}

    def addBands(self, image):
        "Add new bands to the image"
        last_band = max(self.bands)
        self.bands.extend([*range(last_band + 1, last_band + 1 + len(image.bands))])
        return self


def _get_json(file):
    test_dir = Path(os.path.abspath(__file__)).parent
    with open(test_dir / "data" / file) as f:
        return json.load(f)


def _get_TEImageV2(file):
    rr = RasterResults.Schema().load(_get_json(file))
    return TEImageV2(
        images={
            datatype: GEEImage(
                ee_image_mock([*range(len(raster.bands))]),
                bands=raster.bands,
                datatype=raster.datatype,
            )
            for datatype, raster in rr.rasters.items()
        }
    )


def test_teimagev2_rmDuplicates():
    te_image = _get_TEImageV2("RasterResults_sdg-15-3-1-sub-indicators_1.json")
    te_image.merge(_get_TEImageV2("RasterResults_sdg-15-3-1-sub-indicators_2.json"))
    te_image.merge(_get_TEImageV2("RasterResults_sdg-15-3-1-sub-indicators_3.json"))
    assert all(
        len(item[1].bands) == n_bands
        for item, n_bands in zip(te_image.images.items(), [6, 57])
    )
    te_image.rmDuplicates()
    assert all(
        len(item[1].bands) == n_bands
        for item, n_bands in zip(te_image.images.items(), [4, 53])
    )
