import random
import re
import threading
import typing
from time import time
from typing import Union

import backoff
import ee
import requests
from te_schemas import results
from te_schemas.results import Raster, TiledRaster
from te_schemas.schemas import (
    BandInfoSchema,
    CloudResults,
    CloudResultsSchema,
)
from te_schemas.schemas import Url as UrlDeprecated

from . import GEEImageError, GEETaskFailure

_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]+")


def _generate_sanitized_band_names(bands):
    sanitized_names = []
    used_names = set()

    for idx, band in enumerate(bands, start=1):
        raw_name = band.name or f"band_{idx}"
        sanitized = _SANITIZE_PATTERN.sub("_", raw_name).strip("_")

        if not sanitized:
            sanitized = f"band_{idx}"

        if sanitized[0].isdigit():
            sanitized = f"b_{sanitized}"

        base_name = sanitized
        suffix = 2
        while sanitized in used_names:
            sanitized = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(sanitized)
        sanitized_names.append(sanitized)

        band.metadata["gee_band_name"] = sanitized

    return sanitized_names


# Google cloud storage bucket for output
BUCKET = "ldmt"

# Number of minutes a GEE task is allowed to run before timing out and being
# cancelled
TASK_TIMEOUT_MINUTES = 48 * 60


def get_region(geom):
    """Return ee.Geometry from supplied GeoJSON object."""
    poly = get_coords(geom)
    ptype = get_type(geom)

    if ptype.lower() == "multipolygon":
        region = ee.Geometry.MultiPolygon(poly)
    else:
        region = ee.Geometry.Polygon(poly)

    return region


def get_coords(geojson):
    """."""

    if geojson.get("features") is not None:
        return geojson.get("features")[0].get("geometry").get("coordinates")
    elif geojson.get("geometry") is not None:
        return geojson.get("geometry").get("coordinates")
    else:
        return geojson.get("coordinates")


def get_type(geojson):
    """."""

    if geojson.get("features") is not None:
        return geojson.get("features")[0].get("geometry").get("type")
    elif geojson.get("geometry") is not None:
        return geojson.get("geometry").get("type")
    else:
        return geojson.get("type")


class gee_task(threading.Thread):
    """Run earth engine task against the trends.earth API"""

    def __init__(self, task, prefix, logger, metadata=None):
        threading.Thread.__init__(self)
        self.task = task
        self.prefix = prefix
        self.logger = logger
        # self.metadata is used only to facilitate saving the final JSON output
        # for Trends.Earth
        self.metadata = metadata
        self.state = self.task.status().get("state")
        self.start()

    def cancel_hdlr(self, details):
        self.logger.debug(
            "GEE task {} timed out after {} hours".format(
                self.task.status().get("id"), (time() - self.start_time) / (60 * 60)
            )
        )
        ee.data.cancelTask(self.task.status().get("id"))

    def on_backoff_hdlr(self, details):
        try:
            details.update({"task_id": self.task.status().get("id")})
            self.logger.debug(
                "Backing off {wait:0.1f} seconds after {tries} tries "
                "calling function {target} for task {task_id}".format(**details)
            )
        except Exception:
            # Never let a logging failure interfere with the polling loop
            pass

    def poll_for_completion(self):
        @backoff.on_predicate(
            backoff.expo,
            lambda x: x in ["READY", "RUNNING"],
            on_backoff=self.on_backoff_hdlr,
            on_giveup=self.cancel_hdlr,
            max_time=TASK_TIMEOUT_MINUTES * 60,
            factor=3,
            base=1.4,
            max_value=600,
        )
        def get_status(self):
            try:
                self.logger.send_progress(self.task.status().get("progress", 0.0))
            except Exception:
                # send_progress failures must not kill the polling loop.
                # The Environment's GEFLogger.send_progress already guards
                # against this, but older versions or subclasses may not.
                pass
            self.state = self.task.status().get("state")

            return self.state

        return get_status(self)

    def run(self):
        self.task.start()
        self.start_time = time()
        self.logger.debug("Starting GEE task {}.".format(self.task.status().get("id")))
        self.poll_for_completion()

        if not self.state:
            raise GEETaskFailure(self.task)

        if self.state == "COMPLETED":
            self.logger.debug(
                "GEE task {} completed.".format(self.task.status().get("id"))
            )
        elif self.state == "FAILED":
            self.logger.debug(
                "GEE task {} failed: {}".format(
                    self.task.status().get("id"),
                    self.task.status().get("error_message"),
                )
            )
            raise GEETaskFailure(self.task)
        else:
            self.logger.debug(
                "GEE task {} returned status {}: {}".format(
                    self.task.status().get("id"),
                    self.state,
                    self.task.status().get("error_message"),
                )
            )
            raise GEETaskFailure(self.task)

    def get_urls(self):
        @backoff.on_exception(
            backoff.expo,
            requests.exceptions.RequestException,
            on_backoff=self.on_backoff_hdlr,
            max_time=60,
            factor=3,
            base=1.4,
            max_value=600,
        )
        def request_urls(self):
            return requests.get(
                f"https://www.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={self.prefix}"
            )

        resp = request_urls(self)

        if not resp or resp.status_code != 200:
            self.logger.debug(f"Failed to list urls for results from {self.task}")
            raise GEETaskFailure(self.task)

        items = resp.json()["items"]

        if len(items) < 1:
            self.logger.debug("No urls were found for {}".format(self.task))
            raise GEETaskFailure(self.task)
        else:
            urls = []

            for item in items:
                urls.append(UrlDeprecated(item["mediaLink"], item["md5Hash"]))

            return urls

    def get_uris(self):
        @backoff.on_exception(
            backoff.expo,
            requests.exceptions.RequestException,
            on_backoff=self.on_backoff_hdlr,
            max_time=60,
            factor=3,
            base=1.4,
            max_value=600,
        )
        def request_uris(self):
            return requests.get(
                f"https://www.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={self.prefix}"
            )

        resp = request_uris(self)

        if not resp or not resp.json().get("items"):
            self.logger.debug(f"Failed to list uris for results from {self.task}")
            raise GEETaskFailure(self.task)

        items = resp.json()["items"]

        if len(items) < 1:
            self.logger.debug("No uris were found for {}".format(self.task))
            raise GEETaskFailure(self.task)
        else:
            uris = []

            for item in items:
                uris.append(
                    results.URI(
                        uri=item["mediaLink"],
                        etag=results.Etag(
                            hash=item["md5Hash"], type=results.EtagType.GCS_CRC32C
                        ),
                    )
                )

            return uris


# Not using dataclass as not in python 3.6
class TEImage:
    "A class to store GEE images and band info for export to cloud storage"

    def __init__(self, image, band_info):
        self.image = image
        self.band_info = band_info

        self._check_validity()

    def _check_validity(self):
        expected_count = len(self.band_info)
        if expected_count == 0:
            return

        sanitized_names = _generate_sanitized_band_names(self.band_info)

        try:
            self.image = self.image.rename(sanitized_names)
        except ee.ee_exception.EEException as exc:
            raise GEEImageError("Band metadata does not match image bands") from exc

    def merge(self, other):
        "Merge with another TEImage by adding data from other TEImage as new bands"
        self.image = self.image.addBands(other.image)
        self.band_info.extend(other.band_info)

        self._check_validity()

    def addBands(self, bands, band_info):
        "Add new bands to the image"
        self.image = self.image.addBands(bands)
        self.band_info.extend(band_info)

        self._check_validity()

    def selectBands(self, band_names):
        "Select certain bands from the image, dropping all others"
        band_indices = [
            i for i, bi in enumerate(self.band_info) if bi.name in band_names
        ]

        if len(band_indices) < 1:
            raise GEEImageError('Bands "{}" not in image'.format(band_names))

        self.band_info = [self.band_info[i] for i in band_indices]
        self.image = self.image.select(band_indices)

        self._check_validity()

    def getImages(
        self,
        name_filter: Union[str, list],
        field: Union[None, str] = None,
        field_filter: Union[None, str] = None,
    ):
        "Select certain bands from the image(s), dropping all others"
        if isinstance(name_filter, str):
            # make name_filter a length 1 list if it is a string
            name_filter = [name_filter]

        if field:
            assert field_filter is not None
            band_indices = [
                i
                for i, bi in enumerate(self.band_info)
                if (bi.name in name_filter and bi.metadata[field] == field_filter)
            ]
        else:
            band_indices = [
                i for i, bi in enumerate(self.band_info) if bi.name in name_filter
            ]

        if band_indices:
            return self.image.select(band_indices)
        else:
            return None

    def setAddToMap(self, band_names=[]):
        "Set the layers that will be added to the map by default"

        for i in range(len(self.band_info)):
            if self.band_info[i].name in band_names:
                self.band_info[i].add_to_map = True
            else:
                self.band_info[i].add_to_map = False

    def export(
        self,
        geojsons,
        task_name,
        crs,
        logger,
        execution_id=None,
        proj=None,
        maxpixels=1e13,
    ):
        "Export layers to cloud storage"

        if not execution_id:
            execution_id = str(random.randint(1000000, 99999999))
        else:
            execution_id = execution_id

        if not proj:
            proj = self.image.projection()

        tasks = []
        n = 1

        for geojson in geojsons:
            if task_name:
                out_name = "{}_{}_{}".format(execution_id, task_name, n)
            else:
                out_name = "{}_{}".format(execution_id, n)

            export = {
                "image": self.image,
                "description": out_name,
                "fileNamePrefix": out_name,
                "bucket": BUCKET,
                "maxPixels": maxpixels,
                "crs": crs,
                "scale": ee.Number(proj.nominalScale()).getInfo(),
                "region": get_coords(geojson),
                "formatOptions": {"cloudOptimized": True},
            }
            t = gee_task(
                task=ee.batch.Export.image.toCloudStorage(**export),
                prefix=out_name,
                logger=logger,
            )
            tasks.append(t)
            n += 1

        logger.debug("Exporting to cloud storage.")
        urls = []

        for task in tasks:
            task.join()
            urls.extend(task.get_urls())

        gee_results = CloudResults(task_name, self.band_info, urls)
        results_schema = CloudResultsSchema()
        json_results = results_schema.dump(gee_results)

        return json_results


class GEEImage:
    def __init__(
        self,
        image: ee.Image,
        bands: typing.List[results.Band],
        datatype: results.DataType = results.DataType.INT16,
    ):
        self.image = image
        self.bands = bands
        self.datatype = datatype

        self._check_validity()

    def _check_validity(self):
        expected_count = len(self.bands)
        if expected_count == 0:
            return

        sanitized_names = _generate_sanitized_band_names(self.bands)

        try:
            self.image = self.image.rename(sanitized_names)
        except ee.ee_exception.EEException as exc:
            raise GEEImageError("Band metadata does not match image bands") from exc

    def merge(self, other):
        "Merge with another GEEImage object"

        if self.datatype != other.datatype:
            raise GEEImageError(
                f"Attempted to merge {self.datatype} image with "
                f"{other.datatype} image. Both images must have same "
                "datatype."
            )
        self.image = self.image.addBands(other.image)
        self.bands.extend(other.bands)

        self._check_validity()

    def addBands(self, image, bands):
        "Add new bands to the image"
        self.image = self.image.addBands(image)
        self.bands.extend(bands)

        self._check_validity()

    def match_bands(self, bands, reverse=False):
        """
        Returns the indices of bands that match input.

        Matches on BandInfo name and metadata attributes only.
        """

        # Find indices in self.bands that match the specified input bands,
        # matching only on the band names and metadata
        matches = [
            i
            for i, self_band in enumerate(self.bands)
            if any(
                [
                    all(
                        [
                            band.name == self_band.name,
                            band.metadata == self_band.metadata,
                        ]
                    )
                    for band in bands
                ]
            )
        ]
        if reverse:
            # returns only band indices for bands that do NOT match
            return [i for i in range(len(self.bands)) if i not in matches]
        else:
            return matches

    def rmDuplicates(self):
        """
        Removes any bands that are duplicates

        Matches on BandInfo name and metadata attributes only.
        """

        duplicates = []
        for outer_index in range(len(self.bands) - 1):
            if outer_index in duplicates:
                continue
            outer_band = self.bands[outer_index]
            for inner_index in range(outer_index + 1, len(self.bands)):
                if inner_index in duplicates:
                    continue
                inner_band = self.bands[inner_index]

                if all(
                    [
                        outer_band.name == inner_band.name,
                        outer_band.metadata == inner_band.metadata,
                    ]
                ):
                    duplicates.append(inner_index)
        if len(duplicates) > 0:
            self.rmBands(duplicates)

    def rmBands(self, indices):
        """
        Remove bands from the image, based on their indices.
        """

        assert min(indices) >= 0 and max(indices) <= len(self.bands)

        non_match_indices = [i for i in range(len(self.bands)) if i not in indices]

        # Remove these bands, iterating in reverse order to avoid index shifting issues
        for index in sorted(indices, reverse=True):
            del self.bands[index]

        self.image = self.image.select(non_match_indices)

        self._check_validity()

    def cast(self):
        if self.datatype == results.DataType.BYTE:
            self.image = self.image.byte()
        elif self.datatype == results.DataType.UINT16:
            self.image = self.image.uint16()
        elif self.datatype == results.DataType.INT16:
            self.image = self.image.int16()
        elif self.datatype == results.DataType.UINT32:
            self.image = self.image.uint32()
        elif self.datatype == results.DataType.INT32:
            self.image = self.image.int32()
        elif self.datatype == results.DataType.FLOAT32:
            self.image = self.image.float()
        elif self.datatype == results.DataType.FLOAT64:
            self.image = self.image.double()
        else:
            raise GEEImageError(
                f"Unknown datatype {self.datatype}. Datatype "
                "must be supported by GDAL GeoTiff driver."
            )


def teimage_v1_to_teimage_v2(te_image):
    """Upgrade a version 1 TEImage to TEImageV2"""
    datatype = results.DataType.INT16

    bands = []

    for n, band in enumerate(te_image.band_info):
        # Dump and load each band in order to ensure defaults are added
        band = BandInfoSchema().load(BandInfoSchema().dump(band))
        bands.append(results.Band(**BandInfoSchema().dump(band)))

    image = GEEImage(te_image.image, bands=bands, datatype=datatype)

    return TEImageV2({datatype: image})


# Not using dataclass as not in python 3.6
class TEImageV2:
    "A class to store GEE images and band info for export to cloud storage"

    def __init__(self, images: typing.Dict[str, GEEImage] = {}):
        self.images = images

    def add_image(
        self,
        image: ee.Image,
        bands: typing.List[results.Band],
        datatype: results.DataType = results.DataType.INT16,
    ):
        gee_image = GEEImage(image, bands, datatype)

        if datatype not in self.images:
            self.images[datatype] = gee_image
        else:
            self.images[datatype].merge(gee_image)

    def merge(self, other):
        "Merge with another TEImageV2 object"

        for datatype, other_image in other.images.items():
            if datatype in self.images:
                self.images[datatype].merge(other_image)
            else:
                self.images[datatype] = other_image

    def rmDuplicates(self):
        for _, image in self.images.items():
            image.rmDuplicates()

    def getImages(
        self,
        name_filter: Union[str, list],
        field: Union[None, str] = None,
        field_filter: Union[None, str] = None,
    ):
        "Select certain bands from the image(s), dropping all others"
        if isinstance(name_filter, str):
            # make name_filter a length 1 list if it is a string
            name_filter = [name_filter]

        # Validate parameters before processing - this ensures the assertion
        # is triggered even if self.images is empty (e.g., in CI environments)
        if field:
            assert field_filter is not None

        out = []
        for _, image in self.images.items():
            if field:
                band_indices = [
                    i
                    for i, bi in enumerate(image.bands)
                    if (bi.name in name_filter and bi.metadata[field] == field_filter)
                ]
            else:
                band_indices = [
                    i for i, bi in enumerate(image.bands) if bi.name in name_filter
                ]

            if band_indices:
                out.append(image.image.select(band_indices))
        return out

    def selectBands(self, band_names):
        "Select certain bands from the image(s), dropping all others"

        for _, image in self.images.items():
            band_indices = [
                i for i, bi in enumerate(image.bands) if bi.name in band_names
            ]

            if len(band_indices) < 1:
                raise GEEImageError(f'Band(s) "{band_names}" not in image')

            image.bands = [image.bands[i] for i in band_indices]
            image.image = image.image.select(band_indices)

    def setAddToMap(self, band_names=[]):
        "Set the layers that will be added to the map by default"

        for _, image in self.images.items():
            for i in range(len(image.bands)):
                if image.bands[i].name in band_names:
                    image.bands[i].add_to_map = True
                else:
                    image.bands[i].add_to_map = False

    def export(
        self,
        geojsons,
        task_name,
        crs,
        logger,
        execution_id=None,
        proj=None,
        filetype=results.RasterFileType.COG,
        maxpixels=1e13,
    ):
        "Export layers to cloud storage"

        if not execution_id:
            execution_id = str(random.randint(1000000, 99999999))
        else:
            execution_id = execution_id

        tasks = []

        for datatype, image in self.images.items():
            if not proj:
                proj = image.image.projection()

            image.cast()

            n = 1

            for geojson in geojsons:
                if task_name:
                    out_name = "{}_{}_{}_{}".format(
                        execution_id, task_name, datatype.value, n
                    )
                else:
                    out_name = "{}_{}_{}".format(execution_id, datatype.value, n)

                if filetype == results.RasterFileType.COG:
                    as_COG = True
                else:
                    as_COG = False

                export = {
                    "image": image.image,
                    "description": out_name,
                    "fileNamePrefix": out_name,
                    "bucket": BUCKET,
                    "maxPixels": maxpixels,
                    "crs": crs,
                    "scale": ee.Number(proj.nominalScale()).getInfo(),
                    "region": get_coords(geojson),
                    "formatOptions": {"cloudOptimized": as_COG},
                }
                t = gee_task(
                    task=ee.batch.Export.image.toCloudStorage(**export),
                    prefix=out_name,
                    logger=logger,
                    metadata={"datatype": datatype, "bands": image.bands},
                )
                tasks.append(t)
                n += 1

        logger.debug("Exporting to cloud storage.")

        output = {}

        for task in tasks:
            task.join()

            if task.metadata["datatype"] in output:
                output[task.metadata["datatype"]]["uris"].extend(task.get_uris())
            else:
                output[task.metadata["datatype"]] = {
                    "uris": task.get_uris(),
                    "bands": task.metadata["bands"],
                }

        rasters = {}

        for datatype, value in output.items():
            uris = [results.URI.Schema().dump(uri) for uri in value["uris"]]
            bands = [results.Band.Schema().dump(band) for band in value["bands"]]

            if len(uris) > 1:
                rasters[datatype.value] = TiledRaster.Schema().load(
                    {
                        "tile_uris": uris,
                        "bands": bands,
                        "datatype": datatype,
                        "filetype": filetype,
                    }
                )
            else:
                rasters[datatype.value] = Raster.Schema().load(
                    {
                        "uri": uris[0],
                        "bands": bands,
                        "datatype": datatype,
                        "filetype": filetype,
                    }
                )

        gee_results = results.RasterResults(name=task_name, rasters=rasters, data={})

        return results.RasterResults.Schema().dump(gee_results)
