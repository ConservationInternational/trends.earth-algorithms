import ee
import threading
import random
import requests

from time import time, sleep

from . import GEETaskFailure, GEEImageError
from te_schemas.schemas import CloudResults, CloudResultsSchema, Url


# Google cloud storage bucket for output
BUCKET = "ldmt"

# Number of minutes a GEE task is allowed to run before timing out and being 
# cancelled
TASK_TIMEOUT_MINUTES = 48 * 60


def get_region(geom):
    """Return ee.Geometry from supplied GeoJSON object."""
    poly = get_coords(geom)
    ptype = get_type(geom)
    if ptype.lower() == 'multipolygon':
        region = ee.Geometry.MultiPolygon(poly)
    else:
        region = ee.Geometry.Polygon(poly)
    return region


def get_coords(geojson):
    """."""
    if geojson.get('features') is not None:
        return geojson.get('features')[0].get('geometry').get('coordinates')
    elif geojson.get('geometry') is not None:
        return geojson.get('geometry').get('coordinates')
    else:
        return geojson.get('coordinates')


def get_type(geojson):
    """."""
    if geojson.get('features') is not None:
        return geojson.get('features')[0].get('geometry').get('type')
    elif geojson.get('geometry') is not None:
        return geojson.get('geometry').get('type')
    else:
        return geojson.get('type')


class gee_task(threading.Thread):
    """Run earth engine task against the trends.earth API"""

    def __init__(self, task, prefix, logger):
        threading.Thread.__init__(self)
        self.task = task
        self.prefix = prefix
        self.logger = logger
        self.state = self.task.status().get('state')
        self.start()

    def run(self):
        self.task.start()
        self.logger.debug("Starting GEE task {}.".format(self.task.status().get('id')))
        self.state = self.task.status().get('state')
        self.start_time = time()
        while self.state == 'READY' or self.state == 'RUNNING':
            task_progress = self.task.status().get('progress', 0.0)
            self.logger.send_progress(task_progress)
            self.logger.debug("GEE task {} progress {}.".format(self.task.status().get('id'), task_progress))
            sleep(60)
            self.state = self.task.status().get('state')
            if (time() - self.start_time) / 60 > TASK_TIMEOUT_MINUTES:
                self.logger.debug("GEE task {} timed out after {} hours".format(self.task.status().get('id'), (time() - self.start_time) / (60*60)))
                ee.data.cancelTask(self.task.status().get('id'))
                raise GEETaskFailure(self.task)
        if self.state == 'COMPLETED':
            self.logger.debug("GEE task {} completed.".format(self.task.status().get('id')))
        elif self.state == 'FAILED':
            self.logger.debug("GEE task {} failed: {}".format(self.task.status().get('id'), self.task.status().get('error_message')))
            raise GEETaskFailure(self.task)
        else:
            self.logger.debug("GEE task {} returned status {}: {}".format(self.task.status().get('id'), self.state, self.task.status().get('error_message')))
            raise GEETaskFailure(self.task)

    def status(self):
        self.state = self.task.status().get('state')
        return self.state

    def get_urls(self):
        resp = requests.get(
            f'https://www.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={self.prefix}'
        )
        if not resp or resp.status_code != 200:
            raise GEETaskFailure(
                f'Failed to list urls for results from {self.task}'
            )

        items = resp.json()['items']

        if len(items) < 1:
            raise GEETaskFailure('No urls were found for {}'.format(self.task))
        else:
            urls = []
            for item in items:
                urls.append(Url(item['mediaLink'], item['md5Hash']))
            return urls


class TEImage(object):
    "A class to store GEE images and band info for export to cloud storage"
    def __init__(self, image, band_info):
        self.image = image
        self.band_info = band_info

        self._check_validity()
    
    def _check_validity(self):
        if len(self.band_info) != len(self.image.getInfo()['bands']):
            raise GEEImageError(
                f'Band info length ({len(self.band_info)}) does not match '
                f'number of bands in image ({self.image.getInfo()["bands"]})'
            )

    def merge(self, other):
        "Merge with another TEImage object"
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
        band_indices = [i for i, bi in enumerate(self.band_info) if bi.name in band_names]
        if len(band_indices) < 1:
            raise GEEImageError('Bands "{}" not in image'.format(band_names))

        self.band_info = [self.band_info[i] for i in band_indices]
        self.image = self.image.select(band_indices)

        self._check_validity()

    def setAddToMap(self, band_names=[]):
        "Set the layers that will be added to the user's map in QGIS by default"
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
        proj=None
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
                out_name = '{}_{}_{}'.format(execution_id, task_name, n)
            else:
                out_name = '{}_{}'.format(execution_id, n)

            export = {'image': self.image,
                      'description': out_name,
                      'fileNamePrefix': out_name,
                      'bucket': BUCKET,
                      'maxPixels': 1e13,
                      'crs': crs,
                      'scale': ee.Number(proj.nominalScale()).getInfo(),
                      'region': get_coords(geojson)}
            t = gee_task(
                task=ee.batch.Export.image.toCloudStorage(**export),
                prefix=out_name,
                logger=logger
            )
            tasks.append(t)
            n += 1
            
        logger.debug("Exporting to cloud storage.")
        urls = []
        for task in tasks:
            task.join()
            urls.extend(task.get_urls())

        gee_results = CloudResults(
            task_name,
            self.band_info,
            urls
        )
        results_schema = CloudResultsSchema()
        json_results = results_schema.dump(gee_results)

        return json_results
