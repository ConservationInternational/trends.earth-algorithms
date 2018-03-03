import json
import ee
import threading
import random

from time import sleep

from landdegradation import GEETaskFailure, GEEImageError

# Below is for usage in testing scripts
sen_geojson = '{"type":"Polygon","coordinates":[[[-16.713729,13.594959],[-17.126107,14.373516],[-17.625043,14.729541],[-17.185173,14.919477],[-16.700706,15.621527],[-16.463098,16.135036],[-16.12069,16.455663],[-15.623666,16.369337],[-15.135737,16.587282],[-14.577348,16.598264],[-14.099521,16.304302],[-13.435738,16.039383],[-12.830658,15.303692],[-12.17075,14.616834],[-12.124887,13.994727],[-11.927716,13.422075],[-11.553398,13.141214],[-11.467899,12.754519],[-11.513943,12.442988],[-11.658301,12.386583],[-12.203565,12.465648],[-12.278599,12.35444],[-12.499051,12.33209],[-13.217818,12.575874],[-13.700476,12.586183],[-15.548477,12.62817],[-15.816574,12.515567],[-16.147717,12.547762],[-16.677452,12.384852],[-16.841525,13.151394],[-15.931296,13.130284],[-15.691001,13.270353],[-15.511813,13.27857],[-15.141163,13.509512],[-14.712197,13.298207],[-14.277702,13.280585],[-13.844963,13.505042],[-14.046992,13.794068],[-14.376714,13.62568],[-14.687031,13.630357],[-15.081735,13.876492],[-15.39877,13.860369],[-15.624596,13.623587],[-16.713729,13.594959]]]}'

tza_geojson = '{"type":"Polygon","coordinates":[[[33.903711,-0.95],[34.07262,-1.05982],[37.69869,-3.09699],[37.7669,-3.67712],[39.20222,-4.67677],[38.74054,-5.90895],[38.79977,-6.47566],[39.44,-6.84],[39.47,-7.1],[39.19469,-7.7039],[39.25203,-8.00781],[39.18652,-8.48551],[39.53574,-9.11237],[39.9496,-10.0984],[40.31659,-10.3171],[39.521,-10.89688],[38.427557,-11.285202],[37.82764,-11.26879],[37.47129,-11.56876],[36.775151,-11.594537],[36.514082,-11.720938],[35.312398,-11.439146],[34.559989,-11.52002],[34.28,-10.16],[33.940838,-9.693674],[33.73972,-9.41715],[32.759375,-9.230599],[32.191865,-8.930359],[31.556348,-8.762049],[31.157751,-8.594579],[30.74,-8.34],[30.2,-7.08],[29.62,-6.52],[29.419993,-5.939999],[29.519987,-5.419979],[29.339998,-4.499983],[29.753512,-4.452389],[30.11632,-4.09012],[30.50554,-3.56858],[30.75224,-3.35931],[30.74301,-3.03431],[30.52766,-2.80762],[30.46967,-2.41383],[30.758309,-2.28725],[30.816135,-1.698914],[30.419105,-1.134659],[30.76986,-1.01455],[31.86617,-1.02736],[33.903711,-0.95]]]}'

# Google cloud storage bucket for output
BUCKET = "ldmt"


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

    def __init__(self, task, name, out_name, logger):
        threading.Thread.__init__(self)
        self.task = task
        self.name = name
        self.out_name = out_name
        self.logger = logger
        self.state = self.task.status().get('state')
        self.start()

    def run(self):
        self.task_id = self.task.status().get('id')
        self.logger.debug("Starting GEE task {}.".format(self.task_id))
        self.task.start()
        self.state = self.task.status().get('state')
        while self.state == 'READY' or self.state == 'RUNNING':
            task_progress = self.task.status().get('progress', 0.0)
            self.logger.send_progress(task_progress)
            self.logger.debug("GEE task {} progress {}.".format(self.task_id, task_progress))
            self.state = self.task.status().get('state')
            sleep(60)
        if self.state == 'COMPLETED':
            self.logger.debug("GEE task {} completed.".format(self.task_id))
        if self.state == 'FAILED':
            self.logger.debug("GEE task {} failed: {}".format(self.task_id, self.task.status().get('error_message')))
            raise GEETaskFailure(self.task)

    def status(self):
        self.state = self.task.status().get('state')
        return self.state

    def get_URL_base(self):
        return "http://{}.storage.googleapis.com".format(BUCKET)

    def get_files(self):
        return "{}.tif".format(self.out_name)


class TEImage(object):
    def __init__(self, image, band_info):
        self.image = image
        self.band_info = band_info

        if len(self.band_info) != len(self.image.getInfo()['bands']):
            raise GEEImageError('Band info length does not match number of bands in image')

    def addBands(self, bands, band_info):
        self.image.addBands(bands)
        self.band_info.extend(band_info)

        if len(self.band_info) != len(self.image.getInfo()['bands']):
            raise GEEImageError('Band info length does not match number of bands in image')

    def selectBands(self, band_names):
        new_band_info = [i for i in self.band_info if i in band_names]
        if len(new_band_info) < 1:
            return None

        self.band_info = new_band_info
        self.image = self.image.select(band_names)

        if len(self.band_info) != len(self.image.getInfo()['bands']):
            raise GEEImageError('Band info length does not match number of bands in image')

    def export(self, proj, geojson, task_name, logger, execution_id=None):
        proj = proj
        geojson = geojson
        task_name = task_name
        logger = logger
        if not execution_id:
            execution_id = str(random.randint(1000000, 99999999))
        else:
            execution_id = execution_id

        if task_name:
            out_name = '{}_{}'.format(execution_id, task_name)
        else:
            out_name = '{}'.format(execution_id)
        export = {'image': self.image,
                  'description': out_name,
                  'fileNamePrefix': out_name,
                  'bucket': BUCKET,
                  'maxPixels': 1e13,
                  'scale': ee.Number(proj.nominalScale()).getInfo(),
                  'region': get_coords(geojson)}

        logger.debug("Exporting to cloud storage.")
        self.task = gee_task(ee.batch.Export.image.toCloudStorage(**export),
                             task_name,
                             out_name, logger)
        self.task.join()

        gee_results = CloudResults(task_name,
                                   URLList(self.task.get_URL_base(),
                                           self.task.get_files()))
        results_schema = CloudResultsSchema()
        json_results = results_schema.dump(gee_results)

        return json_results.data
