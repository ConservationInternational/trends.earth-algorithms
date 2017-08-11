import json
import ee

from time import sleep

# Below is for usage in testing scripts
sen_geojson = '{"type":"FeatureCollection","features":[{"type":"Feature","id":"SEN","properties":{"name":"Senegal"},"geometry":{"type":"Polygon","coordinates":[[[-16.713729,13.594959],[-17.126107,14.373516],[-17.625043,14.729541],[-17.185173,14.919477],[-16.700706,15.621527],[-16.463098,16.135036],[-16.12069,16.455663],[-15.623666,16.369337],[-15.135737,16.587282],[-14.577348,16.598264],[-14.099521,16.304302],[-13.435738,16.039383],[-12.830658,15.303692],[-12.17075,14.616834],[-12.124887,13.994727],[-11.927716,13.422075],[-11.553398,13.141214],[-11.467899,12.754519],[-11.513943,12.442988],[-11.658301,12.386583],[-12.203565,12.465648],[-12.278599,12.35444],[-12.499051,12.33209],[-13.217818,12.575874],[-13.700476,12.586183],[-15.548477,12.62817],[-15.816574,12.515567],[-16.147717,12.547762],[-16.677452,12.384852],[-16.841525,13.151394],[-15.931296,13.130284],[-15.691001,13.270353],[-15.511813,13.27857],[-15.141163,13.509512],[-14.712197,13.298207],[-14.277702,13.280585],[-13.844963,13.505042],[-14.046992,13.794068],[-14.376714,13.62568],[-14.687031,13.630357],[-15.081735,13.876492],[-15.39877,13.860369],[-15.624596,13.623587],[-16.713729,13.594959]]]}}]}'

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

def run_task(task, logger):
    'Run an earth engine task against the ldmp API'
    logger.debug("Starting task {}.".format(task.status().get('id')))
    task.start()
    task_state = task.status().get('state')
    while task_state == 'READY' or task_state == 'RUNNING':
        task_progress = task.status().get('progress', 0.0)
        # Update GEF-EXECUTION progress
        logger.send_progress(task_progress)
        # Print message
        logger.debug("GEE task progress {}.".format(task_progress))
        task_state = task.status().get('state')
        sleep(5)
    if task_state == 'complete':
        logger.debug("GEE task completed.")
    if task_state == 'FAILED':
        logger.debug("GEE task failed: {}.".format(task.status().get('error_message')))
    return task_state
