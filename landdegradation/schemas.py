from marshmallow import Schema, fields, ComposableDict

class GEEResults(object):
    def __init__(self, type, datasets):
        self.type = type
        self.datasets = datasets
    
class GEEResultsSchema(Schema):
    type = fields.Str()
    datasets = fields.Nested(DatasetSchema(), many=True)
        
# Schema for numeric data for plotting within a timeseries object
class TimeSeries(object):
    def __init__(self, time, y, name=None):
        self.time = time
        self.y = y
        self.name = name

class TimeSeriesSchema(Schema):
    time = fields.List(fields.Float())
    y = fields.List(fields.Float())
    name = fields.Str()
    
class TimeSeriesTable(object):
    def __init__(self, type, table):
        self.type = type
        self.table = table
    
class TimeSeriesTableSchema(Schema):
    type = fields.Str()
    table = fields.Nested(TimeSeriesSchema(), many=True)
    
# Schema for downloads
class Dataset(object):
    def __init__(self, band_number, no_data_value):
        self.band_number = band_number
        self.no_data_value = no_data_value

class DatasetSchema(Schema):
    band_number = fields.Number(, many=True)
    no_data_value = fields.Number()

class DatasetList(object):
    def __init__(self, datasets={}):
        self.datasets = datasets

class UrlList(object):
    def __init__(self, base, files=[]):
        self.base = base
        self.files = files

class URLListSchema(Schema):
    base = fields.Str()
    files = fields.List(fields.Str())
    
class DatasetsSchema(Schema):
    type = fields.Str()
    datasets = ComposableDict(fields.Nested(DatasetSchema))
    urls = fields.Nested(URLListSchema())
