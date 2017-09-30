from marshmallow import Schema, fields, pprint

# Schema for numeric data for plotting within a timeseries object
class TimeSeries(object):
    def __init__(self, time, y, name=None):
        self.time = time
        self.y = y
        self.name = name

class TimeSeriesSchema(Schema):
    time = fields.List()
    y = fields.List()
    name = fields.Str()
    
class TimeSeriesTable(object):
    def __init__(self, type, table):
        self.type = type
        self.table = table
    
class TimeSeriesTableSchema(Schema):
    type = fields.Str()
    table = fields.Nested(TimeSeriesSchema(), many=True)
    
class GEEResults(object):
    def __init__(self, type, datasets):
        self.type = type
        self.datasets = datasets
        
class CloudDataset(object):
    def __init__(self, type, dataset, urls):
        self.type = type
        self.dataset = dataset
        self.urls = urls
        
class CloudUrl(object):
    def __init__(self, url):
        self.url = url

class CloudURLSchema(Schema):
    url = fields.Str()
    
class CloudDatasetSchema(Schema):
    dataset = fields.Str()
    type = fields.Str()
    urls = fields.Nested(CloudURLSchema(), many=True)
    
class GEEResultsSchema(Schema):
    type = fields.Str()
    datasets = fields.Nested(CloudDatasetSchema(), many=True)
