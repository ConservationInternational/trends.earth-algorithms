from marshmallow import Schema, fields


################################################################################
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
    def __init__(self, name, table):
        self.type = "TimeSeriesTable"
        self.name = name
        self.table = table


class TimeSeriesTableSchema(Schema):
    type = fields.Str()
    name = fields.Str()
    table = fields.Nested(TimeSeriesSchema(), many=True)


################################################################################
# Schema for downloads

class ComposableDict(fields.Dict):
    '''https://github.com/marshmallow-code/marshmallow/issues/432#issuecomment-210282401'''

    def __init__(self, inner, *args, **kwargs):
        self.inner = inner
        super(ComposableDict, self).__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj):
        return {
            key: self.inner._serialize(val, key, value)
            for key, val in value.datasets.items()
        }


class Metadata(object):
    def __init__(self, band_number, no_data_value):
        self.band_number = band_number
        self.no_data_value = no_data_value


class MetadataSchema(Schema):
    band_number = fields.Integer()
    no_data_value = fields.Number()


class DatasetList(object):
    def __init__(self, datasets={}):
        self.datasets = datasets


class URLList(object):
    def __init__(self, base, files=[]):
        self.base = base
        self.files = files


class URLListSchema(Schema):
    base = fields.Str()
    files = fields.List(fields.Str())

class CloudResults(object):
    def __init__(self, name, datasets, urls):
        self.type = "CloudResults"
        self.name = name
        self.datasets = datasets
        self.urls = urls

class CloudResultsSchema(Schema):
    type = fields.Str()
    name = fields.Str()
    datasets = ComposableDict(fields.Nested(MetadataSchema))
    urls = fields.Nested(URLListSchema())
