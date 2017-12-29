__version__ = '0.3'


class LandDegradationError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg=None):
        if msg is None:
            msg = "An error occurred in the landdegradation module"
        super(LandDegradationError, self).__init__(msg)


class GEEError(LandDegradationError):
    """Error related to GEE"""

    def __init__(self, msg="Error with GEE JSON IO"):
        super(LandDegradationError, self).__init__(msg)


class GEEIOError(GEEError):
    """Error related to GEE"""

    def __init__(self, msg="Error with GEE JSON IO"):
        super(GEEError, self).__init__(msg)


class GEETaskFailure(GEEError):
    """Error running task on GEE"""

    def __init__(self, task):
        super(GEEError, self).__init__("Task {} failed".format(task.status().get('id')))
        self.task = task
