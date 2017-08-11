class LandDegradationError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, msg=None):
        if msg is None:
            # Set some default useful error message
            msg = "An error occurred in the landdegradation module"
        super(LandDegradationError, self).__init__(msg)

class GEEError(LandDegradationError):
    """Error related to GEE"""
    def __init__(self, task):
        super(LandDegradationError, self).__init__(None)

class GEEIOError(GEEError):
    """Error related to GEE"""
    def __init__(self, task):
        super(GEEError, self).__init__("Error with GEE JSON IO")

class GEETaskFailure(GEEError):
    """Error running task on GEE"""
    def __init__(self, task):
        super(GEEError, self).__init__("Task {} failed".format(task.status().get('id')))
        self.task = task
