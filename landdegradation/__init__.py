class LandDegradationError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, msg=None):
        if msg is None:
            # Set some default useful error message
            msg = "An error occurred in the landdegradation module"
        super(LandDegradationError, self).__init__(msg)

class GEEFailure(LandDegradationError):
    """Error running task on GEE"""
    def __init__(self, task):
        super(LandDegradationError, self).__init__("Task {} failed".format(task.status().get('id')))
        self.task = task
