class LandDegradationError(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, msg=None):
        if msg is None:
            # Set some default useful error message
            msg = "An error occured with car %s" % car
        super(CarError, self).__init__(msg)
        self.car = carpass

class GEEFailure(LandDegradationError):
    """Error running task on GEE"""
    def __init__(self, task):
        super(LandDegradationError, self).__init__(msg="Task {} failed".format(task.status().get('id')))
        self.task = task
