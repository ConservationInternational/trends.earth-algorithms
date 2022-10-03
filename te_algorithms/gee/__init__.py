from .. import TEAlgorithmsError


class GEEError(TEAlgorithmsError):
    """Error related to GEE"""

    def __init__(self, msg="Error with GEE JSON IO"):
        super(TEAlgorithmsError, self).__init__(msg)


class GEEIOError(GEEError):
    """Error related to GEE"""

    def __init__(self, msg="Error with GEE JSON IO"):
        super(GEEError, self).__init__(msg)


class GEEImageError(GEEError):
    """Error related to GEE"""

    def __init__(self, msg="Error with GEE image handling"):
        super(GEEError, self).__init__(msg)


class GEETaskFailure(GEEError):
    """Error running task on GEE"""

    def __init__(self, task):
        super(GEEError, self).__init__("Task {} failed".format(task.status().get("id")))
        self.task = task
