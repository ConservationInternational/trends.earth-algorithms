try:
    from te_algorithms._version import __version__, __git_sha__, __git_date__
except ImportError:
    __version__ = "unknown"
    __git_sha__ = "unknown"
    __git_date__ = "unknown"
    import logging

    logging.warning(
        "te_algorithms version could not be determined. "
        "If you're running from source, please run 'invoke set-version' first. "
        "If you're running from a package, this may indicate a packaging issue."
    )


class TEAlgorithmsError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg=None):
        if msg is None:
            msg = "An error occurred in the te_algorithms module"
        super().__init__(msg)
