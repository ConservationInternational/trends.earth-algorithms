import os
import re
import json
import logging

logger = logging.getLogger(__name__)

plugin_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(plugin_dir, 'version.json')) as f:
    version_info = json.load(f)
__version__ = version_info['version']
__version_major__ = re.sub(r'([0-9]+)(\.[0-9]+)+$', r'\g<1>', __version__)
__revision__ = version_info['revision']
__release_date__ = version_info['release_date']


class TEAlgorithmsError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg=None):
        if msg is None:
            msg = "An error occurred in the te_algorithms module"
        super(TEAlgorithmsError, self).__init__(msg)
