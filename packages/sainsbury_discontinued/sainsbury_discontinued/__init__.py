# -*- coding: utf-8 -*-
import logging
from sainsbury_discontinued.config import config
from sainsbury_discontinued.config import logging_config
import os



# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False



with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()



