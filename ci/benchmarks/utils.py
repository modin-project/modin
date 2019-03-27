from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

import logging
import time


@contextmanager
def time_logger(name):
    """This logs the time usage of a code block"""
    start_time = time.time()
    yield
    end_time = time.time()
    total_time = end_time - start_time

    logging.info("%s; time: %ss", name, total_time)
