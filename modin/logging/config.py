# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Module contains ``ModinFormatter`` class.

``ModinFormatter`` and the associated functions are used for logging configuration.
"""

import logging
from logging.handlers import RotatingFileHandler
import datetime as dt
import os
import uuid
import platform
import psutil
import pkg_resources
import threading
import time
from modin.config import LogMemoryInterval, LogFileSize, LogMode

__LOGGER_CONFIGURED__: bool = False


class ModinFormatter(logging.Formatter):  # noqa: PR01
    """Implement custom formatter to log at microsecond granularity."""

    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as formatted text.

        This custom logging formatter inherits from the logging module and
        records timestamps at the microsecond level of granularity.

        Parameters
        ----------
        record : LogRecord
            The specified LogRecord object.
        datefmt : str, default: None
            Used with time.ststrftime() to format time record.

        Returns
        -------
        datetime
            Datetime object containing microsecond timestamp.
        """
        ct = dt.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            # Format datetime object ct to microseconds
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{t},{record.msecs:03}"
        return s


def bytes_int_to_str(num_bytes, suffix="B"):
    """
    Scale bytes to its human-readable format (e.g: 1253656678 => '1.17GB').

    Parameters
    ----------
    num_bytes : int
        Number of bytes.
    suffix : str, default: "B"
        Suffix to add to conversion of num_bytes.

    Returns
    -------
    str
        Human-readable string format.
    """
    factor = 1000
    for unit in ["", "K", "M", "G", "T", "P"]:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}{suffix}"
        num_bytes /= factor
    return f"{num_bytes:.2f}{1000+P}{suffix}"


def configure_logging():
    """Configure Modin logging by setting up directory structure and formatting."""
    global __LOGGER_CONFIGURED__
    logger = logging.getLogger("modin.logger")
    job_id = uuid.uuid4().hex
    log_filename = f".modin/logs/job_{job_id}/trace.log"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    logfile = RotatingFileHandler(
        filename=log_filename,
        mode="a",
        maxBytes=LogFileSize.get() * int(1e6),
        backupCount=10,
    )
    formatter = ModinFormatter(
        fmt="%(process)d, %(thread)d, %(asctime)s, %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S.%f",
    )
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    if LogMode.get() == "enable_api_only":
        logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    logger.info(f"OS Version: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    modin_version = pkg_resources.get_distribution("modin").version
    pandas_version = pkg_resources.get_distribution("pandas").version
    num_physical_cores = str(psutil.cpu_count(logical=False))
    num_total_cores = str(psutil.cpu_count(logical=True))
    logger.info(f"Modin Version: {modin_version}")
    logger.info(f"Pandas Version: {pandas_version}")
    logger.info(f"Physical Cores: {num_physical_cores}")
    logger.info(f"Total Cores: {num_total_cores}")

    if LogMode.get() != "enable_api_only":
        mem_sleep = LogMemoryInterval.get()
        mem_logger = logging.getLogger("modin_memory.logger")
        mem_log_filename = f".modin/logs/job_{job_id}/memory.log"
        logfile = RotatingFileHandler(
            filename=mem_log_filename,
            mode="a",
            maxBytes=LogFileSize.get() * int(1e6),
            backupCount=10,
        )
        logfile.setFormatter(formatter)
        mem_logger.addHandler(logfile)
        mem_logger.setLevel(logging.DEBUG)

        svmem = psutil.virtual_memory()
        mem_logger.info(f"Memory Total: {bytes_int_to_str(svmem.total)}")
        mem_logger.info(f"Memory Available: {bytes_int_to_str(svmem.available)}")
        mem_logger.info(f"Memory Used: {bytes_int_to_str(svmem.used)}")
        mem = threading.Thread(
            target=memory_thread, args=[mem_logger, mem_sleep], daemon=True
        )
        mem.start()

    __LOGGER_CONFIGURED__ = True


def memory_thread(logger, sleep_time):
    """
    Configure Modin logging system memory profiling thread.

    Parameters
    ----------
    logger : logging.Logger
        The logger object.
    sleep_time : int
        The interval at which to profile system memory.
    """
    while True:
        rss_mem = bytes_int_to_str(psutil.Process().memory_info().rss)
        svmem = psutil.virtual_memory()
        logger.info(f"Memory Percentage: {svmem.percent}%")
        logger.info(f"RSS Memory: {rss_mem}")
        time.sleep(sleep_time)


def get_logger():
    """
    Configure Modin logger based on Modin config and returns the logger.

    Returns
    -------
    logging.Logger
        The Modin logger.
    """
    if not __LOGGER_CONFIGURED__ and LogMode.get() != "disable":
        configure_logging()
    return logging.getLogger("modin.logger")
