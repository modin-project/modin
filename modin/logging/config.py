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
import pandas
import threading
import time
from typing import Optional

import modin
from modin.config import LogMemoryInterval, LogFileSize, LogMode

__LOGGER_CONFIGURED__: bool = False


class ModinFormatter(logging.Formatter):  # noqa: PR01
    """Implement custom formatter to log at microsecond granularity."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
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
        str
            Datetime string containing microsecond timestamp.
        """
        ct = dt.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            # Format datetime object ct to microseconds
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{t},{record.msecs:03}"
        return s


def bytes_int_to_str(num_bytes: int, suffix: str = "B") -> str:
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
    # Convert n_bytes to float b/c we divide it by factor
    n_bytes: float = num_bytes
    for unit in ["", "K", "M", "G", "T", "P"]:
        if n_bytes < factor:
            return f"{n_bytes:.2f}{unit}{suffix}"
        n_bytes /= factor
    return f"{n_bytes * 1000:.2f}P{suffix}"


def _create_logger(
    namespace: str, job_id: str, log_name: str, log_level: int
) -> logging.Logger:
    """
    Create and configure logger as Modin expects it to be.

    Parameters
    ----------
    namespace : str
        Logging namespace to use, e.g. "modin.logger.default".
    job_id : str
        Part of path to where logs are stored.
    log_name : str
        Name of the log file to create.
    log_level : int
        Log level as accepted by `Logger.setLevel()`.

    Returns
    -------
    Logger
        Logger object configured per Modin settings.
    """
    log_filename = f".modin/logs/job_{job_id}/{log_name}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    logger = logging.getLogger(namespace)
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
    logger.setLevel(log_level)

    return logger


def configure_logging() -> None:
    """Configure Modin logging by setting up directory structure and formatting."""
    global __LOGGER_CONFIGURED__
    job_id = uuid.uuid4().hex

    logger = _create_logger(
        "modin.logger.default",
        job_id,
        "trace",
        logging.INFO if LogMode.get() == "enable_api_only" else logging.DEBUG,
    )

    logger.info(f"OS Version: {platform.platform()}")
    logger.info(f"Python Version: {platform.python_version()}")
    num_physical_cores = str(psutil.cpu_count(logical=False))
    num_total_cores = str(psutil.cpu_count(logical=True))
    logger.info(f"Modin Version: {modin.__version__}")
    logger.info(f"Pandas Version: {pandas.__version__}")
    logger.info(f"Physical Cores: {num_physical_cores}")
    logger.info(f"Total Cores: {num_total_cores}")

    if LogMode.get() != "enable_api_only":
        mem_sleep = LogMemoryInterval.get()
        mem_logger = _create_logger(
            "modin_memory.logger", job_id, "memory", logging.DEBUG
        )

        svmem = psutil.virtual_memory()
        mem_logger.info(f"Memory Total: {bytes_int_to_str(svmem.total)}")
        mem_logger.info(f"Memory Available: {bytes_int_to_str(svmem.available)}")
        mem_logger.info(f"Memory Used: {bytes_int_to_str(svmem.used)}")
        mem = threading.Thread(
            target=memory_thread, args=[mem_logger, mem_sleep], daemon=True
        )
        mem.start()

    _create_logger("modin.logger.errors", job_id, "error", logging.INFO)

    __LOGGER_CONFIGURED__ = True


def memory_thread(logger: logging.Logger, sleep_time: int) -> None:
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


def get_logger(namespace: str = "modin.logger.default") -> logging.Logger:
    """
    Configure Modin logger based on Modin config and returns the logger.

    Parameters
    ----------
    namespace : str, default: "modin.logger.default"
        Which namespace to use for logging.

    Returns
    -------
    logging.Logger
        The Modin logger.
    """
    if not __LOGGER_CONFIGURED__ and LogMode.get() != "disable":
        configure_logging()
    return logging.getLogger(namespace)
