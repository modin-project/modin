import logging
import datetime as dt
import os
import uuid
import platform
import psutil
import pkg_resources
import threading
import time
from modin.config import LogMode

__LOGGER_CONFIGURED__: bool = False


class ModinFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        """
        Return the creation time of the specified LogRecord as formatted text.
        This custom logging formatter inherits from the logging module and
        records timestamps at the microsecond level of granularity.

        Parameters
        ----------
        record: LogRecord
            The specified LogRecord object.
        datefmt: str, default: None
            Used with time.ststrftime() to format time record.

        Returns
        -------
        datetime
            datetime object containing microsecond timestamp.
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
    Scale bytes to its human-readable format (e.g: 1253656678 => '1.17GiB').

    Parameters
    ----------
    num_bytes: int
        Number of bytes.
    suffix: str, default: "B"
        Suffix to add to conversion of num_bytes.

    Returns
    -------
    str
        Human-readable string format.
    """
    factor = 1024
    if num_bytes > 1000000000000000000:
        raise ValueError("System memory exceeds expectations")
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if num_bytes < factor:
            return f"{num_bytes:.2f}{unit}{suffix}"
        num_bytes /= factor


def configure_logging(level):
    global __LOGGER_CONFIGURED__
    logger = logging.getLogger("modin.logger")
    job_id = uuid.uuid4().hex
    log_filename = f".modin/logs/job_{job_id}.log"

    if not os.path.isdir(".modin/logs"):
        os.makedirs(os.path.dirname(log_filename), exist_ok=False)

    logfile = logging.FileHandler(log_filename, "a")
    formatter = ModinFormatter(
        fmt="%(process)d, %(thread)d, %(asctime)s, %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S.%f",
    )
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    logger.setLevel(level)

    __LOGGER_CONFIGURED__ = True


def memory_thread(logger, sleep_time):
    while True:
        svmem = psutil.virtual_memory()
        logger.info(f"Memory Percentage: {svmem.percent}%")
        time.sleep(sleep_time)


def get_logger(mem_sleep=5):
    if not __LOGGER_CONFIGURED__ and LogMode.get() != "none":
        if LogMode.get() == "api_only":
            configure_logging(logging.INFO)
        else:
            configure_logging(logging.DEBUG)

        logger = logging.getLogger("modin.logger")
        logger.info(f"OS Version: {platform.platform()}")
        logger.info(f"Python Version: {platform.python_version()}")
        modin_version = pkg_resources.get_distribution("modin").version
        pandas_version = pkg_resources.get_distribution("pandas").version
        num_physical_cores = str(psutil.cpu_count(logical=False))
        num_total_cores = str(psutil.cpu_count(logical=True))
        svmem = psutil.virtual_memory()
        logger.info(f"Modin Version: {modin_version}")
        logger.info(f"Pandas Version: {pandas_version}")
        logger.info(f"Physical Cores: {num_physical_cores}")
        logger.info(f"Total Cores: {num_total_cores}")
        logger.info(f"Memory Total: {bytes_int_to_str(svmem.total)}")
        logger.info(f"Memory Available: {bytes_int_to_str(svmem.available)}")
        logger.info(f"Memory Used: {bytes_int_to_str(svmem.used)}")
        logger.info(f"Memory Percentage: {svmem.percent}%")

        if LogMode.get() != "api_only":
            try:
                mem = threading.Thread(target=memory_thread, args=[logger, mem_sleep])
                mem.start()
            except (KeyboardInterrupt, SystemExit):
                mem.join()
                sys.exit()

    return logging.getLogger("modin.logger")
