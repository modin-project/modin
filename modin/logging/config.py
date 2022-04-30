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
            s = "%s,%03d" % (t, record.msecs)
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


def memory_thread(logger):
    while True:
        svmem = psutil.virtual_memory()
        logger.info(f"Memory Percentage: {svmem.percent}%")
        time.sleep(5)


def get_logger():
    if not __LOGGER_CONFIGURED__ and LogMode.get() != "none":
        if LogMode.get() == "api_only":
            configure_logging(logging.INFO)
        else:
            configure_logging(logging.DEBUG)

        logger = logging.getLogger("modin.logger")
        logger.info("OS Version: " + platform.platform())
        logger.info("Python Version: " + platform.python_version())
        logger.info("Modin Version: " + pkg_resources.get_distribution("modin").version)
        logger.info(
            "Pandas Version: " + pkg_resources.get_distribution("pandas").version
        )
        logger.info("Physical Cores: " + str(psutil.cpu_count(logical=False)))
        logger.info("Total Cores: " + str(psutil.cpu_count(logical=True)))
        svmem = psutil.virtual_memory()
        logger.info(f"Memory Total: {bytes_int_to_str(svmem.total)}")
        logger.info(f"Memory Available: {bytes_int_to_str(svmem.available)}")
        logger.info(f"Memory Used: {bytes_int_to_str(svmem.used)}")
        logger.info(f"Memory Percentage: {svmem.percent}%")

        if LogMode.get() != "api_only":
            try:
                mem = threading.Thread(target=memory_thread, args=[logger])
                mem.start()
            except (KeyboardInterrupt, SystemExit):
                mem.join()
                sys.exit()

    return logging.getLogger("modin.logger")
