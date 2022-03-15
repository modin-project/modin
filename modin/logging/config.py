import logging
import datetime as dt
import os
import uuid
import platform
import psutil
import pkg_resources

__LOGGER_CONFIGURED__: bool = False


class MyFormatter(logging.Formatter):
    converter = dt.datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s


def configure_logging():
    global __LOGGER_CONFIGURED__
    logger = logging.getLogger("modin.logger")
    job_id = uuid.uuid4().hex
    log_filename = f".modin/logs/job_{job_id}.log"

    if not os.path.isdir(".modin/logs"):
        os.makedirs(os.path.dirname(log_filename), exist_ok=False)

    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(log_filename, "a")
    formatter = MyFormatter(fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%f')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    __LOGGER_CONFIGURED__ = True


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_logger():
    if not __LOGGER_CONFIGURED__:
        configure_logging()
        logger = logging.getLogger("modin.logger")
        logger.info("OS Version: " + platform.platform())
        logger.info("Python Version: " + platform.python_version())
        logger.info("Modin Version: " + pkg_resources.get_distribution("modin").version)
        logger.info("Pandas Version: " + pkg_resources.get_distribution("pandas").version)
        logger.info("Physical Cores: " + str(psutil.cpu_count(logical=False)))
        logger.info("Total Cores: " + str(psutil.cpu_count(logical=True)))
        svmem = psutil.virtual_memory()
        logger.info(f"Memory Total: {get_size(svmem.total)}")
        logger.info(f"Memory Available: {get_size(svmem.available)}")
        logger.info(f"Memory Used: {get_size(svmem.used)}")
        logger.info(f"Memory Percentage: {svmem.percent}%")
    return logging.getLogger("modin.logger")
