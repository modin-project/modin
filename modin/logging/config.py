import logging
import os

__LOGGER_CONFIGURED__: bool = False


def configure_logging():
    global __LOGGER_CONFIGURED__
    logger = logging.getLogger("modin.logger")
    job_id = "0001"
    log_filename = f".modin/logs/job_{job_id}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(log_filename, "a")
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d: %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
    )
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    __LOGGER_CONFIGURED__ = True


def get_logger():
    if not __LOGGER_CONFIGURED__:
        configure_logging()
    return logging.getLogger("modin.logger")
