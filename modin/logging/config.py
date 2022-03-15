import logging
import datetime as dt
import os

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
    job_id = "0001"
    log_filename = f".modin/logs/job_{job_id}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(log_filename, "a")
    formatter = MyFormatter(fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%f')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    __LOGGER_CONFIGURED__ = True


def get_logger():
    if not __LOGGER_CONFIGURED__:
        configure_logging()
    return logging.getLogger("modin.logger")
