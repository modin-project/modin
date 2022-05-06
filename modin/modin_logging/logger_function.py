from .config import get_logger
from functools import wraps
from modin.config import LogMode


def logger_decorator(modin_layer: str, function_name: str, log_level: str):
    def decorator(f):
        @wraps(f)
        def run_and_log(*args, **kwargs):
            if LogMode.get() == "disable":
                return f(*args, **kwargs)

            logger = get_logger()
            try:
                logger_level = getattr(logger, log_level.lower())
            except AttributeError:
                raise AttributeError(f"Invalid log_level: {log_level}")

            logger_level(f"START::{modin_layer.upper()}::{function_name}")
            result = f(*args, **kwargs)
            logger_level(f"STOP::{modin_layer.upper()}::{function_name}")
            return result

        return run_and_log

    return decorator
