from .config import get_logger
from functools import wraps
from modin.config import LogMode


def logger_decorator(modin_layer: str, function_name: str, log_level: str):
    def decorator(f):
        @wraps(f)
        def run_and_log(*args, **kwargs):
            if LogMode.get() != "none":
                logger = get_logger()
                try:
                    getattr(logger, log_level.lower())(
                        f"START::{modin_layer.upper()}::{function_name}"
                    )
                except AttributeError:
                    raise AttributeError(f"Invalid log_level: {log_level}")
                result = f(*args, **kwargs)
                getattr(logger, log_level.lower())(
                    f"STOP::{modin_layer.upper()}::{function_name}"
                )
                return result
            else:
                return f(*args, **kwargs)

        return run_and_log

    return decorator
