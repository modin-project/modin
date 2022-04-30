from functools import wraps
from .config import get_logger
from types import FunctionType, MethodType
from modin.config import LogMode


def logger_class_wrapper(classname, name, method):
    @wraps(method)
    def log_wrap(*args, **kwargs):
        if LogMode.get() != "none":
            logger = get_logger()
            logger.info(f"START::PANDAS-API::{classname}.{name}")
            result = method(*args, **kwargs)
            logger.info(f"END::PANDAS-API::{classname}.{name}")
            return result
        else:
            return method(*args, **kwargs)

    return log_wrap


class LoggerMetaClass(type):
    def __new__(mcs, classname, bases, class_dict):
        new_class_dict = {}
        for attribute_name, attribute in class_dict.items():
            if (
                isinstance(attribute, (FunctionType, MethodType))
                and attribute_name != "__getattribute__"
            ):
                attribute = logger_class_wrapper(classname, attribute_name, attribute)
            new_class_dict[attribute_name] = attribute
        return type.__new__(mcs, classname, bases, new_class_dict)
