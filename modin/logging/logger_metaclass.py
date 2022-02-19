from functools import wraps
from .config import get_logger
from types import FunctionType, MethodType


def logger_class_wrapper(classname, name, method):
    @wraps(method)
    def log_wrap(*args, **kwargs):
        logger = get_logger()
        logger.info(f"START::PANDAS-API::{classname}.{name}")
        result = method(*args, **kwargs)
        logger.info(f"END::PANDAS-API::{classname}.{name}")
        return result

    return log_wrap


class LoggerMetaClass(type):
    def __new__(mcs, classname, bases, class_dict):
        new_class_dict = {}
        for attribute_name, attribute in class_dict.items():
            if (
                isinstance(attribute, (FunctionType, MethodType))
                and attribute_name != "__getattribute__"
            ):  # and (attribute_name[0] != "_" or attribute_name[1] == "_"):
                attribute = logger_class_wrapper(classname, attribute_name, attribute)
            new_class_dict[attribute_name] = attribute
        return type.__new__(mcs, classname, bases, new_class_dict)
