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

from functools import wraps
from types import FunctionType, MethodType

from .config import get_logger
from modin.config import LogMode


def logger_class_wrapper(classname, name, method):
    """
    Helper method for Modin logging used by the LoggerMetaClass
    to log internal Modin functions.

    Parameters
    ----------
    classname : str
        The name of the class the LoggerMetaClass is being applied to.
    name : str
        The name of the Modin function within the class.
    method : str
        The function to apply on the arguments.

    Returns
    -------
    func
        A decorator function.
    """

    @wraps(method)
    def log_wrap(*args, **kwargs):
        if LogMode.get() != "disable":
            logger = get_logger()
            logger.info(f"START::PANDAS-API::{classname}.{name}")
            result = method(*args, **kwargs)
            logger.info(f"END::PANDAS-API::{classname}.{name}")
            return result
        return method(*args, **kwargs)

    return log_wrap


def metaclass_resolver(*classes):
    """
    Metaclass resolver helper function that resolves metaclass typing issues
    resulting from class inheritance.

    Parameters
    ----------
    *classes : dict
        Dictionary of parent classes to resolve metaclass conflicts for

    Returns
    -------
    str
        The correct Metaclass that resolves inheritance conflicts.
    """
    metaclass = tuple(set(type(cls) for cls in classes))
    metaclass = (
        metaclass[0]
        if len(metaclass) == 1
        else type("_".join(mcls.__name__ for mcls in metaclass), metaclass, {})
    )
    return metaclass("_".join(cls.__name__ for cls in classes), classes, {})


class LoggerMetaClass(type):
    def __new__(mcs, classname, bases, class_dict):
        new_class_dict = {}
        seen_attributes = {}
        exclude_attributes = {"__getattribute__"}
        for attribute_name, attribute in class_dict.items():
            if (
                isinstance(attribute, (FunctionType, MethodType))
                and attribute_name not in exclude_attributes
            ):
                if attribute not in seen_attributes:
                    seen_attributes[attribute] = logger_class_wrapper(
                        classname, attribute_name, attribute
                    )
                attribute = seen_attributes[attribute]
            new_class_dict[attribute_name] = attribute
        return type.__new__(mcs, classname, bases, new_class_dict)
