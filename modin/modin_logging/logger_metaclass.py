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
from .config import get_logger
from types import FunctionType, MethodType
from modin.config import LogMode


def logger_class_wrapper(classname, name, method):
    """
    Helper method for Modin logging used by the LoggerMetaClass
    to log internal Modin functions.

    Parameters
    ----------
    classname: str
        The name of the class the LoggerMetaClass is being applied to.
    name: str
        The name of the Modin function within the class.
    method: str
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
