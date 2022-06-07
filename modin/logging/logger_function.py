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

"""
Module contains ``logger_decorator`` function.

``logger_decorator`` is used for decorating individual Modin functions.
"""

from typing import Optional
from types import FunctionType, MethodType
from functools import wraps
from logging import Logger

from modin.config import LogMode
from .config import get_logger


def logger_decorator(
    modin_layer: str = "PANDAS-API",
    name: Optional[str] = None,
    log_level: Optional[str] = "info",
):
    """
    Log Decorator used on specific internal Modin functions or classes.

    Parameters
    ----------
    modin_layer : str, default: "PANDAS-API"
        Specified by the logger (e.g. PANDAS-API).
    name : str, optional
        The name of the function the decorator is being applied to.
        Taken from the decorated function name if not specified.
    log_level : str, default: "info"
        The log level (INFO, DEBUG, WARNING, etc.).

    Returns
    -------
    func
        A decorator function.
    """
    log_level = log_level.lower()
    assert hasattr(Logger, log_level.lower()), f"Invalid log level: {log_level}"

    def decorator(obj):
        """Decorate function or class to add logs to Modin API function(s)."""

        if isinstance(obj, type):
            for attr_name, attr_value in vars(obj).items():
                if isinstance(attr_value, (FunctionType, MethodType)):
                    setattr(
                        obj,
                        attr_name,
                        logger_decorator(
                            modin_layer,
                            f"{name or obj.__name__}.{attr_name}",
                            log_level,
                        )(attr_value),
                    )
            return obj

        start_line = f"START::{modin_layer.upper()}::{name or obj.__name__}"
        stop_line = f"STOP::{modin_layer.upper()}::{name or obj.__name__}"

        @wraps(obj)
        def run_and_log(*args, **kwargs):
            """
            Compute function with logging if Modin logging is enabled.

            Parameters
            ----------
            *args : tuple
                The function arguments.
            **kwargs : dict
                The function keyword arguments.

            Returns
            -------
            Any
            """
            if LogMode.get() == "disable":
                return obj(*args, **kwargs)

            logger = get_logger()
            logger_level = getattr(logger, log_level)
            logger_level(start_line)
            try:
                result = obj(*args, **kwargs)
            except BaseException:
                logger.exception(stop_line)
                raise
            logger_level(stop_line)
            return result

        return run_and_log

    return decorator
