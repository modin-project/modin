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

from functools import wraps

from modin.config import LogMode
from .config import get_logger


def logger_decorator(modin_layer: str, function_name: str, log_level: str):
    """
    Log Decorator used on specific internal Modin functions.

    Parameters
    ----------
    modin_layer : str
        Specified by the logger (PANDAS-API).
    function_name : str
        The name of the function the decorator is being applied to.
    log_level : str
        The log level (logging.INFO, logging.DEBUG, logging.WARNING, etc.).

    Returns
    -------
    func
        A decorator function.
    """

    def decorator(f):
        """Decorate function to add logs to Modin API function."""

        @wraps(f)
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
