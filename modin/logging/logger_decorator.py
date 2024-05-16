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
Module contains the functions designed for the enable/disable of logging.

``enable_logging`` is used for decorating individual Modin functions or classes.
"""

from __future__ import annotations

from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, overload

from modin.config import LogMode

from .config import LogLevel, get_logger

_MODIN_LOGGER_NOWRAP = "__modin_logging_nowrap__"

Fn = TypeVar("Fn", bound=Any)


def disable_logging(func: Callable) -> Any:
    """
    Disable logging of one particular function. Useful for decorated classes.

    Parameters
    ----------
    func : callable
        A method in a logger-decorated class for which logging should be disabled.

    Returns
    -------
    func
        A function with logging disabled.
    """
    setattr(func, _MODIN_LOGGER_NOWRAP, True)
    return func


@overload
def enable_logging(modin_layer: Fn) -> Fn:
    # This helps preserve typings when the decorator is used without parentheses
    pass


@overload
def enable_logging(
    modin_layer: str = "PANDAS-API",
    name: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
) -> Callable[[Fn], Fn]:
    pass


def enable_logging(
    modin_layer: str | Fn = "PANDAS-API",
    name: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
) -> Callable[[Fn], Fn] | Fn:
    """
    Log Decorator used on specific Modin functions or classes.

    Parameters
    ----------
    modin_layer : str or object to decorate, default: "PANDAS-API"
        Specified by the logger (e.g. PANDAS-API).
        If it's an object to decorate, call logger_decorator() on it with default arguments.
    name : str, optional
        The name of the object the decorator is being applied to.
        Composed from the decorated object name if not specified.
    log_level : LogLevel, default: LogLevel.INFO
        The log level (LogLevel.INFO, LogLevel.DEBUG, LogLevel.WARNING, etc.).

    Returns
    -------
    func
        A decorator function.
    """
    if not isinstance(modin_layer, str):
        # assume the decorator is used in a form without parenthesis like:
        # @enable_logging
        # def func()
        return enable_logging()(modin_layer)

    def decorator(obj: Fn) -> Fn:
        """Decorate function or class to add logs to Modin API function(s)."""
        if isinstance(obj, type):
            seen: Dict[Any, Any] = {}
            for attr_name, attr_value in vars(obj).items():
                if isinstance(
                    attr_value, (FunctionType, MethodType, classmethod, staticmethod)
                ) and not hasattr(attr_value, _MODIN_LOGGER_NOWRAP):
                    try:
                        wrapped = seen[attr_value]
                    except KeyError:
                        wrapped = seen[attr_value] = enable_logging(
                            modin_layer,
                            f"{name or obj.__name__}.{attr_name}",
                            log_level,
                        )(attr_value)

                    setattr(obj, attr_name, wrapped)
            return obj  # type: ignore [return-value]
        elif isinstance(obj, classmethod):
            return classmethod(decorator(obj.__func__))  # type: ignore [return-value, arg-type]
        elif isinstance(obj, staticmethod):
            return staticmethod(decorator(obj.__func__))  # type: ignore [return-value, arg-type]

        assert isinstance(modin_layer, str), "modin_layer is somehow not a string!"

        start_line = f"START::{modin_layer.upper()}::{name or obj.__name__}"
        stop_line = f"STOP::{modin_layer.upper()}::{name or obj.__name__}"

        @wraps(obj)
        def run_and_log(*args: Tuple, **kwargs: Dict) -> Any:
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
            logger.log(log_level, start_line)
            try:
                result = obj(*args, **kwargs)
            except BaseException as e:
                # Only log the exception if a deeper layer of the modin stack has not
                # already logged it.
                if not hasattr(e, "_modin_logged"):
                    # use stack_info=True so that even if we are a few layers deep in
                    # modin, we log a stack trace that includes calls to higher layers
                    # of modin
                    get_logger("modin.logger.errors").exception(
                        stop_line, stack_info=True
                    )
                    e._modin_logged = True  # type: ignore[attr-defined]
                raise
            finally:
                logger.log(log_level, stop_line)
            return result

        # make sure we won't decorate multiple times
        return disable_logging(run_and_log)

    return decorator
