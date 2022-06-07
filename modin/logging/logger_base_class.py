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
Module contains ``LoggerBase`` class.

``LoggerBase`` is used for adding logging to Modin classes and their subclasses.
"""
from typing import Optional

from .logger_function import logger_decorator


class LoggerBase:
    """
    Ensure all subclasses of the class being inherited are logged, too.

    Notes
    -----
    This mixin must go first in class bases declaration to have the desired effect.
    """

    def __init_subclass__(
        cls,
        modin_layer: str = "PANDAS-API",
        class_name: Optional[str] = None,
        log_level: Optional[str] = "info",
        **kwargs,
    ) -> None:
        """
        Apply logging decorator to all children of LoggerBase.

        Parameters
        ----------
        modin_layer : str, default: "PANDAS-API"
            Specified by the logger (e.g. PANDAS-API).
        class_name : str, optional
            The name of the class the decorator is being applied to.
            Composed from the decorated class name if not specified.
        log_level : str, default: "info"
            The log level (INFO, DEBUG, WARNING, etc.).
        **kwargs : dict
        """
        super().__init_subclass__(**kwargs)
        logger_decorator(modin_layer, class_name, log_level)(cls)
