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
Module contains ``LoggerMetaClass`` class.

``LoggerMetaClass`` is used for adding logging to Modin classes.
"""
from typing import Optional

from .logger_function import logger_decorator


def metaclass_resolver(*classes):
    """
    Resolve metaclass typing issues resulting from class inheritance.

    Parameters
    ----------
    *classes : dict
        Dictionary of parent classes to resolve metaclass conflicts for.

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


class LoggerMixin:
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
        super().__init_subclass__(**kwargs)
        logger_decorator(modin_layer, class_name, log_level)(cls)
