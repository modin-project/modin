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

import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from .config import Engine, StorageFormat

from . import _version


def custom_formatwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    # ignore everything except the message
    return "{}: {}\n".format(category.__name__, message)


warnings.formatwarning = custom_formatwarning
# Filter numpy version warnings because they are not relevant
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="Large object of size")


def set_execution(
    engine: Any = None, storage_format: Any = None
) -> Tuple["Engine", "StorageFormat"]:
    """
    Method to set the _pair_ of execution engine and storage format format simultaneously.
    This is needed because there might be cases where switching one by one would be
    impossible, as not all pairs of values are meaningful.

    The method returns pair of old values, so it is easy to return back.
    """
    from .config import Engine, StorageFormat

    old_engine, old_storage_format = None, None
    # defer callbacks until both entities are set
    if engine is not None:
        old_engine = Engine._put_nocallback(engine)
    if storage_format is not None:
        old_storage_format = StorageFormat._put_nocallback(storage_format)
    # execute callbacks if something was changed
    if old_engine is not None:
        Engine._check_callbacks(old_engine)
    if old_storage_format is not None:
        StorageFormat._check_callbacks(old_storage_format)

    return old_engine, old_storage_format


__version__ = _version.get_versions()["version"]
