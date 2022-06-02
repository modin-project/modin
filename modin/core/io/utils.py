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

"""Collection of utility functions for distributed io."""

import os
import pathlib
import re
from typing import Optional, Union
import fsspec

S3_ADDRESS_REGEX = re.compile("[sS]3://(.*?)/(.*)")


def is_local_path(path) -> bool:
    """
    Return ``True`` if the specified `path` is a local path, ``False`` otherwise.

    Parameters
    ----------
    path : str, path object or file-like object
        The path to check.

    Returns
    -------
    Whether the `path` points to a local file.
    """
    try:
        fsspec.open_local(
            "/".join(path.split("/")[:-1])
        )  # Remove file name since that may not exist
        local_device_id = os.stat(os.getcwd()).st_dev
        path_device_id = get_device_id(path)
        return path_device_id == local_device_id
    except Exception:
        return False


def get_device_id(path: Union[str, pathlib.PurePath]) -> Optional[int]:
    """
    Return the result of `os.stat(path).st_dev` for the portion of `path` that exists locally.

    Parameters
    ----------
    path : str, path object
        The path to check.

    Returns
    -------
    The `st_dev` field of `os.stat` of the portion of the `path` that exists locally, None if no
    part of the path exists locally.
    """
    index = 1
    path_list = list(pathlib.Path(path).parts)
    if path_list[0] == "/":
        index += 1
    try:
        os.stat(os.path.join(*path_list[:index]))
    except Exception:
        return None
    while os.path.exists(os.path.join(*path_list[:index])) and index <= len(path_list):
        index += 1
    index -= 1
    return os.stat(os.path.join(*path_list[:index])).st_dev
