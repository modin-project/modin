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

S3_ADDRESS_REGEX = re.compile("[sS]3://(.*?)/(.*)")


def is_local_path(path_or_buf) -> bool:
    """
    Return True if the specified path_or_buf is a local path, False otherwise.

    Parameters
    ----------
    path_or_buf : str, path object or file-like object
        The path or buffer to check.

    Returns
    -------
    Whether the `path_or_buf` points to a local file.
    """
    if isinstance(path_or_buf, str):
        if S3_ADDRESS_REGEX.match(path_or_buf) is not None or "://" in path_or_buf:
            return False  # S3 or network path.
    if isinstance(path_or_buf, str) or isinstance(path_or_buf, pathlib.PurePath):
        return os.path.exists(path_or_buf)
    return False
