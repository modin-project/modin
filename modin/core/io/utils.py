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
import re
import fsspec

IS_FILE_ONLY_REGEX = re.compile("[^\/]*\.\w+")  # noqa: W605


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

    Notes
    -----
    If the filesystem corresponds to a `ZipFileSystem`, `TarFileSystem` or `CachingFileSystem`,
    this code will return `False` even if it is local.
    """
    try:
        if IS_FILE_ONLY_REGEX.match(path) is not None:
            # If we are passed just a filename, we will perform our check on the current working
            # directory.
            parent_dir = os.getcwd()
        else:
            # If we are passed a full path, we want to remove the filename from it.
            parent_dir = "/".join(path.split("/")[:-1])
        fs = fsspec.core.url_to_fs(parent_dir)[0]  # Grab just the FileSystem object
        if hasattr(
            fs, "local_file"
        ):  # If the FS does not have the `local_file` attr, it is not local.
            # We still need to check that it is not a mounted file - as fsspec treats mounted
            # files the same as local ones, but we want to distinguish between local and mounted.
            local_device_id = os.stat(os.path.abspath(os.sep)).st_dev
            path_device_id = os.stat(parent_dir).st_dev
            return path_device_id == local_device_id
        return False
    except Exception:
        # If an exception is raised, it means we tried to open a filesystem that requires additional
        # dependencies. This means that it is definitely not a local filesystem, so we can return
        # `False` here.
        return False
