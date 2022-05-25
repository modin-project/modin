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
Module houses `FileDispatcher` class.

`FileDispatcher` can be used as abstract base class for dispatchers of specific file formats or
for direct files processing.
"""

import fsspec
import os
import re
from modin.config import StorageFormat
from modin.utils import import_optional_dependency
from modin.logging import logger_decorator
import numpy as np

S3_ADDRESS_REGEX = re.compile("[sS]3://(.*?)/(.*)")
NOT_IMPLEMENTED_MESSAGE = "Implement in children classes!"


class OpenFile:
    """
    OpenFile is a context manager for an input file.

    OpenFile uses fsspec to open files on __enter__. On __exit__, it closes the
    fsspec file. This class exists to encapsulate the special behavior in
    __enter__ around anon=False and anon=True for s3 buckets.

    Parameters
    ----------
    file_path : str
        String that represents the path to the file (paths to S3 buckets
        are also acceptable).
    mode : str, default: "rb"
        String, which defines which mode file should be open.
    compression : str, default: "infer"
        File compression name.
    **kwargs : dict
        Keywords arguments to be passed into ``fsspec.open`` function.

    Attributes
    ----------
    file_path : str
        String that represents the path to the file
    mode : str
        String that defines which mode the file should be opened in.
    compression : str
        File compression name.
    file : fsspec.core.OpenFile
        The opened file.
    kwargs : dict
        Keywords arguments to be passed into ``fsspec.open`` function.
    """

    def __init__(self, file_path, mode="rb", compression="infer", **kwargs):
        self.file_path = file_path
        self.mode = mode
        self.compression = compression
        self.kwargs = kwargs

    def __enter__(self):
        """
        Open the file with fsspec and return the opened file.

        Returns
        -------
        fsspec.core.OpenFile
            The opened file.
        """
        try:
            from botocore.exceptions import NoCredentialsError

            credential_error_type = (
                NoCredentialsError,
                PermissionError,
            )
        except ModuleNotFoundError:
            credential_error_type = ()

        args = (self.file_path, self.mode, self.compression)

        self.file = fsspec.open(*args, **self.kwargs)
        try:
            return self.file.open()
        except credential_error_type:
            self.kwargs["anon"] = True
            self.file = fsspec.open(*args, **self.kwargs)
        return self.file.open()

    def __exit__(self, *args):
        """
        Close the file.

        Parameters
        ----------
        *args : any type
            Variable positional arguments, all unused.
        """
        self.file.close()


class FileDispatcher:
    """
    Class handles util functions for reading data from different kinds of files.

    Notes
    -----
    `_read`, `deploy`, `parse` and `materialize` are abstract methods and should be
    implemented in the child classes (functions signatures can differ between child
    classes).
    """

    frame_cls = None
    frame_partition_cls = None
    query_compiler_cls = None

    @classmethod
    @logger_decorator("PANDAS-API", "FileDispatcher.read", "INFO")
    def read(cls, *args, **kwargs):
        """
        Read data according passed `args` and `kwargs`.

        Parameters
        ----------
        *args : iterable
            Positional arguments to be passed into `_read` function.
        **kwargs : dict
            Keywords arguments to be passed into `_read` function.

        Returns
        -------
        query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `read` is high-level function that calls specific for defined storage format, engine and
        dispatcher class `_read` function with passed parameters and performs some
        postprocessing work on the resulting query_compiler object.
        """
        query_compiler = cls._read(*args, **kwargs)
        # TODO (devin-petersohn): Make this section more general for non-pandas kernel
        # implementations.
        if StorageFormat.get() == "Pandas":
            import pandas as kernel_lib
        elif StorageFormat.get() == "Cudf":
            import cudf as kernel_lib
        else:
            raise NotImplementedError("FIXME")

        if hasattr(query_compiler, "dtypes") and any(
            isinstance(t, kernel_lib.CategoricalDtype) for t in query_compiler.dtypes
        ):
            dtypes = query_compiler.dtypes
            return query_compiler.astype(
                {
                    t: dtypes[t]
                    for t in dtypes.index
                    if isinstance(dtypes[t], kernel_lib.CategoricalDtype)
                }
            )
        return query_compiler

    @classmethod
    def _read(cls, *args, **kwargs):
        """
        Perform reading of the data from file.

        Should be implemented in the child class.

        Parameters
        ----------
        *args : iterable
            Positional arguments of the function.
        **kwargs : dict
            Keywords arguments of the function.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    @logger_decorator("PANDAS-API", "FileDispatcher.get_path", "INFO")
    def get_path(cls, file_path):
        """
        Process `file_path` in accordance to it's type.

        Parameters
        ----------
        file_path : str, os.PathLike[str] object or file-like object
            The file, or a path to the file. Paths to S3 buckets are also
            acceptable.

        Returns
        -------
        str
            Updated or verified `file_path` parameter.

        Notes
        -----
        if `file_path` is an S3 bucket, parameter will be returned as is, otherwise
        absolute path will be returned.
        """
        if isinstance(file_path, str) and S3_ADDRESS_REGEX.search(file_path):
            return file_path
        else:
            return os.path.abspath(file_path)

    @classmethod
    @logger_decorator("PANDAS-API", "FileDispatcher.file_size", "INFO")
    def file_size(cls, f):
        """
        Get the size of file associated with file handle `f`.

        Parameters
        ----------
        f : file-like object
            File-like object, that should be used to get file size.

        Returns
        -------
        int
            File size in bytes.
        """
        cur_pos = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(cur_pos, os.SEEK_SET)
        return size

    @classmethod
    @logger_decorator("PANDAS-API", "FileDispatcher.file_exists", "INFO")
    def file_exists(cls, file_path):
        """
        Check if `file_path` exists.

        Parameters
        ----------
        file_path : str
            String that represents the path to the file (paths to S3 buckets
            are also acceptable).

        Returns
        -------
        bool
            Whether file exists or not.
        """
        if isinstance(file_path, str):
            match = S3_ADDRESS_REGEX.search(file_path)
            if match is not None:
                if file_path[0] == "S":
                    file_path = "{}{}".format("s", file_path[1:])
                S3FS = import_optional_dependency(
                    "s3fs", "Module s3fs is required to read S3FS files."
                )
                from botocore.exceptions import NoCredentialsError

                s3fs = S3FS.S3FileSystem(anon=False)
                exists = False
                try:
                    exists = s3fs.exists(file_path) or exists
                except NoCredentialsError:
                    pass
                s3fs = S3FS.S3FileSystem(anon=True)
                return exists or s3fs.exists(file_path)
        return os.path.exists(file_path)

    @classmethod
    def deploy(cls, func, *args, num_returns=1, **kwargs):  # noqa: PR01
        """
        Deploy remote task.

        Should be implemented in the task class (for example in the `RayTask`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def parse(self, func, args, num_returns):  # noqa: PR01
        """
        Parse file's data in the worker process.

        Should be implemented in the parser class (for example in the `PandasCSVParser`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def materialize(cls, obj_id):  # noqa: PR01
        """
        Get results from worker.

        Should be implemented in the task class (for example in the `RayTask`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        row_lengths : list
            Partitions rows lengths.
        column_widths : list
            Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partition objects.
        """
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )
