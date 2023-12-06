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
Implement various accessor classes for DataFrame and Series API.

SparseFrameAccessor implements API of pandas.DataFrame.sparse accessor.

SparseAccessor implements API of pandas.Series.sparse accessor.

CachedAccessor implements API of pandas.core.accessor.CachedAccessor
"""

import pickle

import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype

from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings


class BaseSparseAccessor(ClassLogger):
    """
    Base class for various sparse DataFrame accessor classes.

    Parameters
    ----------
    data : DataFrame or Series
        Object to operate on.
    """

    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None):
        self._parent = data
        self._validate(data)

    @classmethod
    def _validate(cls, data):
        """
        Verify that `data` dtypes are compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : DataFrame
            Object to check.

        Raises
        ------
        NotImplementedError
            Function is implemented in child classes.
        """
        raise NotImplementedError

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert dataset to pandas type and call a pandas sparse.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._parent._default_to_pandas(
            lambda parent: op(parent.sparse, *args, **kwargs)
        )


@_inherit_docstrings(pandas.core.arrays.sparse.accessor.SparseFrameAccessor)
class SparseFrameAccessor(BaseSparseAccessor):
    @classmethod
    def _validate(cls, data):
        """
        Verify that `data` dtypes are compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : DataFrame
            Object to check.

        Raises
        ------
        AttributeError
            If check fails.
        """
        dtypes = data.dtypes
        if not all(isinstance(t, SparseDtype) for t in dtypes):
            raise AttributeError(cls._validation_msg)

    @property
    def density(self):
        return self._parent._default_to_pandas(pandas.DataFrame.sparse).density

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None):
        ErrorMessage.default_to_pandas("`from_spmatrix`")
        return pd.DataFrame(
            pandas.DataFrame.sparse.from_spmatrix(data, index=index, columns=columns)
        )

    def to_dense(self):
        return self._default_to_pandas(pandas.DataFrame.sparse.to_dense)

    def to_coo(self):
        return self._default_to_pandas(pandas.DataFrame.sparse.to_coo)


@_inherit_docstrings(pandas.core.arrays.sparse.accessor.SparseAccessor)
class SparseAccessor(BaseSparseAccessor):
    @classmethod
    def _validate(cls, data):
        """
        Verify that `data` dtype is compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : Series
            Object to check.

        Raises
        ------
        AttributeError
            If check fails.
        """
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(cls._validation_msg)

    @property
    def density(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).density

    @property
    def fill_value(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).fill_value

    @property
    def npoints(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).npoints

    @property
    def sp_values(self):
        return self._parent._default_to_pandas(pandas.Series.sparse).sp_values

    @classmethod
    def from_coo(cls, A, dense_index=False):
        return cls._default_to_pandas(
            pandas.Series.sparse.from_coo, A, dense_index=dense_index
        )

    def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels=False):
        return self._default_to_pandas(
            pandas.Series.sparse.to_coo,
            row_levels=row_levels,
            column_levels=column_levels,
            sort_labels=sort_labels,
        )

    def to_dense(self):
        return self._default_to_pandas(pandas.Series.sparse.to_dense)


@_inherit_docstrings(pandas.core.accessor.CachedAccessor)
class CachedAccessor(ClassLogger):
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


class ExperimentalFunctions:
    """
    Namespace class for accessing experimental Modin functions.

    Parameters
    ----------
    data : DataFrame or Series
        Object to operate on.
    """

    def __init__(self, data):
        self._data = data

    def to_pickle_distributed(
        self,
        filepath_or_buffer,
        compression: CompressionOptions = "infer",
        protocol: int = pickle.HIGHEST_PROTOCOL,
        storage_options: StorageOptions = None,
    ):
        """
        Pickle (serialize) object to file.

        This experimental feature provides parallel writing into multiple pickle files which are
        defined by glob pattern, otherwise (without glob pattern) default pandas implementation is used.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            File path where the pickled object will be stored.
        compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default: 'infer'
            A string representing the compression to use in the output file. By
            default, infers from the file extension in specified path.
            Compression mode may be any of the following possible
            values: {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}. If compression
            mode is 'infer' and path_or_buf is path-like, then detect
            compression mode from the following extensions:
            '.gz', '.bz2', '.zip' or '.xz'. (otherwise no compression).
            If dict given and mode is 'zip' or inferred as 'zip', other entries
            passed as additional compression options.
        protocol : int, default: pickle.HIGHEST_PROTOCOL
            Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL (see `pickle docs <https://docs.python.org/3/library/pickle.html>`_
            paragraph 12.1.2 for details). The possible  values are 0, 1, 2, 3, 4, 5. A negative value
            for the protocol parameter is equivalent to setting its value to HIGHEST_PROTOCOL.
        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc., if using a URL that will be parsed by
            fsspec, e.g., starting "s3://", "gcs://". An error will be raised if providing
            this argument with a non-fsspec URL. See the fsspec and backend storage
            implementation docs for the set of allowed keys and values.
        """
        from modin.experimental.pandas.io import to_pickle_distributed

        to_pickle_distributed(
            self._data,
            filepath_or_buffer=filepath_or_buffer,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )
