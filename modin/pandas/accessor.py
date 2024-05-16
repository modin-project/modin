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

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Union

import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype

from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.io import to_dask, to_ray
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.pandas import DataFrame, Series


class BaseSparseAccessor(ClassLogger):
    """
    Base class for various sparse DataFrame accessor classes.

    Parameters
    ----------
    data : DataFrame or Series
        Object to operate on.
    """

    _parent: Union[DataFrame, Series]
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data: Union[DataFrame, Series] = None):
        self._parent = data
        self._validate(data)

    @classmethod
    def _validate(cls, data: Union[DataFrame, Series]):
        """
        Verify that `data` dtypes are compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : DataFrame or Series
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
    def _validate(cls, data: DataFrame):
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
    def _validate(cls, data: Series):
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

    def __get__(self, obj, cls):  # noqa: GL08
        if obj is None:
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


class ModinAPI:
    """
    Namespace class for accessing additional Modin functions that are not available in pandas.

    Parameters
    ----------
    data : DataFrame or Series
        Object to operate on.
    """

    _data: Union[DataFrame, Series]

    def __init__(self, data: Union[DataFrame, Series]):
        self._data = data

    def to_pandas(self):
        """
        Convert a Modin DataFrame/Series object to a pandas DataFrame/Series object.

        Returns
        -------
        pandas.Series or pandas.DataFrame
        """
        return self._data._to_pandas()

    def to_ray(self):
        """
        Convert a Modin DataFrame/Series to a Ray Dataset.

        Returns
        -------
        ray.data.Dataset
            Converted object with type depending on input.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Ray Dataset if Modin uses a Ray engine.
        """
        return to_ray(self._data)

    def to_dask(self):
        """
        Convert a Modin DataFrame/Series to a Dask DataFrame/Series.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            Converted object with type depending on input.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Dask DataFrame/Series if Modin uses a Dask engine.
        """
        return to_dask(self._data)

    def to_pickle_glob(
        self,
        filepath_or_buffer,
        compression: CompressionOptions = "infer",
        protocol: int = pickle.HIGHEST_PROTOCOL,
        storage_options: StorageOptions = None,
    ) -> None:
        """
        Pickle (serialize) object to file.

        This experimental feature provides parallel writing into multiple pickle files which are
        defined by glob pattern, otherwise (without glob pattern) default pandas implementation is used.

        Parameters
        ----------
        filepath_or_buffer : str
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
        from modin.experimental.pandas.io import to_pickle_glob

        to_pickle_glob(
            self._data,
            filepath_or_buffer=filepath_or_buffer,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )

    def to_parquet_glob(
        self,
        path,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions = None,
        **kwargs,
    ) -> None:  # noqa: PR01
        """
        Write a DataFrame to the binary parquet format.

        This experimental feature provides parallel writing into multiple parquet files which are
        defined by glob pattern, otherwise (without glob pattern) default pandas implementation is used.

        Notes
        -----
        * Only string type supported for `path` argument.
        * The rest of the arguments are the same as for `pandas.to_parquet`.
        """
        from modin.experimental.pandas.io import to_parquet_glob

        if path is None:
            raise NotImplementedError(
                "`to_parquet_glob` doesn't support path=None, use `to_parquet` in that case."
            )

        to_parquet_glob(
            self._data,
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            storage_options=storage_options,
            **kwargs,
        )

    def to_json_glob(
        self,
        path_or_buf=None,
        orient=None,
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=False,
        compression="infer",
        index=None,
        indent=None,
        storage_options: StorageOptions = None,
        mode="w",
    ) -> None:  # noqa: PR01
        """
        Convert the object to a JSON string.

        Notes
        -----
        * Only string type supported for `path_or_buf` argument.
        * The rest of the arguments are the same as for `pandas.to_json`.
        """
        from modin.experimental.pandas.io import to_json_glob

        if path_or_buf is None:
            raise NotImplementedError(
                "`to_json_glob` doesn't support path_or_buf=None, use `to_json` in that case."
            )

        to_json_glob(
            self._data,
            path_or_buf=path_or_buf,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
            indent=indent,
            storage_options=storage_options,
            mode=mode,
        )

    def to_xml_glob(
        self,
        path_or_buffer=None,
        index=True,
        root_name="data",
        row_name="row",
        na_rep=None,
        attr_cols=None,
        elem_cols=None,
        namespaces=None,
        prefix=None,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
        parser="lxml",
        stylesheet=None,
        compression="infer",
        storage_options=None,
    ) -> None:  # noqa: PR01
        """
        Render a DataFrame to an XML document.

        Notes
        -----
        * Only string type supported for `path_or_buffer` argument.
        * The rest of the arguments are the same as for `pandas.to_xml`.
        """
        from modin.experimental.pandas.io import to_xml_glob

        if path_or_buffer is None:
            raise NotImplementedError(
                "`to_xml_glob` doesn't support path_or_buffer=None, use `to_xml` in that case."
            )

        to_xml_glob(
            self._data,
            path_or_buffer=path_or_buffer,
            index=index,
            root_name=root_name,
            row_name=row_name,
            na_rep=na_rep,
            attr_cols=attr_cols,
            elem_cols=elem_cols,
            namespaces=namespaces,
            prefix=prefix,
            encoding=encoding,
            xml_declaration=xml_declaration,
            pretty_print=pretty_print,
            parser=parser,
            stylesheet=stylesheet,
            compression=compression,
            storage_options=storage_options,
        )
