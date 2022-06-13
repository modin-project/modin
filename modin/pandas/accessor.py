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

import pandas
from pandas.core.arrays.sparse.dtype import SparseDtype

import modin.pandas as pd
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
from modin.logging import LoggerMetaClass, metaclass_resolver


class BaseSparseAccessor(object, metaclass=LoggerMetaClass):
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

    def _validate(self, data):
        """
        Verify that `data` dtypes are compatible with `pandas.core.arrays.sparse.dtype.SparseDtype`.

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
class SparseFrameAccessor(metaclass_resolver(BaseSparseAccessor)):
    def _validate(self, data):
        """
        Verify that `data` dtypes are compatible with `pandas.core.arrays.sparse.dtype.SparseDtype`.

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
            raise AttributeError(self._validation_msg)

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
class SparseAccessor(metaclass_resolver(BaseSparseAccessor)):
    def _validate(self, data):
        """
        Verify that `data` dtype is compatible with `pandas.core.arrays.sparse.dtype.SparseDtype`.

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
            raise AttributeError(self._validation_msg)

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
class CachedAccessor(object, metaclass=LoggerMetaClass):
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj
