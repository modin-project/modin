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

"""Implement utils for pandas component."""

from pandas import MultiIndex
from modin.utils import hashable


def from_non_pandas(df, index, columns, dtype):
    """
    Convert a non-pandas DataFrame into Modin DataFrame.

    Parameters
    ----------
    df : object
        Non-pandas DataFrame.
    index : object
        Index for non-pandas DataFrame.
    columns : object
        Columns for non-pandas DataFrame.
    dtype : type
        Data type to force.

    Returns
    -------
    modin.pandas.DataFrame
        Converted DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    new_qc = FactoryDispatcher.from_non_pandas(df, index, columns, dtype)
    if new_qc is not None:
        from .dataframe import DataFrame

        return DataFrame(query_compiler=new_qc)
    return new_qc


def from_pandas(df):
    """
    Convert a pandas DataFrame to a Modin DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas DataFrame to convert.

    Returns
    -------
    modin.pandas.DataFrame
        A new Modin DataFrame object.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=FactoryDispatcher.from_pandas(df))


def from_arrow(at):
    """
    Convert an Arrow Table to a Modin DataFrame.

    Parameters
    ----------
    at : Arrow Table
        The Arrow Table to convert from.

    Returns
    -------
    DataFrame
        A new Modin DataFrame object.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=FactoryDispatcher.from_arrow(at))


def is_scalar(obj):
    """
    Return True if given object is scalar.

    This method works the same as is_scalar method from pandas but
    it is optimized for Modin frames. For BasePandasDataset objects
    pandas version of is_scalar tries to access missing attribute
    causing index scan. This triggers execution for lazy frames and
    we avoid it by handling BasePandasDataset objects separately.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if given object is scalar and False otherwise.
    """
    from pandas.api.types import is_scalar as pandas_is_scalar
    from .base import BasePandasDataset

    return not isinstance(obj, BasePandasDataset) and pandas_is_scalar(obj)


def from_modin_frame_to_mi(df, sortorder=None, names=None):
    """
    Make a pandas.MultiIndex from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted to pandas.MultiIndex.
    sortorder : int, default: None
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : list-like, optional
        If no names are provided, use the column names, or tuple of column
        names if the columns is a MultiIndex. If a sequence, overwrite
        names with the given sequence.

    Returns
    -------
    pandas.MultiIndex
        The pandas.MultiIndex representation of the given DataFrame.
    """
    from .dataframe import DataFrame

    if isinstance(df, DataFrame):
        from modin.error_message import ErrorMessage

        ErrorMessage.default_to_pandas("`MultiIndex.from_frame`")
        df = df._to_pandas()
    return _original_pandas_MultiIndex_from_frame(df, sortorder, names)


def is_label(obj, label, axis=0):
    """
    Check whether or not 'obj' contain column or index level with name 'label'.

    Parameters
    ----------
    obj : modin.pandas.DataFrame, modin.pandas.Series or modin.core.storage_formats.base.BaseQueryCompiler
        Object to check.
    label : object
        Label name to check.
    axis : {0, 1}, default: 0
        Axis to search for `label` along.

    Returns
    -------
    bool
        True if check is successful, False otherwise.
    """
    qc = getattr(obj, "_query_compiler", obj)
    return hashable(label) and (
        label in qc.get_axis(axis ^ 1) or label in qc.get_index_names(axis)
    )


_original_pandas_MultiIndex_from_frame = MultiIndex.from_frame
MultiIndex.from_frame = from_modin_frame_to_mi
