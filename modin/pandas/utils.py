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


def from_non_pandas(df, index, columns, dtype):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME : PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    from modin.data_management.factories.dispatcher import EngineDispatcher

    new_qc = EngineDispatcher.from_non_pandas(df, index, columns, dtype)
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
        A new Modin DataFrame object.
    """
    from modin.data_management.factories.dispatcher import EngineDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=EngineDispatcher.from_pandas(df))


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
    from modin.data_management.factories.dispatcher import EngineDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=EngineDispatcher.from_arrow(at))


def is_scalar(obj):
    """
    Return True if given object is scalar.

    This method wrks the same as is_scalar method from Pandas but
    it is optimized for Modin frames. For BasePandasDataset objects
    Pandas version of is_scalar tries to access missing attribute
    causing index scan. This tiggers execution for lazy frames and
    we avoid it by handling BasePandasDataset objects separately.

    Parameters
    ----------
    val : object
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
    sortorder : int, optional
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


_original_pandas_MultiIndex_from_frame = MultiIndex.from_frame
MultiIndex.from_frame = from_modin_frame_to_mi
