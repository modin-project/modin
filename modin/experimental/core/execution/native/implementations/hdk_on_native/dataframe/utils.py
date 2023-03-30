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

"""Utilities for internal use by the ``HdkOnNativeDataframe``."""

from collections import OrderedDict

import pandas as pd
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pyarrow as pa
from pyarrow.types import is_dictionary

from modin.error_message import ErrorMessage


class LazyProxyCategoricalDtype(pd.CategoricalDtype):
    """
    Proxy class for lazily retrieving categorical dtypes from arrow tables.

    Parameters
    ----------
    table : pyarrow.Table
        Source table.
    column_name : str
        Column name.
    """

    def __init__(self, table: pa.Table, column_name: str):
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=table is None,
            extra_log="attempted to bind 'None' pyarrow table to a lazy category",
        )
        self._table = table
        self._column_name = column_name
        self._ordered = False
        self._lazy_categories = None

    def _new(self, table: pa.Table, column_name: str) -> pd.CategoricalDtype:
        """
        Create a new proxy, if either table or column name are different.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.
        column_name : str
            Column name.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._table is None:
            # The table has been materialized, we don't need a proxy anymore.
            return pd.CategoricalDtype(self.categories)
        elif table is self._table and column_name == self._column_name:
            return self
        else:
            return LazyProxyCategoricalDtype(table, column_name)

    @property
    def _categories(self):  # noqa: GL08
        if self._table is not None:
            chunks = self._table.column(self._column_name).chunks
            cat = pd.concat([chunk.dictionary.to_pandas() for chunk in chunks])
            self._lazy_categories = self.validate_categories(cat.unique())
            self._table = None  # The table is not required any more
        return self._lazy_categories

    @_categories.setter
    def _set_categories(self, categories):  # noqa: GL08
        self._lazy_categories = categories
        self._table = None


def check_join_supported(join_type: str):
    """
    Check if join type is supported by HDK.

    Parameters
    ----------
    join_type : str
        Join type.

    Returns
    -------
    None
    """
    if join_type not in ("inner", "left"):
        raise NotImplementedError(
            f"{join_type} join is not supported by the HDK engine"
        )


def check_cols_to_join(what, df, col_names):
    """
    Check the frame columns.

    Check if the frame (`df`) has the specified columns (`col_names`). The names referring to
    the index columns are replaced with the actual index column names.

    Parameters
    ----------
    what : str
        Attribute name.
    df : HdkOnNativeDataframe
        The dataframe.
    col_names : list of str
        The column names to check.

    Returns
    -------
    Tuple[HdkOnNativeDataframe, list]
        The aligned data frame and column names.
    """
    cols = df.columns
    new_col_names = col_names
    for i, col in enumerate(col_names):
        if col in cols:
            continue
        new_name = None
        if df._index_cols is not None:
            for c in df._index_cols:
                if col == df._index_name(c):
                    new_name = c
                    break
        elif df._index_cache is not None:
            new_name = f"__index__{0}_{col}"
            df = df._maybe_materialize_rowid()
        if new_name is None:
            raise ValueError(f"'{what}' references unknown column {col}")
        if new_col_names is col_names:
            # We are replacing the index names in the original list,
            # but creating a copy.
            new_col_names = col_names.copy()
        new_col_names[i] = new_name
    return df, new_col_names


def get_data_for_join_by_index(
    left,
    right,
    how,
    left_on,
    right_on,
    sort,
    suffixes,
):
    """
    Return the column names, dtypes and expres, required for join by index.

    This is a helper function, used by `HdkOnNativeDataframe.join()`, when joining by index.

    Parameters
    ----------
    left : HdkOnNativeDataframe
        A frame to join.
    right : HdkOnNativeDataframe
        A frame to join with.
    how : str
        A type of join.
    left_on : list of str
        A list of columns for the left frame to join on.
    right_on : list of str
        A list of columns for the right frame to join on.
    sort : bool
        Sort the result by join keys.
    suffixes : list-like of str
        A length-2 sequence of suffixes to add to overlapping column names
        of left and right operands respectively.

    Returns
    -------
    tuple

    The index columns, exprs, dtypes and columns.
    """

    def to_empty_pandas_df(df):
        # Create an empty pandas frame with the same columns and index.
        idx = df._index_cache
        if idx is not None:
            idx = idx[:1]
        elif df._index_cols is not None:
            if len(df._index_cols) > 1:
                arrays = [[i] for i in range(len(df._index_cols))]
                names = [df._index_name(n) for n in df._index_cols]
                idx = pd.MultiIndex.from_arrays(arrays, names=names)
            else:
                idx = pd.Index(name=df._index_name(df._index_cols[0]))
        return pd.DataFrame(columns=df.columns, index=idx)

    new_dtypes = []
    exprs = OrderedDict()
    merged = to_empty_pandas_df(left).merge(
        to_empty_pandas_df(right),
        how=how,
        left_on=left_on,
        right_on=right_on,
        sort=sort,
        suffixes=suffixes,
    )

    if len(merged.index.names) == 1 and (merged.index.names[0] is None):
        index_cols = None
    else:
        index_cols = left._mangle_index_names(merged.index.names)
        for orig_name, mangled_name in zip(merged.index.names, index_cols):
            # Using _dtypes here since it contains all column names,
            # including the index.
            df = left if mangled_name in left._dtypes else right
            exprs[orig_name] = df.ref(mangled_name)
            new_dtypes.append(df._dtypes[mangled_name])

    left_col_names = set(left.columns)
    right_col_names = set(right.columns)
    for col in merged.columns:
        orig_name = col
        if orig_name in left_col_names:
            df = left
        elif orig_name in right_col_names:
            df = right
        elif suffixes is None:
            raise ValueError(f"Unknown column {col}")
        elif (
            col.endswith(suffixes[0])
            and (orig_name := col[0 : -len(suffixes[0])]) in left_col_names
            and orig_name in right_col_names
        ):
            df = left  # Overlapping column from the left frame
        elif (
            col.endswith(suffixes[1])
            and (orig_name := col[0 : -len(suffixes[1])]) in right_col_names
            and orig_name in left_col_names
        ):
            df = right  # Overlapping column from the right frame
        else:
            raise ValueError(f"Unknown column {col}")
        exprs[col] = df.ref(orig_name)
        new_dtypes.append(df._dtypes[orig_name])

    return index_cols, exprs, new_dtypes, merged.columns


def arrow_to_pandas(at: pa.Table) -> pd.DataFrame:
    """
    Convert the specified arrow table to pandas.

    Parameters
    ----------
    at : pyarrow.Table
        The table to convert.

    Returns
    -------
    pandas.DataFrame
    """

    def mapper(at):
        if is_dictionary(at) and isinstance(at.value_type, ArrowIntervalType):
            # The default mapper fails with TypeError: unhashable type: 'dict'
            return _CategoricalDtypeMapper
        return None

    return at.to_pandas(types_mapper=mapper)


class _CategoricalDtypeMapper:  # noqa: GL08
    @staticmethod
    def __from_arrow__(arr):  # noqa: GL08
        values = []
        # Using OrderedDict as an ordered set to preserve the categories order
        categories = OrderedDict()
        chunks = arr.chunks if isinstance(arr, pa.ChunkedArray) else (arr,)
        for chunk in chunks:
            assert isinstance(chunk, pa.DictionaryArray)
            cat = chunk.dictionary.to_pandas()
            values.append(chunk.indices.to_pandas().map(cat))
            categories.update((c, None) for c in cat)
        return pd.Categorical(
            pd.concat(values, ignore_index=True),
            dtype=pd.CategoricalDtype(categories, ordered=True),
        )
