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

import re

import typing
from typing import Tuple, Union, List, Any
from functools import lru_cache
from collections import OrderedDict

import numpy as np
import pandas
from pandas import Timestamp
from pandas.core.dtypes.common import get_dtype, is_string_dtype
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pyarrow as pa
from pyarrow.types import is_dictionary

from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL

EMPTY_ARROW_TABLE = pa.Table.from_pandas(pandas.DataFrame({}))


class ColNameCodec:
    IDX_COL_NAME = "__index__"
    ROWID_COL_NAME = "__rowid__"
    UNNAMED_IDX_COL_NAME = "__index__0__N"

    _IDX_NAME_PATTERN = re.compile(f"{IDX_COL_NAME}\\d+_(.*)")
    _RESERVED_NAMES = (MODIN_UNNAMED_SERIES_LABEL, ROWID_COL_NAME)
    _COL_TYPES = Union[str, int, float, Timestamp, None]
    _COL_NAME_TYPE = Union[_COL_TYPES, Tuple[_COL_TYPES, ...]]

    def _encode_tuple(values: Tuple[_COL_TYPES, ...]) -> str:  # noqa: GL08
        dst = ["_T"]
        count = len(values)
        for value in values:
            if isinstance(value, str):
                dst.append(value.replace("_", "_Q"))
            else:
                dst.append(ColNameCodec._ENCODERS[type(value)](value))
            count -= 1
            if count != 0:
                dst.append("_T")
        return "".join(dst)

    def _decode_tuple(encoded: str) -> Tuple[_COL_TYPES, ...]:  # noqa: GL08
        items = []
        for item in encoded[2:].split("_T"):
            dec = (
                None
                if len(item) < 2 or item[0] != "_"
                else ColNameCodec._DECODERS.get(item[1], None)
            )
            items.append(item.replace("_Q", "_") if dec is None else dec(item))
        return tuple(items)

    _ENCODERS = {
        tuple: _encode_tuple,
        type(None): lambda v: "_N",
        str: lambda v: "_E" if len(v) == 0 else "_S" + v[1:] if v[0] == "_" else v,
        int: lambda v: f"_I{v}",
        float: lambda v: f"_F{v}",
        Timestamp: lambda v: f"_D{v.timestamp()}_{v.tz}",
    }

    _DECODERS = {
        "T": _decode_tuple,
        "N": lambda v: None,
        "E": lambda v: "",
        "S": lambda v: "_" + v[2:],
        "I": lambda v: int(v[2:]),
        "F": lambda v: float(v[2:]),
        "D": lambda v: Timestamp.fromtimestamp(
            float(v[2 : (idx := v.index("_", 2))]), tz=v[idx + 1 :]
        ),
    }

    @staticmethod
    @lru_cache(1024)
    def encode(
        name: _COL_NAME_TYPE,
        ignore_reserved: bool = True,
    ) -> str:
        """
        Encode column name.

        The supported name types are specified in the type hints. Non-string names
        are converted to string and prefixed with a corresponding tag.

        Parameters
        ----------
        name : str, int, float, Timestamp, None, tuple
            Column name to be encoded.
        ignore_reserved : bool, default: True
            Do not encode reserved names.

        Returns
        -------
        str
            Encoded name.
        """
        if (
            ignore_reserved
            and isinstance(name, str)
            and (
                name.startswith(ColNameCodec.IDX_COL_NAME)
                or name in ColNameCodec._RESERVED_NAMES
            )
        ):
            return name

        try:
            return ColNameCodec._ENCODERS[type(name)](name)
        except KeyError:
            raise TypeError(f"Unsupported column name: {name}")

    @staticmethod
    @lru_cache(1024)
    def decode(name: str) -> _COL_NAME_TYPE:
        """
        Decode column name, previously encoded with encode_col_name().

        Parameters
        ----------
        name : str
            Encoded name.

        Returns
        -------
        str, int, float, Timestamp, None, tuple
            Decoded name.
        """
        if (
            len(name) < 2
            or name[0] != "_"
            or name.startswith(ColNameCodec.IDX_COL_NAME)
            or name in ColNameCodec._RESERVED_NAMES
        ):
            return name

        try:
            return ColNameCodec._DECODERS[name[1]](name)
        except KeyError:
            raise ValueError(f"Invalid encoded column name: {name}")

    @staticmethod
    def mangle_index_names(names: List[_COL_NAME_TYPE]) -> List[str]:
        """
        Return mangled index names for index labels.

        Mangled names are used for index columns because index
        labels cannot always be used as HDK table column
        names. E.e. label can be a non-string value or an
        unallowed string (empty strings, etc.) for a table column
        name.

        Parameters
        ----------
        names : list of str
            Index labels.

        Returns
        -------
        list of str
            Mangled names.
        """
        pref = ColNameCodec.IDX_COL_NAME
        return [f"{pref}{i}_{ColNameCodec.encode(n)}" for i, n in enumerate(names)]

    @staticmethod
    def demangle_index_names(
        cols: List[str],
    ) -> Union[_COL_NAME_TYPE, List[_COL_NAME_TYPE]]:
        """
        Demangle index column names to index labels.

        Parameters
        ----------
        cols : list of str
            Index column names.

        Returns
        -------
        list or a single demangled name
            Demangled index names.
        """
        if len(cols) == 1:
            return ColNameCodec.demangle_index_name(cols[0])
        return [ColNameCodec.demangle_index_name(n) for n in cols]

    @staticmethod
    def demangle_index_name(col: str) -> _COL_NAME_TYPE:
        """
        Demangle index column name into index label.

        Parameters
        ----------
        col : str
            Index column name.

        Returns
        -------
        str
            Demangled index name.
        """
        match = ColNameCodec._IDX_NAME_PATTERN.search(col)
        if match:
            name = match.group(1)
            if name == MODIN_UNNAMED_SERIES_LABEL:
                return None
            return ColNameCodec.decode(name)
        return col

    @staticmethod
    def concat_index_names(frames) -> typing.OrderedDict[str, Any]:
        """
        Calculate the index names and dtypes.

        Calculate the index names and dtypes, that the index
        columns will have after the frames concatenation.

        Parameters
        ----------
        frames : list[HdkOnNativeDataframe]

        Returns
        -------
        typing.OrderedDict[str, Any]
        """
        first = frames[0]
        names = OrderedDict()
        if first._index_width() > 1:
            # When we're dealing with a MultiIndex case the resulting index
            # inherits the levels from the first frame in concatenation.
            dtypes = first._dtypes
            for n in first._index_cols:
                names[n] = dtypes[n]
        else:
            # In a non-MultiIndex case, we check if all the indices have the same
            # names, and if they do - inherit the name and dtype from the first frame,
            # otherwise return metadata matching unnamed RangeIndex.
            mangle = ColNameCodec.mangle_index_names
            idx_names = set()
            for f in frames:
                if f._index_cols is not None:
                    idx_names.update(f._index_cols)
                elif f.has_index_cache:
                    idx_names.update(mangle(f.index.names))
                else:
                    idx_names.add(ColNameCodec.UNNAMED_IDX_COL_NAME)
                if len(idx_names) > 1:
                    idx_names = [ColNameCodec.UNNAMED_IDX_COL_NAME]
                    break

            name = next(iter(idx_names))
            # Inherit the Index's dtype from the first frame.
            if first._index_cols is not None:
                names[name] = first._dtypes.iloc[0]
            elif first.has_index_cache:
                names[name] = first.index.dtype
            else:
                # A trivial index with no name
                names[name] = get_dtype(int)
        return names


def build_categorical_from_at(table, column_name):
    """
    Build ``pandas.CategoricalDtype`` from a dictionary column of the passed PyArrow Table.

    Parameters
    ----------
    table : pyarrow.Table
    column_name : str

    Returns
    -------
    pandas.CategoricalDtype
    """
    chunks = table.column(column_name).chunks
    cat = pandas.concat([chunk.dictionary.to_pandas() for chunk in chunks])
    return pandas.CategoricalDtype(cat.unique())


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
        raise NotImplementedError(f"{join_type} join")


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
                if col == ColNameCodec.demangle_index_name(c):
                    new_name = c
                    break
        elif df.has_index_cache:
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
        idx = df._index_cache.get() if df.has_index_cache else None
        if idx is not None:
            idx = idx[:1]
        elif df._index_cols is not None:
            if len(df._index_cols) > 1:
                arrays = [[i] for i in range(len(df._index_cols))]
                names = [ColNameCodec.demangle_index_name(n) for n in df._index_cols]
                idx = pandas.MultiIndex.from_arrays(arrays, names=names)
            else:
                idx = pandas.Index(
                    name=ColNameCodec.demangle_index_name(df._index_cols[0])
                )
        return pandas.DataFrame(columns=df.columns, index=idx)

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
        index_cols = ColNameCodec.mangle_index_names(merged.index.names)
        for name in index_cols:
            # Using _dtypes here since it contains all column names,
            # including the index.
            df = left if name in left._dtypes else right
            exprs[name] = df.ref(name)
            new_dtypes.append(df._dtypes[name])

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


def maybe_range(numbers: Union[List[int], range]) -> Union[List[int], range]:
    """
    Try to convert the specified sequence of numbers to a range.

    Parameters
    ----------
    numbers : list of ints or range

    Returns
    -------
    list of ints or range
    """
    if len(numbers) > 2 and not is_range_like(numbers):
        diff = numbers[1] - numbers[0]
        is_range = True
        for i in range(2, len(numbers)):
            if (numbers[i] - numbers[i - 1]) != diff:
                is_range = False
                break
        if is_range:
            numbers = range(numbers[0], numbers[-1] + diff, diff)
    return numbers


def to_arrow_type(dtype) -> pa.lib.DataType:
    """
    Convert the specified dtype to arrow.

    Parameters
    ----------
    dtype : dtype

    Returns
    -------
    pa.lib.DataType
    """
    if is_string_dtype(dtype):
        return pa.from_numpy_dtype(str)
    return pa.from_numpy_dtype(dtype)


def get_common_arrow_type(t1: pa.lib.DataType, t2: pa.lib.DataType) -> pa.lib.DataType:
    """
    Get common arrow data type.

    Parameters
    ----------
    t1 : pa.lib.DataType
    t2 : pa.lib.DataType

    Returns
    -------
    pa.lib.DataType
    """
    if t1 == t2:
        return t1
    if pa.types.is_string(t1):
        return t1
    if pa.types.is_string(t2):
        return t2
    if pa.types.is_null(t1):
        return t2
    if pa.types.is_null(t2):
        return t1

    t1 = t1.to_pandas_dtype()
    t2 = t2.to_pandas_dtype()
    return pa.from_numpy_dtype(np.promote_types(t1, t2))


def arrow_to_pandas(at: pa.Table) -> pandas.DataFrame:
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

    df = at.to_pandas(types_mapper=mapper)
    dtype = {}
    for idx, _type in enumerate(at.schema.types):
        if isinstance(_type, pa.lib.TimestampType) and _type.unit != "ns":
            dtype[at.schema.names[idx]] = f"datetime64[{_type.unit}]"
    if dtype:
        # TODO: remove after https://github.com/apache/arrow/pull/35656 is merge
        df = df.astype(dtype)
    return df


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
        return pandas.Categorical(
            pandas.concat(values, ignore_index=True),
            dtype=pandas.CategoricalDtype(categories, ordered=True),
        )
