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

from typing import Tuple, Union
from functools import lru_cache
from collections import OrderedDict

import pandas as pd
from pandas import Timestamp
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pyarrow as pa
from pyarrow.types import is_dictionary

from modin.utils import MODIN_UNNAMED_SERIES_LABEL


class ColNameCodec:
    IDX_COL_NAME = "__index__"
    ROWID_COL_NAME = "__rowid__"

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
    cat = pd.concat([chunk.dictionary.to_pandas() for chunk in chunks])
    return pd.CategoricalDtype(cat.unique())


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
    if pa.types.is_string(t1):
        return t1
    if pa.types.is_string(t2):
        return t2
    if pa.types.is_null(t1):
        return t2
    if pa.types.is_null(t2):
        return t1
    if t1.bit_width > t2.bit_width:
        return t1
    if t1.bit_width < t2.bit_width:
        return t2
    return t2 if pa.types.is_floating(t2) else t1


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
