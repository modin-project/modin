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
from collections import OrderedDict

import pandas as pd
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import pyarrow as pa
from pyarrow.types import is_dictionary

from string import ascii_uppercase, ascii_lowercase, digits
from typing import List, Tuple, Union

import pandas
from pandas import Timestamp
import pyarrow

from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL

IDX_COL_NAME = "__index__"
ROWID_COL_NAME = "__rowid__"

# Bytes 62 and 63 are encoded with 2 characters
_BASE_EXT = ("_A", "_B")
_BASE_LIST = tuple(ascii_uppercase + ascii_lowercase + digits) + _BASE_EXT
_BASE_DICT = dict((c, i) for i, c in enumerate(_BASE_LIST))
_NON_ALPHANUM_PATTERN = re.compile("[^a-zA-Z0-9]+")
# Number of bytes in the tailing chunk
_TAIL_LEN = {"_0": 0, "_1": 1, "_2": 2}
_RESERVED_NAMES = (MODIN_UNNAMED_SERIES_LABEL, ROWID_COL_NAME)
_COL_TYPES = Union[str, int, float, Timestamp, None]
_COL_NAME_TYPE = Union[_COL_TYPES, Tuple[_COL_TYPES, ...]]


def encode_col_name(
    name: _COL_NAME_TYPE,
    ignore_reserved: bool = True,
) -> str:
    """
    Encode column name, using the alphanumeric and underscore characters only.

    The supported name types are specified in the type hints. Non-string names
    are converted to string and prefixed with a corresponding tag. The strings
    are encoded in the following way:
      - All alphanum characters are left as is. I.e., if the column name
        consists from the alphanum characters only, the original name is
        returned.
      - Non-alphanum parts of the name are encoded, using a customized
        version of the base64 algorithm, that allows alphanum characters only.

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
    if name is None:
        return "_N"
    if isinstance(name, int):
        return f"_I{str(name)}"
    if isinstance(name, float):
        return f"_F{str(name)}"
    if isinstance(name, Timestamp):
        return f"_D{encode_col_name((name.timestamp(), str(name.tz)))}"
    if isinstance(name, tuple):
        dst = ["_T"]
        count = len(name)
        for n in name:
            dst.append(encode_col_name(n))
            count -= 1
            if count != 0:
                dst.append("_S")  # Separator
        return "".join(dst)
    if len(name) == 0:
        return "_E"
    if ignore_reserved and (name.startswith(IDX_COL_NAME) or name in _RESERVED_NAMES):
        return name

    non_alpha = _NON_ALPHANUM_PATTERN.search(name)
    if not non_alpha:
        # If the name consists only from alphanum characters, return it as is.
        return name

    dst = []
    off = 0
    while non_alpha:
        start = non_alpha.start()
        end = non_alpha.end()
        dst.append(name[off:start])
        _quote(name[start:end], dst)
        off = end
        non_alpha = _NON_ALPHANUM_PATTERN.search(name, off)
    dst.append(name[off:])
    return "".join(dst)


def decode_col_name(name: str) -> _COL_NAME_TYPE:
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
    if name.startswith("_"):
        if name.startswith(IDX_COL_NAME) or name in _RESERVED_NAMES:
            return name
        char = name[1]
        if char == "N":
            return None
        if char == "I":
            return int(name[2:])
        if char == "F":
            return float(name[2:])
        if char == "D":
            stamp = decode_col_name(name[2:])
            return Timestamp.fromtimestamp(stamp[0], tz=stamp[1])
        if char == "T":
            dst = [decode_col_name(n) for n in name[2:].split("_S")]
            return tuple(dst)
        if char == "E":
            return ""

    idx = name.find("_Q")
    if idx == -1:
        return name

    dst = []
    off = 0
    end = len(name)
    while idx != -1:
        dst.append(name[off:idx])
        off = _unquote(name, dst, idx, end)
        idx = name.find("_Q", off)
    dst.append(name[off:])
    return "".join(dst)


def _quote(src: str, dst: List[str]):  # noqa: GL08
    base = _BASE_LIST
    raw = src.encode()
    rem = len(raw) % 3
    append = dst.append
    append("_Q")

    for bytes3 in zip(*[iter(raw)] * 3):
        i24 = bytes3[0] << 16 | bytes3[1] << 8 | bytes3[2]
        append(base[(i24 >> 18) & 0x3F])
        append(base[(i24 >> 12) & 0x3F])
        append(base[(i24 >> 6) & 0x3F])
        append(base[i24 & 0x3F])
    if rem == 1:
        i24 = raw[-1] << 16
        append(base[(i24 >> 18) & 0x3F])
        append(base[(i24 >> 12) & 0x3F])
        append("_1")
    elif rem == 2:
        i24 = raw[-2] << 16 | raw[-1] << 8
        append(base[(i24 >> 18) & 0x3F])
        append(base[(i24 >> 12) & 0x3F])
        append(base[(i24 >> 6) & 0x3F])
        append("_2")
    else:
        append("_0")


def _unquote(src: str, dst: List[str], off, end) -> int:  # noqa: GL08
    assert src[off : off + 2] == "_Q"
    base = _BASE_DICT
    raw = bytearray()
    append = raw.append
    off += 2

    while off < end:
        chars = src[off : off + 4]
        if "_" not in chars:
            i24 = (
                (base[chars[0]] << 18)
                | (base[chars[1]] << 12)
                | (base[chars[2]] << 6)
                | base[chars[3]]
            )
            append((i24 >> 16) & 0xFF)
            append((i24 >> 8) & 0xFF)
            append(i24 & 0xFF)
            off += 4
        else:
            i24 = 0
            tail_len = 3
            for i in range(0, len(chars)):
                char = src[off]
                off += 1
                if char == "_":
                    off += 1
                    char = src[off - 2 : off]
                    if char in _TAIL_LEN:
                        tail_len = _TAIL_LEN[char]
                        end = off
                        break
                i24 |= base[char] << (6 * (3 - i))

            for i in range(0, tail_len):
                append((i24 >> (8 * (2 - i))) & 0xFF)
    dst.append(raw.decode())
    assert src[off - 2 : off] in _TAIL_LEN
    return off


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
