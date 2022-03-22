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

"""Utility function for testing OmniSciOnNative implementation for DataFrame exchange protocol."""

import pandas
import numpy as np
from typing import Dict

from modin.core.dataframe.pandas.exchange.dataframe_protocol.from_dataframe import (
    from_dataframe_to_pandas,
    protocol_df_chunk_to_pandas,
)
from modin.experimental.core.execution.native.implementations.omnisci_on_native.test.utils import (
    ForceOmnisciImport,
)


def split_df_into_chunks(df, n_chunks):
    """
    Split passed DataFrame into `n_chunks` along row axis.

    Parameters
    ----------
    df : DataFrame
        DataFrame to split into chunks.
    n_chunks : int
        Number of chunks to split `df` into.

    Returns
    -------
    list of DataFrames
    """
    chunks = []
    for i in range(n_chunks):
        start = i * len(df) // n_chunks
        end = (i + 1) * len(df) // n_chunks
        chunks.append(df.iloc[start:end])

    return chunks


def export_frame(md_df, from_omnisci=False, **kwargs):
    """
    Construct ``pandas.DataFrame`` from ``modin.pandas.DataFrame`` using DataFrame exchange protocol.

    Parameters
    ----------
    md_df : modin.pandas.DataFrame
        DataFrame to convert to pandas.
    from_omnisci : bool, default: False
        Whether to forcibly use data exported from OmniSci. If `True`, import DataFrame's
        data into OmniSci and then export it back, so the origin for underlying `md_df`
        data is OmniSci.
    **kwargs : dict
        Additional parameters to pass to the ``from_dataframe_to_pandas`` function.

    Returns
    -------
    pandas.DataFrame
    """
    if not from_omnisci:
        return from_dataframe_to_pandas_assert_chunking(md_df, **kwargs)

    with ForceOmnisciImport(md_df) as instance:
        md_df_exported = instance.export_frames()[0]
        exported_df = from_dataframe_to_pandas_assert_chunking(md_df_exported, **kwargs)

    return exported_df


def from_dataframe_to_pandas_assert_chunking(df, n_chunks=None, **kwargs):
    """
    Build a ``pandas.DataFrame`` from a `__dataframe__` object splitting it into `n_chunks`.

    The function asserts that the `df` was split exactly into `n_chunks` before converting them to pandas.

    Parameters
    ----------
    df : DataFrame
        Object supporting the exchange protocol, i.e. `__dataframe__` method.
    n_chunks : int, optional
        Number of chunks to split `df`.

    Returns
    -------
    pandas.DataFrame
    """
    if n_chunks is None:
        return from_dataframe_to_pandas(df, n_chunks=n_chunks, **kwargs)

    protocol_df = df.__dataframe__()
    chunks = list(protocol_df.get_chunks(n_chunks))
    assert len(chunks) == n_chunks

    pd_chunks = [None] * len(chunks)
    for i in range(len(chunks)):
        pd_chunks[i] = protocol_df_chunk_to_pandas(chunks[i], **kwargs)

    pd_df = pandas.concat(pd_chunks, axis=0, ignore_index=True)

    index_obj = protocol_df.metadata.get(
        "modin.index", protocol_df.metadata.get("pandas.index", None)
    )
    if index_obj is not None:
        pd_df.index = index_obj

    return pd_df


def get_data_of_all_types(
    has_nulls=False, exclude_dtypes=None, include_dtypes=None
) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary containing every datatype that is supported by Omnisci implementation of the exchange protocol.

    Parameters
    ----------
    has_nulls : bool, default: False
        Whether to include columns containing null values.
    exclude_dtypes : list, optional
        List of type prefixes to exclude in the dictionary. For example,
        passing ``["int", "float"]`` excludes all of the signed integer (``int16``,
         ``int32``, ``int64``) and float (``float32``, ``float64``) types.
    include_dtypes : list, optional
        List of type prefixes to include in the dictionary. For example,
        passing ``["int", "float"]`` will include ONLY signed integer (``int16``,
         ``int32``, ``int64``) and float (``float32``, ``float64``) types.

    Returns
    -------
    dict
        Dictionary to pass to a DataFrame constructor. The keys are string column names
        that are equal to the type name of the according column. Columns containing null
        types have a ``"_null"`` suffix in their names.
    """
    bool_data = {}
    int_data = {}
    uint_data = {}
    float_data = {}
    datetime_data = {}
    string_data = {}
    category_data = {}

    # bool
    bool_data["bool"] = np.array([True, False, True, True] * 10, dtype=bool)

    # int
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f"int{width}")
        max_val, min_val = np.iinfo(dtype).max, np.iinfo(dtype).min
        int_data[f"int{width}"] = np.array(
            [max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype
        )

    # uint
    for width in (8, 16, 32, 64):
        dtype = getattr(np, f"uint{width}")
        max_val, min_val = np.iinfo(dtype).max, np.iinfo(dtype).min
        uint_data[f"uint{width}"] = np.array(
            [max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype
        )

    # float
    for width in (32, 64):
        dtype = getattr(np, f"float{width}")
        max_val, min_val = np.finfo(dtype).max, np.finfo(dtype).min
        float_data[f"float{width}"] = np.array(
            [max_val, max_val - 1, min_val + 1, min_val + 2] * 10, dtype=dtype
        )
        if has_nulls:
            float_data[f"float{width}_null"] = np.array(
                [max_val, None, min_val + 1, min_val + 2] * 10, dtype=dtype
            )

    # datetime
    for unit in ("s", "ms", "ns"):
        datetime_data[f"datetime64[{unit}]"] = np.array(
            [0, 1, 2, 3] * 10, dtype=np.dtype(f"datetime64[{unit}]")
        )
        if has_nulls:
            datetime_data[f"datetime64[{unit}]_null"] = np.array(
                [0, None, 2, 3] * 10, dtype=np.dtype(f"datetime64[{unit}]")
            )

    # string
    string_data["string"] = np.array(
        # Test multi-byte characters as well to ensure that the chunking works correctly for them
        ["English: test string", " ", "Chinese: 测试字符串", "Russian: тестовая строка"]
        * 10
    )
    if has_nulls:
        string_data["string_null"] = np.array(
            ["English: test string", None, "Chinese: 测试字符串", "Russian: тестовая строка"]
            * 10
        )

    # category
    category_data["category_string"] = pandas.Categorical(
        ["Sample", "te", " ", "xt"] * 10
    )
    # OmniSci does not support non-string categories
    # category_data["category_int"] = pandas.Categorical([1, 2, 3, 4] * 10)
    if has_nulls:
        category_data["category_string_null"] = pandas.Categorical(
            ["Sample", None, " ", "xt"] * 10
        )

    data = {
        **bool_data,
        **int_data,
        **uint_data,
        **float_data,
        **datetime_data,
        **string_data,
        **category_data,
    }

    if include_dtypes is not None:
        filtered_keys = (
            key
            for key in data.keys()
            if any(key.startswith(dtype) for dtype in include_dtypes)
        )
        data = {key: data[key] for key in filtered_keys}

    if exclude_dtypes is not None:
        filtered_keys = (
            key
            for key in data.keys()
            if not any(key.startswith(dtype) for dtype in exclude_dtypes)
        )
        data = {key: data[key] for key in filtered_keys}

    return data
