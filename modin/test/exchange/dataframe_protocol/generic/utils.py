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

"""Utility assertion functions for Dataframe exchange protocol tests that are common for every execution backend."""

from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    DTypeKind,
    pandas_dtype_to_arrow_c,
    ColumnNullType,
)
from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
from modin.pandas.utils import from_dataframe
from modin.pandas.test.utils import (
    df_equals,
    assert_index_equal,
)


def assert_protocol_buffer_equal(array, buffer_and_dtype):
    """
    Test APIs of the Modin protocol buffer.

    The order of APIs to be tested matches the original one.

    Parameters
    ----------
    array : np.ndarray
        NumPy array of the Modin buffer data.
    buffer_and_dtype : tuple
        Two-element tuple whose first element is a buffer containing the data
        and whose second element is the data buffer's associated dtype.
    """
    buf, dtype = buffer_and_dtype
    assert buf.bufsize == array.dtype.itemsize
    assert buf.ptr == array.__array_interface__["data"][0]
    assert dtype == array.dtype
    # TODO: assert dlpack and dlpack.device


def check_column_dtype(modin_col, modin_colx):
    """
    Test `col.dtype`.

    Parameters
    ----------
    modin_col : modin.pandas.Series
        Modin series object representing a dataframe's column.
    modin_colx : ProtocolColumn
        Modin protocol column object.
    """

    modin_col_dtype = modin_col.dtype[0]
    modin_colx_dtype = modin_colx.dtype[0]
    assert modin_colx_dtype[0] == modin_col_dtype  # dtype kind
    if modin_colx_dtype[0] in (
        DTypeKind.INT,
        DTypeKind.UINT,
        DTypeKind.FLOAT,
        DTypeKind.BOOL,
    ):
        assert modin_colx_dtype[1] == modin_col_dtype.itemsize * 8  # bitmask
        assert modin_colx_dtype[2] == pandas_dtype_to_arrow_c(
            modin_col_dtype
        )  # Arrow C format string
        assert modin_colx_dtype[3] == modin_col_dtype.byteorder  # byteorder
    # TODO: Handle all the protocol dtypes
    # else:
    #     pass


def check_column_describe_categorical(modin_col, modin_colx):
    """
    Test `col.describe_categorical()`.

    Parameters
    ----------
    modin_col : modin.pandas.Series
        Modin series object representing a dataframe's column.
    modin_colx : ProtocolColumn
        Modin protocol column object.
    """

    # TODO: TBD
    pass


def check_column_describe_null(modin_col, modin_colx):
    """
    Test `col.describe_null()`.

    Parameters
    ----------
    modin_col : modin.pandas.Series
        Modin series object representing a dataframe's column.
    modin_colx : ProtocolColumn
        Modin protocol column object.
    """

    modin_colx_dtype = modin_colx.dtype[0]
    if modin_colx_dtype in (
        DTypeKind.INT,
        DTypeKind.UINT,
        DTypeKind.FLOAT,
        DTypeKind.BOOL,
        DTypeKind.DATETIME,
    ):
        null, value = modin_colx.describe_null
        assert null in (ColumnNullType.USE_NAN, ColumnNullType.NON_NULLABLE)
        assert value is None
    # TODO: Handle other cases
    # else:
    #     pass


def check_column_get_buffers(modin_col, modin_colx):
    """
    Test `col.get_buffers()`.

    Parameters
    ----------
    modin_col : modin.pandas.Series
        Modin series object representing a dataframe's column.
    modin_colx : ProtocolColumn
        Modin protocol column object.
    """

    if modin_colx.null_count == 0:
        assert modin_colx.get_buffers()["validity"] is None
    else:
        assert_protocol_buffer_equal(
            # get mask for Modin as the first parameter,
            modin_colx.get_buffers()["validity"],
        )

    if modin_colx.dtype[0] == DTypeKind.CATEGORICAL:
        # We do modin_col.to_pandas() to get codes because
        # Modin doesn't yet support the flow https://github.com/modin-project/modin/issues/4187
        assert_protocol_buffer_equal(
            modin_colx.get_buffers()["data"], modin_col.to_pandas().values.codes
        )
        assert modin_colx.get_buffers()["offsets"] is None

    elif modin_colx.dtype[0] == DTypeKind.STRING:
        assert_protocol_buffer_equal(
            # get mask for Modin as the first parameter,
            modin_colx.get_buffers()["data"],
        )
        assert_protocol_buffer_equal(
            # get mask for Modin as the first parameter,
            modin_colx.get_buffers()["offsets"],
        )

    else:
        assert_protocol_buffer_equal(modin_col.values, modin_colx.get_buffers()["data"])
        assert modin_colx.get_buffers()["offsets"] is None


def assert_protocol_column_equal(modin_col, modin_colx):
    """
    Test APIs of the Modin protocol column.

    The order of APIs to be tested matches the original one.

    Parameters
    ----------
    modin_col : modin.pandas.Series
        Modin series object representing a dataframe's column.
    modin_colx : ProtocolColumn
        Modin protocol column object.
    """

    assert modin_colx.size == len(modin_col)
    # TODO: Check offset for ``OmnisciOnNative`` execution
    assert modin_colx.offset == 0
    check_column_dtype(modin_col, modin_colx)
    assert modin_colx.num_chunks() == len(
        modin_col._query_compiler._modin_frame._partitions[0]
    )
    check_column_describe_categorical(modin_col, modin_colx)
    check_column_describe_null(modin_col, modin_colx)
    assert modin_colx.null_count == modin_col.isna().sum()
    for k, v in modin_colx.metadata():
        assert k == "modin.index"
        assert assert_index_equal(v, modin_col.index)
    assert modin_colx.num_chunks() == len(
        modin_col._query_compiler._modin_frame._partitions[0]
    )
    if modin_colx.num_chunks() > 1:
        partitions = unwrap_partitions(modin_col, axis=0)
        for idx, chunk in enumerate(modin_colx.get_chunks()):
            tmp_modin_df = from_partitions(partitions[idx], axis=0)
            df_equals(tmp_modin_df, from_dataframe(chunk))
    elif modin_colx.num_chunks() == 1:
        chunk = modin_colx.get_chunks()
        df_equals(modin_col, from_dataframe(chunk))
    else:
        raise RuntimeError(f"Invalid number of chunks: {modin_colx.num_chunks()}")

    check_column_get_buffers(modin_col, modin_colx)


def assert_protocol_dataframe_equal(modin_df, modin_dfx):
    """
    Test APIs of the Modin protocol dataframe.

    The order of APIs to be tested matches the original one.

    Parameters
    ----------
    modin_df : modin.pandas.DataFrame
        Modin dataframe object.
    modin_dfx : ProtocolDataframe
        Modin protocol dataframe object.
    """

    for k, v in modin_dfx.metadata().items():
        assert k == "modin.index"
        assert assert_index_equal(v, modin_df.index)
    assert modin_dfx.num_columns() == len(modin_df.columns)
    assert modin_dfx.num_rows() == len(modin_df)
    assert modin_dfx.num_chunks() == len(
        modin_df._query_compiler._modin_frame._partitions[0]
    )
    assert modin_dfx.column_names() == list(modin_df.columns)
    for col_idx, col_name in enumerate(modin_dfx.column_names()):
        assert_protocol_column_equal(
            modin_df.loc[:, col_name], modin_dfx.get_column_by_name(col_name)
        )
        assert_protocol_column_equal(
            modin_df.iloc[:, col_idx], modin_dfx.get_column(col_idx)
        )
    for idx, colx in enumerate(modin_dfx.get_columns()):
        assert_protocol_column_equal(modin_df.iloc[:, idx], colx)
    col_sequence = list(range(modin_dfx.num_columns() / 2))
    modin_dfx2 = modin_dfx.select_columns(col_sequence)
    modin_df2 = modin_df.iloc[:, col_sequence]
    df_equals(modin_df2, from_dataframe(modin_dfx2))
    column_names = modin_dfx.column_names()[: len(col_sequence)]
    modin_dfx3 = modin_dfx.select_columns_by_name(column_names)
    modin_df3 = modin_df.loc[:, column_names]
    df_equals(modin_df3, from_dataframe(modin_dfx3))
    if modin_dfx.num_chunks() > 1:
        partitions = unwrap_partitions(modin_df, axis=0)
        for idx, chunk in enumerate(modin_dfx.get_chunks()):
            tmp_modin_df = from_partitions(partitions[idx], axis=0)
            df_equals(tmp_modin_df, from_dataframe(chunk))
    elif modin_dfx.num_chunks() == 1:
        chunk = modin_dfx.get_chunks()
        df_equals(modin_df, from_dataframe(chunk))
    else:
        raise RuntimeError(f"Invalid number of chunks: {modin_dfx.num_chunks()}")
