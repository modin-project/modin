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

"""Dataframe exchange protocol tests that are specific for pandas storage format implementation."""


import pytest
import numpy as np

import modin.pandas as pd
from modin.config import NPartitions
from modin.core.dataframe.pandas.exchange.dataframe_protocol.exception import (
    NoValidityBuffer,
)
from modin.core.dataframe.pandas.exchange.dataframe_protocol.from_dataframe import (
    from_dataframe_to_pandas,
    primitive_column_to_ndarray,
    protocol_df_chunk_to_pandas,
    set_nulls,
)
from modin.pandas.test.utils import df_equals, test_data, test_data_categorical
from modin.pandas.utils import from_dataframe
from modin.test.test_utils import warns_that_defaulting_to_pandas


def test_categorical_get_data_buffer():
    arr = [0, 1, -1]
    df = pd.DataFrame({"a": arr})
    df["a"] = df["a"].astype("category")
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    dataBufX = colX._get_data_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0


@pytest.mark.skip(reason="NotImplementedError")
def test_column_categorical_buffer():
    arr = [0, 1, -1]
    df = pd.DataFrame({"a": arr})
    df["a"] = df["a"].astype("category")
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    bufX = colX.get_buffers()

    dataBuf, dataDtype = bufX["data"]
    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    assert dataDtype[0] == 21


def test_column_string_buffer():
    arr = ["a", "b", "c"]
    df = pd.DataFrame({"a": arr})
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    bufX = colX.get_buffers()

    dataBuf, dataDtype = bufX["data"]
    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    assert dataDtype[0] == 21

    dataBuf, dataDtype = bufX["validity"]
    assert dataBuf.bufsize > 0
    assert dataBuf.ptr != 0
    device, _ = dataBuf.__dlpack_device__()

    assert dataDtype[0] == 1

    dataBuf.__repr__()

    dataBufX = colX._get_data_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0

    dataBufX = colX._get_validity_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0

    dataBufX = colX._get_offsets_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0


@pytest.mark.skip(reason="doesn't correct work dtype for exchange")
def test_describe_null():
    modin_df = pd.DataFrame({"x": [0, 0.5, 1.3, "a"]})
    dfX = modin_df.__dataframe__()
    colX = dfX.get_column(0)
    with pytest.raises(NotImplementedError):
        colX.describe_null


@pytest.mark.skip(reason="missing from interface")
def test_dlpack():
    modin_df = pd.DataFrame(test_data["int_data"])
    dfX = modin_df.__dataframe__()
    colX = dfX.get_column(0)
    dataBufX = colX._get_data_buffer()
    """
    TODO: deleted `with` if we will be supported __dlpack__
    """
    with pytest.raises(NotImplementedError):
        dataBufX.__dlpack__


@pytest.mark.skip(reason="problem with types of protocol")
def test_dtype():
    modin_df = pd.DataFrame({"x": [0, 0.5, 1.3, "a"]})
    dfX = modin_df.__dataframe__()
    colX = dfX.get_column(0)
    col, _ = primitive_column_to_ndarray(colX)
    assert modin_df["x"].dtype == col.dtype


def test_from_dataframe_to_pandas():
    modin_df = pd.DataFrame(test_data["int_data"])
    dfX = modin_df.__dataframe__()
    with pytest.raises(ValueError):
        from_dataframe_to_pandas(dfX)

    from_dataframe_to_pandas(modin_df)


@pytest.mark.parametrize("num_partitions", [1, 2, 4, 6])
def test_get_chunks(num_partitions):
    NPartitions.put(num_partitions)
    df = pd.DataFrame({"x": np.arange(200)})
    df2 = df.__dataframe__()
    assert df2.num_chunks() == num_partitions
    _ = iter(df2.get_chunks())
    for i in range(1, num_partitions):
        _ = iter(df2.get_chunks(i))


def test_get_validity_buffer_nan_data_error():
    arr = [0, np.nan, 1, np.nan, -1, np.nan]
    df = pd.DataFrame({"a": arr})
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    with pytest.raises(NoValidityBuffer) as exc:
        colX._get_validity_buffer()

    assert "This column uses NaN as null so does not have a separate mask" in str(
        exc.value
    )


def test_int_nan_data_buffer():
    arr = [0, np.nan, 1, np.nan, -1, np.nan]
    df = pd.DataFrame({"a": arr})
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    dataBufX = colX._get_data_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0


def test_protocol_df_chunk_to_pandas():
    modin_df = pd.DataFrame({1: [0, 0.5, 1.3, "a"]})
    dfX = modin_df.__dataframe__()
    with pytest.raises(ValueError):
        protocol_df_chunk_to_pandas(dfX)


def test_set_nulls():
    pd_df = pd.DataFrame({"A": (test_data_categorical["ordered"])})
    arr = np.arange(test_data_categorical["ordered"].size)
    df = pd.DataFrame(pd_df)

    col = df.__dataframe__().get_column_by_name("A")
    assert set_nulls(arr, col, None) is arr
    assert set_nulls(arr, col, None, False) is arr


def test_simple_import():
    modin_df_producer = pd.DataFrame(test_data["int_data"])
    # Our configuration in pytest.ini requires that we explicitly catch all
    # instances of defaulting to pandas, this one raises a warning on `.from_dataframe`
    with warns_that_defaulting_to_pandas():
        modin_df_consumer = from_dataframe(modin_df_producer)

    # TODO: the following assertions verify that `from_dataframe` doesn't return
    # the same object untouched due to optimization branching, it actually should
    # do so but the logic is not implemented yet, so the assertions are passing
    # for now. It's required to replace the producer's type with a different one
    # to consumer when we have some other implementation of the protocol as the
    # assertions may start failing shortly.
    assert modin_df_producer is not modin_df_consumer
    assert (
        modin_df_producer._query_compiler._modin_frame
        is not modin_df_consumer._query_compiler._modin_frame
    )

    df_equals(modin_df_producer, modin_df_consumer)


def test_string_get_validity_buffer():
    arr = ["a", "b", "c"]
    df = pd.DataFrame({"a": arr})
    dfX = df.__dataframe__()
    colX = dfX.get_column(0)
    dataBufX = colX._get_validity_buffer()
    assert dataBufX[0].bufsize > 0
    assert dataBufX[0].ptr != 0
