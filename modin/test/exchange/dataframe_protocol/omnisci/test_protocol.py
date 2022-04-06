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

"""Dataframe exchange protocol tests that are specific for OmniSci implementation."""

import pytest
import pyarrow as pa
import pandas
import numpy as np

import modin.pandas as pd
from modin.core.dataframe.pandas.exchange.dataframe_protocol.from_dataframe import (
    primitive_column_to_ndarray,
    buffer_to_ndarray,
    set_nulls,
)
from modin.pandas.utils import from_arrow, from_dataframe
from modin.pandas.test.utils import df_equals
from modin.test.test_utils import warns_that_defaulting_to_pandas
from .utils import get_data_of_all_types, split_df_into_chunks, export_frame


@pytest.mark.parametrize("data_has_nulls", [True, False])
@pytest.mark.parametrize("from_omnisci", [True, False])
@pytest.mark.parametrize("n_chunks", [None, 3, 5, 12])
def test_simple_export(data_has_nulls, from_omnisci, n_chunks):
    if from_omnisci:
        # OmniSci can't import 'uint64' as well as booleans
        # issue for bool: https://github.com/modin-project/modin/issues/4299
        exclude_dtypes = ["bool", "uint64"]
    else:
        exclude_dtypes = None

    data = get_data_of_all_types(
        has_nulls=data_has_nulls, exclude_dtypes=exclude_dtypes
    )
    md_df = pd.DataFrame(data)

    exported_df = export_frame(md_df, from_omnisci, n_chunks=n_chunks)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("n_chunks", [2, 4, 7])
@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_export_aligned_at_chunks(n_chunks, data_has_nulls):
    """Test export from DataFrame exchange protocol when internal PyArrow table is equaly chunked."""
    # Modin DataFrame constructor can't process PyArrow's category when using `from_arrow`, so exclude it
    data = get_data_of_all_types(has_nulls=data_has_nulls, exclude_dtypes=["category"])
    pd_df = pandas.DataFrame(data)
    pd_chunks = split_df_into_chunks(pd_df, n_chunks)

    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)
    assert (
        len(md_df._query_compiler._modin_frame._partitions[0][0].get().column(0).chunks)
        == md_df.__dataframe__().num_chunks()
        == n_chunks
    )

    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks * 2)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks * 3)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_export_unaligned_at_chunks(data_has_nulls):
    """
    Test export from DataFrame exchange protocol when internal PyArrow table's chunks are unaligned.

    Arrow table allows for its columns to be chunked independently. Unaligned chunking means that
    each column has its individual chunking and so some preprocessing is required in order
    to emulate equaly chunked columns in the protocol.
    """
    # Modin DataFrame constructor can't process PyArrow's category when using `from_arrow`, so exclude it
    data = get_data_of_all_types(has_nulls=data_has_nulls, exclude_dtypes=["category"])
    pd_df = pandas.DataFrame(data)
    # divide columns in 3 groups: unchunked, 2-chunked, 7-chunked
    chunk_groups = [1, 2, 7]
    chunk_col_ilocs = [
        slice(
            i * len(pd_df.columns) // len(chunk_groups),
            (i + 1) * len(pd_df.columns) // len(chunk_groups),
        )
        for i in range(len(chunk_groups))
    ]

    pd_chunk_groups = [
        split_df_into_chunks(pd_df.iloc[:, cols], n_chunks)
        for n_chunks, cols in zip(chunk_groups, chunk_col_ilocs)
    ]
    at_chunk_groups = [
        pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in chunk_group])
        for chunk_group in pd_chunk_groups
    ]

    chunked_at = at_chunk_groups[0]
    # TODO: appending columns one by one looks inefficient, is there a better way?
    for _at in at_chunk_groups[1:]:
        for field in _at.schema:
            chunked_at = chunked_at.append_column(field, _at[field.name])
    md_df = from_arrow(chunked_at)

    # verify that test generated the correct chunking
    internal_at = md_df._query_compiler._modin_frame._partitions[0][0].get()
    for n_chunks_group, cols in zip(chunk_groups, chunk_col_ilocs):
        for col in internal_at.select(range(cols.start, cols.stop)).columns:
            assert len(col.chunks) == n_chunks_group

    n_chunks = md_df.__dataframe__().num_chunks()

    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks * 2)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=n_chunks * 3)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_export_indivisible_chunking(data_has_nulls):
    """
    Test ``.get_chunks(n_chunks)`` when internal PyArrow table's is 'indivisibly chunked'.

    The setup for the test is a PyArrow table having one of the chunk consisting of a single row,
    meaning that the chunk can't be subdivide.
    """
    data = get_data_of_all_types(has_nulls=data_has_nulls, exclude_dtypes=["category"])
    pd_df = pandas.DataFrame(data)
    pd_chunks = (pd_df.iloc[:1], pd_df.iloc[1:])

    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)
    assert (
        len(md_df._query_compiler._modin_frame._partitions[0][0].get().column(0).chunks)
        == md_df.__dataframe__().num_chunks()
        == 2
    )
    # Meaning that we can't subdivide first chunk
    np.testing.assert_array_equal(
        md_df.__dataframe__()._chunk_slices, [0, 1, len(pd_df)]
    )

    exported_df = export_frame(md_df, n_chunks=2)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=4)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, n_chunks=40)
    df_equals(md_df, exported_df)


def test_export_when_delayed_computations():
    """
    Test that export works properly when OmnisciOnNative has delayed computations.

    If there are delayed functions and export is required, it has to trigger the execution
    first prior materializing protocol's buffers, so the buffers contain actual result
    of the computations.
    """
    # OmniSci can't import 'uint64' as well as booleans, so exclude them
    # issue for bool: https://github.com/modin-project/modin/issues/4299
    data = get_data_of_all_types(has_nulls=True, exclude_dtypes=["uint64", "bool"])
    md_df = pd.DataFrame(data)
    pd_df = pandas.DataFrame(data)

    md_res = md_df.fillna({"float32_null": 32.0, "float64_null": 64.0})
    pd_res = pd_df.fillna({"float32_null": 32.0, "float64_null": 64.0})
    assert (
        not md_res._query_compiler._modin_frame._has_arrow_table()
    ), "There are no delayed computations for the frame"

    exported_df = export_frame(md_res)
    df_equals(exported_df, pd_res)


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_simple_import(data_has_nulls):
    """Test that ``modin.pandas.utils.from_dataframe`` works properly."""
    data = get_data_of_all_types(data_has_nulls)

    modin_df_producer = pd.DataFrame(data)
    internal_modin_df_producer = modin_df_producer.__dataframe__()
    # Our configuration in pytest.ini requires that we explicitly catch all
    # instances of defaulting to pandas, this one raises a warning on `.from_dataframe`
    with warns_that_defaulting_to_pandas():
        modin_df_consumer = from_dataframe(modin_df_producer)
        internal_modin_df_consumer = from_dataframe(internal_modin_df_producer)

    # TODO: the following assertions verify that `from_dataframe` doesn't return
    # the same object untouched due to optimization branching, it actually should
    # do so but the logic is not implemented yet, so the assertions are passing
    # for now. It's required to replace the producer's type with a different one
    # to consumer when we have some other implementation of the protocol as the
    # assertions may start failing shortly.
    assert modin_df_producer is not modin_df_consumer
    assert internal_modin_df_producer is not internal_modin_df_consumer
    assert (
        modin_df_producer._query_compiler._modin_frame
        is not modin_df_consumer._query_compiler._modin_frame
    )

    df_equals(modin_df_producer, modin_df_consumer)
    df_equals(modin_df_producer, internal_modin_df_consumer)


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_zero_copy_export_for_primitives(data_has_nulls):
    """Test that basic data types can be zero-copy exported from OmnisciOnNative dataframe."""
    data = get_data_of_all_types(
        has_nulls=data_has_nulls, include_dtypes=["int", "uint", "float"]
    )
    at = pa.Table.from_pydict(data)

    md_df = from_arrow(at)
    protocol_df = md_df.__dataframe__(allow_copy=False)

    for i, col in enumerate(protocol_df.get_columns()):
        col_arr, memory_owner = primitive_column_to_ndarray(col)

        exported_ptr = col_arr.__array_interface__["data"][0]
        producer_ptr = at.column(i).chunks[0].buffers()[-1].address
        # Verify that the pointers of produce and exported objects point to the same data
        assert producer_ptr == exported_ptr

    # Can't export `md_df` zero-copy no more as it has delayed 'fillna' operation
    md_df = md_df.fillna({"float32": 32.0})
    non_zero_copy_protocol_df = md_df.__dataframe__(allow_copy=False)

    with pytest.raises(RuntimeError):
        col_arr, memory_owner = primitive_column_to_ndarray(
            non_zero_copy_protocol_df.get_column_by_name("float32")
        )


def test_bitmask_chunking():
    """Test that making a virtual chunk in a middle of a byte of a bitmask doesn't cause problems."""
    at = pa.Table.from_pydict({"col": [True, False, True, True, False] * 5})
    assert at["col"].type.bit_width == 1

    md_df = from_arrow(at)
    # Column length is 25, n_chunks is 2, meaning that the split will occur in the middle
    # of the second byte
    exported_df = export_frame(md_df, n_chunks=2)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("data_has_nulls", [True, False])
@pytest.mark.parametrize("n_chunks", [2, 9])
def test_buffer_of_chunked_at(data_has_nulls, n_chunks):
    """Test that getting buffers of physically chunked column works properly."""
    data = get_data_of_all_types(
        # For the simplicity of the test include only primitive types, so the test can use
        # only one function to export a column instead of if-elsing to find a type-according one
        has_nulls=data_has_nulls,
        include_dtypes=["bool", "int", "uint", "float"],
    )

    pd_df = pandas.DataFrame(data)
    pd_chunks = split_df_into_chunks(pd_df, n_chunks)

    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)

    protocol_df = md_df.__dataframe__()
    for i, col in enumerate(protocol_df.get_columns()):
        assert col.num_chunks() > 1
        assert len(col._pyarrow_table.column(0).chunks) > 1

        buffers = col.get_buffers()
        data_buff, data_dtype = buffers["data"]
        result = buffer_to_ndarray(data_buff, data_dtype, col.offset, col.size)
        result = set_nulls(result, col, buffers["validity"])

        # Our configuration in pytest.ini requires that we explicitly catch all
        # instances of defaulting to pandas, this one raises a warning on `.to_numpy()`
        with warns_that_defaulting_to_pandas():
            reference = md_df.iloc[:, i].to_numpy()

        np.testing.assert_array_equal(reference, result)

    protocol_df = md_df.__dataframe__(allow_copy=False)
    for i, col in enumerate(protocol_df.get_columns()):
        assert col.num_chunks() > 1
        assert len(col._pyarrow_table.column(0).chunks) > 1

        # Catch exception on attempt of doing a copy due to chunks combining
        with pytest.raises(RuntimeError):
            col.get_buffers()
