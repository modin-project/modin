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
import modin.pandas as pd
import pyarrow as pa
import pandas

from modin.pandas.test.utils import df_equals
from modin.pandas.utils import from_arrow

from .utils import get_all_types, split_df_into_chunks, export_frame


@pytest.mark.parametrize("data_has_nulls", [True, False])
@pytest.mark.parametrize("from_omnisci", [True, False])
def test_simple_export(data_has_nulls, from_omnisci):
    if from_omnisci:
        # OmniSci can't import 'uint64' as well as booleans
        # issue for bool: https://github.com/modin-project/modin/issues/4299
        exclude_dtypes = ["bool", "uint64"]
    else:
        exclude_dtypes = None

    data = get_all_types(has_nulls=data_has_nulls, exclude_dtypes=exclude_dtypes)
    md_df = pd.DataFrame(data)

    exported_df = export_frame(md_df, from_omnisci)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, from_omnisci, nchunks=3)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, from_omnisci, nchunks=5)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, from_omnisci, nchunks=12)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("nchunks", [2, 4, 7])
@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_export_aligned_at_chunks(nchunks, data_has_nulls):
    """Test export from DataFrame exchange protocol when internal arrow table is equaly chunked."""
    # Modin DataFrame constructor can't process pyarrow's category, so exclude it
    data = get_all_types(has_nulls=data_has_nulls, exclude_dtypes=["category"])
    pd_df = pandas.DataFrame(data)
    pd_chunks = split_df_into_chunks(pd_df, nchunks)

    chunked_at = pa.concat_tables([pa.Table.from_pandas(pd_df) for pd_df in pd_chunks])
    md_df = from_arrow(chunked_at)
    assert (
        len(md_df._query_compiler._modin_frame._partitions[0][0].get().column(0).chunks)
        == nchunks
    )

    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks * 2)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks * 3)
    df_equals(md_df, exported_df)


@pytest.mark.parametrize("data_has_nulls", [True, False])
def test_export_unaligned_at_chunks(data_has_nulls):
    """
    Test export from DataFrame exchange protocol when internal arrow table's chunks are unaligned.

    Arrow table allows for its columns to be chunked independently. Unaligned chunking means that
    each column has its individual chunking and so some preprocessing is required in order
    to emulate equaly chunked columns in the protocol.
    """
    # Modin DataFrame constructor can't process pyarrow's category, so exclude it
    data = get_all_types(has_nulls=data_has_nulls, exclude_dtypes=["category"])
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
        split_df_into_chunks(pd_df.iloc[:, cols], nchunks)
        for nchunks, cols in zip(chunk_groups, chunk_col_ilocs)
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
    for nchunks_group, cols in zip(chunk_groups, chunk_col_ilocs):
        for col in internal_at.select(range(cols.start, cols.stop)).columns:
            assert len(col.chunks) == nchunks_group

    nchunks = md_df.__dataframe__().num_chunks()

    exported_df = export_frame(md_df)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks * 2)
    df_equals(md_df, exported_df)

    exported_df = export_frame(md_df, nchunks=nchunks * 3)
    df_equals(md_df, exported_df)


def test_export_when_delayed_computations():
    # OmniSci can't import 'uint64' as well as booleans, so exclude them
    # issue for bool: https://github.com/modin-project/modin/issues/4299
    data = get_all_types(has_nulls=True, exclude_dtypes=["uint64", "bool"])
    md_df = pd.DataFrame(data)
    pd_df = pandas.DataFrame(data)

    md_res = md_df.fillna({"float32_null": 32.0, "float64_null": 64.0})
    pd_res = pd_df.fillna({"float32_null": 32.0, "float64_null": 64.0})
    assert (
        not md_res._query_compiler._modin_frame._has_arrow_table()
    ), "There are no delayed computations for the frame"

    exported_df = export_frame(md_res)
    df_equals(exported_df, pd_res)
