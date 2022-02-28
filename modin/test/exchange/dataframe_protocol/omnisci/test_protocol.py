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

from modin.experimental.core.execution.native.implementations.omnisci_on_native.exchange.dataframe_protocol.__utils import (
    from_dataframe,
)

import modin.pandas as pd
import pandas
import numpy as np

from modin.pandas.test.utils import df_equals

data = {
    "a": np.array([1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
    "b": np.array(
        [2**64 - 1, 2**64 - 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint64
    ),
    "c": np.array(np.arange(12), dtype="datetime64[ns]"),
    "d": np.array(["a", "b", "c"] * 4),
    "e": pandas.Categorical(["a", "b", "c"] * 4),
}


def test_export():
    md_df = pd.DataFrame(data)
    exported_df = from_dataframe(md_df._query_compiler._modin_frame)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=3)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=5)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=12)
    df_equals(md_df, exported_df)


data_null = {
    "a": np.array([1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
    "b": np.array(
        [2**64 - 1, 2**64 - 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint64
    ),
    "c": np.array(
        [1, 2, None, 4, 5, 6, None, 8, 9, None, None, 12], dtype="datetime64[ns]"
    ),
    "d": np.array(["a", "b", None] * 4),
    "e": pandas.Categorical(["a", None, "c"] * 4),
}


def test_export_nulls():
    md_df = pd.DataFrame(data_null)
    exported_df = from_dataframe(md_df._query_compiler._modin_frame)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=3)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=5)
    df_equals(md_df, exported_df)

    exported_df = from_dataframe(md_df._query_compiler._modin_frame, nchunks=12)
    df_equals(md_df, exported_df)
