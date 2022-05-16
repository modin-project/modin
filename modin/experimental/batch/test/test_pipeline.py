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

import pytest
import modin.pandas as pd
import numpy as np
from modin.config import Engine
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.pandas.test.utils import df_equals

@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="Only Ray supports the Batch Pipeline API",
)
def test_pipeline():
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    with pytest.warns(UserWarning, match="No pipeline exists. Please call `df._build_batch_pipeline` first to create a batch pipeline."):
        df._add_batch_query(lambda df: df)
    with pytest.warns(UserWarning, match="The Batch Pipeline API is an experimental feature and still under development in Modin."):
        df = df._build_batch_pipeline(lambda df: df, 0)
    query = df._pipeline.nodes_list[0]
    with pytest.warns(UserWarning, match="Existing pipeline discovered. Please call this function again with `overwrite_existing` set to True to overwrite this pipeline."):
        df = df._build_batch_pipeline(lambda df: df.iloc[0], 0)
    assert df._pipeline.nodes_list[0] == query, "Pipeline was overwritten when `overwrite_existing` was not set to True."
    def add_col(df):
        df['new_col'] = df.sum(axis=1)
        return df
    df = df._build_batch_pipeline(add_col, 0, overwrite_existing=True)
    assert df._pipeline.nodes_list[0] != query, "Pipeline was not overwritten when `overwrite_existing` was set to True."
    df = df._add_batch_query(lambda df: df * -30)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}))
    def add_row_to_ptn(df):
        import pandas
        return pandas.concat([df, df.iloc[-1].T])
    df = df._add_batch_query(add_row_to_ptn, is_output=True)
    new_df = df._compute_batch()[0]
    corr_df = add_col(pd.DataFrame(arr))
    corr_df *= -30
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(add_row_to_ptn) for ptn in ptns]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    assert df_equals(corr_df, new_df), "Pipelined DF results differ from non-pipelined df"
    
