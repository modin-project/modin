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
import warnings
import numpy as np
from modin.config import Engine, NPartitions
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.pandas.test.utils import df_equals

@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="Only Ray supports the Batch Pipeline API",
)
def test_pipeline_simple():
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
        return pandas.concat([df, df.iloc[[-1]]])
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
    df_equals(corr_df, new_df)
    num_ptns = NPartitions.get()
    df = df._build_batch_pipeline(lambda df: df, 0, num_partitions=(num_ptns - 1))
    assert NPartitions.get() == (num_ptns - 1), "Pipeline did not change NPartitions.get()"
    NPartitions.put(num_ptns)

def test_pipeline_multiple_outputs():
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True)
    df = df._add_batch_query(lambda df: df + 30, is_output=True)
    new_dfs = df._compute_batch()
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    corr_df = pd.DataFrame(arr) * -30
    df_equals(corr_df, new_dfs[0]) # First output computed correctly
    corr_df = corr_df.rename(columns={i:f"col {i}" for i in range(1000)})
    df_equals(corr_df, new_dfs[1]) # Second output computed correctly
    corr_df += 30
    df_equals(corr_df, new_dfs[2]) # Third output computed correctly
    # Testing Output ID + Multiple Outputs now
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch()
    assert isinstance(new_dfs, dict), "Pipeline did not return a dictionary mapping output_ids to dfs"
    assert 20 in new_dfs, "Output ID 1 not cached correctly"
    assert 21 in new_dfs, "Output ID 2 not cached correctly"
    assert 22 in new_dfs, "Output ID 3 not cached correctly"
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    corr_df = pd.DataFrame(arr) * -30
    df_equals(corr_df, new_dfs[20]) # First output computed correctly
    corr_df = corr_df.rename(columns={i:f"col {i}" for i in range(1000)})
    df_equals(corr_df, new_dfs[21]) # Second output computed correctly
    corr_df += 30
    df_equals(corr_df, new_dfs[22]) # Third output computed correctly

def test_pipeline_postprocessing():
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True)
    df = df._add_batch_query(lambda df: df + 30, is_output=True)
    def new_col_adder(df):
        df['new_col'] = df.iloc[:, -1]
        return df
    new_dfs = df._compute_batch(postprocessor=new_col_adder)
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    corr_df = pd.DataFrame(arr) * -30
    corr_df['new_col'] = corr_df.iloc[:, -1]
    df_equals(corr_df, new_dfs[0])
    corr_df = corr_df.drop(columns=["new_col"])
    corr_df = corr_df.rename(columns={i:f"col {i}" for i in range(1000)})
    corr_df['new_col'] = corr_df.iloc[:, -1]
    df_equals(corr_df, new_dfs[1])
    corr_df = corr_df.drop(columns=['new_col'])
    corr_df += 30
    corr_df['new_col'] = corr_df.iloc[:, -1]
    df_equals(corr_df, new_dfs[2])

def test_pipeline_postprocessing_w_output_id():
    # Testing Postprocessing + Output ID without passing to postprocessor
    def new_col_adder(df):
        df['new_col'] = df.iloc[:, -1]
        return df
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(postprocessor=new_col_adder)
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    # Testing Postprocessing + Output ID with passing to postprocessor
    def new_col_adder(df, o_id):
        df['new_col'] = o_id
        return df
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(postprocessor=new_col_adder, pass_output_id=True)
    corr_df = pd.DataFrame(arr) * -30
    corr_df['new_col'] = 20
    df_equals(corr_df, new_dfs[20])
    corr_df = corr_df.drop(columns=["new_col"])
    corr_df = corr_df.rename(columns={i:f"col {i}" for i in range(1000)})
    corr_df['new_col'] = 21
    df_equals(corr_df, new_dfs[21])
    corr_df = corr_df.drop(columns=['new_col'])
    corr_df += 30
    corr_df['new_col'] = 22
    df_equals(corr_df, new_dfs[22])

def test_pipeline_postprocessing_w_ptn_id():
    arr = np.random.randint(0, 1000, (1000, 1000))
    def new_col_adder(df, ptn_id):
        df['new_col'] = ptn_id
        return df
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(postprocessor=new_col_adder, pass_partition_id=True)
    corr_df = pd.DataFrame(arr) * -30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[20])
    corr_df = corr_df.drop(columns=["new_col"])
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[21])
    corr_df = corr_df.drop(columns=['new_col'])
    corr_df += 30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[22])
    
def test_pipeline_postprocessing_w_all_metadata():
    arr = np.random.randint(0, 1000, (1000, 1000))
    def new_col_adder(df, o_id, ptn_id):
        df['new_col'] = f'{o_id} {ptn_id}'
        return df
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(postprocessor=new_col_adder, pass_partition_id=True, pass_output_id=True)
    corr_df = pd.DataFrame(arr) * -30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, 20, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[20])
    corr_df = corr_df.drop(columns=["new_col"])
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, 21, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[21])
    corr_df = corr_df.drop(columns=['new_col'])
    corr_df += 30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
    ptns = [ptn.add_to_apply_calls(new_col_adder, 22, i) for i, ptn in enumerate(ptns)]
    [ptn.drain_call_queue() for ptn in ptns]
    ptns = [ptn.list_of_blocks for ptn in ptns]
    corr_df = from_partitions(ptns, axis=None)
    df_equals(corr_df, new_dfs[22])

def test_pipeline_final_result_func():
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True)
    df = df._add_batch_query(lambda df: df + 30, is_output=True)
    new_dfs = df._compute_batch(final_result_func=lambda df: df.iloc[-1])
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    corr_df = pd.DataFrame(arr) * -30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[0]) # First output computed correctly
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[1]) # Second output computed correctly
    corr_df += 30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[2]) # Third output computed correctly
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(final_result_func=lambda df: df.iloc[-1])
    assert isinstance(new_dfs, dict), "Pipeline did not return a dictionary mapping output_ids to dfs"
    assert 20 in new_dfs, "Output ID 1 not cached correctly"
    assert 21 in new_dfs, "Output ID 2 not cached correctly"
    assert 22 in new_dfs, "Output ID 3 not cached correctly"
    assert len(new_dfs) == 3, "Pipeline did not return all outputs"
    corr_df = pd.DataFrame(arr) * -30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[20]) # First output computed correctly
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[21]) # Second output computed correctly
    corr_df += 30
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[22]) # Third output computed correctly

def test_pipeline_postproc_and_final():
    arr = np.random.randint(0, 1000, (1000, 1000))
    def new_col_adder(df, o_id, ptn_id):
        df['new_col'] = f'{o_id} {ptn_id}'
        return df
    df = pd.DataFrame(arr)
    df = df._build_batch_pipeline(lambda df: df * -30, 0, is_output=True, output_id=20)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}), is_output=True, output_id=21)
    df = df._add_batch_query(lambda df: df + 30, is_output=True, output_id=22)
    new_dfs = df._compute_batch(postprocessor=new_col_adder, pass_partition_id=True, pass_output_id=True, final_result_func=lambda df: df.iloc[-1])
    corr_df = pd.DataFrame(arr) * -30
    corr_df = new_col_adder(corr_df, 20, 0)
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[20])
    corr_df = corr_df.drop(columns=["new_col"])
    corr_df = pd.DataFrame(corr_df.rename(columns={i:f"col {i}" for i in range(1000)})._to_pandas())
    corr_df = new_col_adder(corr_df, 21, 0)
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[21])
    corr_df = corr_df.drop(columns=['new_col'])
    corr_df += 30
    corr_df = new_col_adder(corr_df, 22, 0)
    corr_df_md = corr_df._query_compiler._modin_frame
    ptn = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0].to_pandas().iloc[-1]
    df_equals(ptn, new_dfs[22])

def test_reptn_after():
    import pandas
    df = pd.DataFrame([list(range(1000))])
    df = df._build_batch_pipeline(lambda df: pandas.concat([df]*1000), 0, repartition_after=True)
    def new_col_adder(df, ptn_id):
        df['new_col'] = ptn_id
        return df
    df = df._add_batch_query(new_col_adder, is_output=True, pass_partition_id=True)
    new_dfs = df._compute_batch()
    assert len(new_dfs[0]['new_col'].unique()) == NPartitions.get()

def test_fan_out():
    df = pd.DataFrame([[0, 1, 2]])
    def new_col_adder(df, ptn_id):
        df['new_col'] = ptn_id
        return df
    def reducer(dfs):
        new_cols = ''.join([str(df['new_col'].values[0]) for df in dfs])
        dfs[0]['new_col1'] = new_cols
        return dfs[0]
    df = df._build_batch_pipeline(new_col_adder, 0, fan_out=True, reduce_fn=reducer, pass_partition_id=True, is_output=True)
    new_df = df._compute_batch()[0]
    corr_df = pd.DataFrame([[0, 1, 2]])
    corr_df['new_col'] = 0
    corr_df['new_col1'] = ''.join([str(i) for i in range(NPartitions.get())])
    df_equals(corr_df, new_df)
