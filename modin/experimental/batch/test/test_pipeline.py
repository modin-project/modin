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
from modin.config import Engine, NPartitions
from modin.distributed.dataframe.pandas.partitions import from_partitions
from modin.experimental.batch.pipeline import PandasQueryPipeline
from modin.pandas.test.utils import df_equals


@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="Only Ray supports the Batch Pipeline API",
)
class TestPipelineRayEngine:
    def test_warnings(self):
        """
        This test ensures that creating a Pipeline object raises the correct warnings.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        # Ensure that building a pipeline warns users that it is an experimental feature
        with pytest.warns(
            UserWarning,
            match="The Batch Pipeline API is an experimental feature and still under development in Modin.",
        ):
            pipeline = PandasQueryPipeline(df)
        with pytest.warns(
            UserWarning,
            match="No outputs to compute. Returning an empty list. Please specify outputs by calling `add_query` with `is_output=True`.",
        ):
            output = pipeline.compute_batch()
        assert output == [], "Empty pipeline did not return an empty list."

    def test_pipeline_simple(self):
        """
        This test creates a simple pipeline and ensures that it runs end to end correctly.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)

        def add_col(df):
            df["new_col"] = df.sum(axis=1)
            return df

        # Build pipeline
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(add_col)
        pipeline.add_query(lambda df: df * -30)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)})
        )

        def add_row_to_ptn(df):
            import pandas

            return pandas.concat([df, df.iloc[[-1]]])

        pipeline.add_query(add_row_to_ptn, is_output=True)
        new_df = pipeline.compute_batch()[0]
        # Build df without pipelining to ensure correctness
        corr_df = add_col(pd.DataFrame(arr))
        corr_df *= -30
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [ptn.add_to_apply_calls(add_row_to_ptn) for ptn in ptns]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        # Compare pipelined and non-pipelined df
        df_equals(corr_df, new_df)
        # Ensure that setting `num_partitions` when creating a pipeline does not change `NPartitions`
        num_ptns = NPartitions.get()
        PandasQueryPipeline(df, num_partitions=(num_ptns - 1))
        assert (
            NPartitions.get() == num_ptns
        ), "Pipeline did not change NPartitions.get()"

    def test_update_df(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df + 3, is_output=True)
        new_df = df * -1
        pipeline.update_df(new_df)
        output_df = pipeline.compute_batch()[0]
        df_equals((df * -1) + 3, output_df)

    def test_multiple_outputs(self):
        """
        This test creates a pipeline with multiple outputs, and checks that all are computed correctly.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True)
        new_dfs = pipeline.compute_batch()
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        corr_df = pd.DataFrame(arr) * -30
        df_equals(corr_df, new_dfs[0])  # First output computed correctly
        corr_df = corr_df.rename(columns={i: f"col {i}" for i in range(1000)})
        df_equals(corr_df, new_dfs[1])  # Second output computed correctly
        corr_df += 30
        df_equals(corr_df, new_dfs[2])  # Third output computed correctly

    def test_output_id(self):
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df, 0)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        with pytest.raises(
            ValueError, match="Output ID must be specified for all nodes."
        ):
            pipeline.add_query(
                lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
                is_output=True,
            )
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True)
        with pytest.raises(
            ValueError, match="Output ID must be specified for all nodes."
        ):
            pipeline.add_query(
                lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
                is_output=True,
                output_id=20,
            )
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df, is_output=True)
        with pytest.raises(
            ValueError,
            match=(
                "`pass_output_id` is set to True, but output ids have not been specified. "
                + "To pass output ids, please specify them using the `output_id` kwarg with pipeline.add_query"
            ),
        ):
            pipeline.compute_batch(postprocessor=lambda df: df, pass_output_id=True)

    def test_output_id_multiple_outputs(self):
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch()
        assert isinstance(
            new_dfs, dict
        ), "Pipeline did not return a dictionary mapping output_ids to dfs"
        assert 20 in new_dfs, "Output ID 1 not cached correctly"
        assert 21 in new_dfs, "Output ID 2 not cached correctly"
        assert 22 in new_dfs, "Output ID 3 not cached correctly"
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        corr_df = pd.DataFrame(arr) * -30
        df_equals(corr_df, new_dfs[20])  # First output computed correctly
        corr_df = corr_df.rename(columns={i: f"col {i}" for i in range(1000)})
        df_equals(corr_df, new_dfs[21])  # Second output computed correctly
        corr_df += 30
        df_equals(corr_df, new_dfs[22])  # Third output computed correctly

    def test_postprocessing(self):
        """
        This test checks that the `postprocessor` argument to `_compute_batch` is handled correctly.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True)

        def new_col_adder(df):
            df["new_col"] = df.iloc[:, -1]
            return df

        new_dfs = pipeline.compute_batch(postprocessor=new_col_adder)
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        corr_df = pd.DataFrame(arr) * -30
        corr_df["new_col"] = corr_df.iloc[:, -1]
        df_equals(corr_df, new_dfs[0])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df = corr_df.rename(columns={i: f"col {i}" for i in range(1000)})
        corr_df["new_col"] = corr_df.iloc[:, -1]
        df_equals(corr_df, new_dfs[1])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df += 30
        corr_df["new_col"] = corr_df.iloc[:, -1]
        df_equals(corr_df, new_dfs[2])

    def test_postprocessing_w_output_id(self):
        """
        This test checks that the `postprocessor` argument is correctly handled when `output_id` is specified.
        """
        # Testing Postprocessing + Output ID without passing to postprocessor
        def new_col_adder(df):
            df["new_col"] = df.iloc[:, -1]
            return df

        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(postprocessor=new_col_adder)
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        # Testing Postprocessing + Output ID with passing to postprocessor

        def new_col_adder(df, o_id):
            df["new_col"] = o_id
            return df

        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(
            postprocessor=new_col_adder, pass_output_id=True
        )
        corr_df = pd.DataFrame(arr) * -30
        corr_df["new_col"] = 20
        df_equals(corr_df, new_dfs[20])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df = corr_df.rename(columns={i: f"col {i}" for i in range(1000)})
        corr_df["new_col"] = 21
        df_equals(corr_df, new_dfs[21])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df += 30
        corr_df["new_col"] = 22
        df_equals(corr_df, new_dfs[22])

    def test_postprocessing_w_ptn_id(self):
        """
        This test checks that the postprocessing is correctly handled when `partition_id` is passed.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))

        def new_col_adder(df, ptn_id):
            df["new_col"] = ptn_id
            return df

        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(
            postprocessor=new_col_adder, pass_partition_id=True
        )
        corr_df = pd.DataFrame(arr) * -30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[20])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[21])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df += 30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [ptn.add_to_apply_calls(new_col_adder, i) for i, ptn in enumerate(ptns)]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[22])

    def test_postprocessing_w_all_metadata(self):
        """
        This test checks that postprocessing is correctly handled when `partition_id` and `output_id` are passed.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))

        def new_col_adder(df, o_id, ptn_id):
            df["new_col"] = f"{o_id} {ptn_id}"
            return df

        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(
            postprocessor=new_col_adder, pass_partition_id=True, pass_output_id=True
        )
        corr_df = pd.DataFrame(arr) * -30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [
            ptn.add_to_apply_calls(new_col_adder, 20, i) for i, ptn in enumerate(ptns)
        ]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[20])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [
            ptn.add_to_apply_calls(new_col_adder, 21, i) for i, ptn in enumerate(ptns)
        ]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[21])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df += 30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptns = corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)
        ptns = [
            ptn.add_to_apply_calls(new_col_adder, 22, i) for i, ptn in enumerate(ptns)
        ]
        [ptn.drain_call_queue() for ptn in ptns]
        ptns = [ptn.list_of_blocks for ptn in ptns]
        corr_df = from_partitions(ptns, axis=None)
        df_equals(corr_df, new_dfs[22])

    def test_final_result_func(self):
        """
        This test checks that when `final_result_func` is specified, outputs are computed correctly.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True)
        new_dfs = pipeline.compute_batch(final_result_func=lambda df: df.iloc[-1])
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        corr_df = pd.DataFrame(arr) * -30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[0])  # First output computed correctly
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[1])  # Second output computed correctly
        corr_df += 30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[2])  # Third output computed correctly
        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(final_result_func=lambda df: df.iloc[-1])
        assert isinstance(
            new_dfs, dict
        ), "Pipeline did not return a dictionary mapping output_ids to dfs"
        assert 20 in new_dfs, "Output ID 1 not cached correctly"
        assert 21 in new_dfs, "Output ID 2 not cached correctly"
        assert 22 in new_dfs, "Output ID 3 not cached correctly"
        assert len(new_dfs) == 3, "Pipeline did not return all outputs"
        corr_df = pd.DataFrame(arr) * -30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[20])  # First output computed correctly
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[21])  # Second output computed correctly
        corr_df += 30
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[22])  # Third output computed correctly

    def test_postproc_and_final(self):
        """
        This test checks that when postprocessor and final_result_func are both present, outputs are computed correctly.
        """
        arr = np.random.randint(0, 1000, (1000, 1000))

        def new_col_adder(df, o_id, ptn_id):
            df["new_col"] = f"{o_id} {ptn_id}"
            return df

        df = pd.DataFrame(arr)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(lambda df: df * -30, is_output=True, output_id=20)
        pipeline.add_query(
            lambda df: df.rename(columns={i: f"col {i}" for i in range(1000)}),
            is_output=True,
            output_id=21,
        )
        pipeline.add_query(lambda df: df + 30, is_output=True, output_id=22)
        new_dfs = pipeline.compute_batch(
            postprocessor=new_col_adder,
            pass_partition_id=True,
            pass_output_id=True,
            final_result_func=lambda df: df.iloc[-1],
        )
        corr_df = pd.DataFrame(arr) * -30
        corr_df = new_col_adder(corr_df, 20, 0)
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[20])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df = pd.DataFrame(
            corr_df.rename(columns={i: f"col {i}" for i in range(1000)})._to_pandas()
        )
        corr_df = new_col_adder(corr_df, 21, 0)
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[21])
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df += 30
        corr_df = new_col_adder(corr_df, 22, 0)
        corr_df_md = corr_df._query_compiler._modin_frame
        ptn = (
            corr_df_md._partition_mgr_cls.row_partitions(corr_df_md._partitions)[0]
            .to_pandas()
            .iloc[-1]
        )
        df_equals(ptn, new_dfs[22])

    def test_reptn_after(self):
        """
        This test checks that the `repartition_after` argument is appropriately handled.
        """
        import pandas

        df = pd.DataFrame([list(range(1000))])
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(
            lambda df: pandas.concat([df] * 1000), repartition_after=True
        )

        def new_col_adder(df, ptn_id):
            df["new_col"] = ptn_id
            return df

        pipeline.add_query(new_col_adder, is_output=True, pass_partition_id=True)
        new_dfs = pipeline.compute_batch()
        assert len(new_dfs[0]["new_col"].unique()) == NPartitions.get()
        # Test that more than one partition causes an error
        import ray

        ptn1 = ray.put(pandas.DataFrame([[0, 1, 2]]))
        ptn2 = ray.put(pandas.DataFrame([[3, 4, 5]]))
        df = from_partitions([ptn1, ptn2], 0)
        pipeline = PandasQueryPipeline(df, 0)
        pipeline.add_query(lambda df: df, repartition_after=True, is_output=True)

        with pytest.raises(
            NotImplementedError,
            match="Dynamic repartitioning is currently only supported for DataFrames with 1 partition.",
        ):
            new_dfs = pipeline.compute_batch()

    def test_fan_out(self):
        """
        This test checks that the fan_out argument is appropriately handled.
        """
        df = pd.DataFrame([[0, 1, 2]])

        def new_col_adder(df, ptn_id):
            df["new_col"] = ptn_id
            return df

        def reducer(dfs):
            new_cols = "".join([str(df["new_col"].values[0]) for df in dfs])
            dfs[0]["new_col1"] = new_cols
            return dfs[0]

        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(
            new_col_adder,
            fan_out=True,
            reduce_fn=reducer,
            pass_partition_id=True,
            is_output=True,
        )
        new_df = pipeline.compute_batch()[0]
        corr_df = pd.DataFrame([[0, 1, 2]])
        corr_df["new_col"] = 0
        corr_df["new_col1"] = "".join([str(i) for i in range(NPartitions.get())])
        df_equals(corr_df, new_df)
        # Test that if more than one partition, all but first are ignored
        import pandas
        import ray

        ptn1 = ray.put(pandas.DataFrame([[0, 1, 2]]))
        ptn2 = ray.put(pandas.DataFrame([[3, 4, 5]]))
        df = from_partitions([ptn1, ptn2], 0)
        pipeline = PandasQueryPipeline(df)
        pipeline.add_query(
            new_col_adder,
            fan_out=True,
            reduce_fn=reducer,
            pass_partition_id=True,
            is_output=True,
        )
        with pytest.raises(
            NotImplementedError,
            match="Fan out is only supported with DataFrames with 1 partition.",
        ):
            new_df = pipeline.compute_batch()[0]

    def test_pipeline_complex(self):
        import pandas
        from os.path import exists
        from os import remove

        df = pd.DataFrame([[0, 1, 2]])

        def new_col_adder(df, ptn_id):
            df["new_col"] = ptn_id
            return df

        def reducer(dfs):
            new_cols = "".join([str(df["new_col"].values[0]) for df in dfs])
            dfs[0]["new_col1"] = new_cols
            return dfs[0]

        pipeline = PandasQueryPipeline(df, num_partitions=24)
        pipeline.add_query(
            new_col_adder,
            fan_out=True,
            reduce_fn=reducer,
            pass_partition_id=True,
            is_output=True,
            output_id=20,
        )
        pipeline.add_query(
            lambda df: pandas.concat([df] * 1000), repartition_after=True
        )
        pipeline.add_query(
            lambda df: df.drop(columns=["new_col"]), is_output=True, output_id=21
        )

        def to_csv(df, ptn_id):
            df.to_csv(f"{ptn_id}.csv")
            return df

        pipeline.add_query(to_csv, is_output=True, output_id=22, pass_partition_id=True)

        def post_proc(df, o_id, ptn_id):
            df["new_col"] = f"{o_id} {ptn_id}"
            return df

        new_dfs = pipeline.compute_batch(
            postprocessor=post_proc,
            pass_partition_id=True,
            pass_output_id=True,
            final_result_func=lambda df: df.iloc[-1],
        )
        corr_df = pd.DataFrame([[0, 1, 2]])
        corr_df["new_col"] = "20 0"
        corr_df["new_col1"] = "".join([str(i) for i in range(24)])
        df_equals(corr_df.iloc[-1], new_dfs[20])
        corr_df = pd.concat([corr_df] * 1000)
        corr_df = corr_df.drop(columns=["new_col"])
        corr_df["new_col"] = "21 0"
        df_equals(corr_df.iloc[0], new_dfs[21])
        for i in range(24):
            assert exists(
                f"{i}.csv"
            ), "CSV File for Partition {i} does not exist, even though dataframe should have been repartitioned."
            remove(f"{i}.csv")
        assert 22 in new_dfs, "Output for output id 22 not generated."


@pytest.mark.skipif(
    Engine.get() == "Ray",
    reason="Ray supports the Batch Pipeline API",
)
def test_pipeline_unsupported_engine():
    """
    This test ensures that trying to use the Pipeline API with an unsupported Engine raises errors.
    """
    # Check that pipeline does not allow `Engine` to not be Ray.
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(
        NotImplementedError,
        match="Batch Pipeline API is only implemented for Ray Engine.",
    ):
        PandasQueryPipeline(df)

    eng = Engine.get()
    Engine.put("Ray")
    # Check that even if Engine is Ray, if the df is not backed by Ray, the Pipeline does not allow initialization.
    with pytest.raises(
        NotImplementedError,
        match="Batch Pipeline API is only implemented for Ray Engine.",
    ):
        PandasQueryPipeline(df, 0)
    new_df = pd.DataFrame([[1, 2, 3], [5, 6, 7]])
    pipeline = PandasQueryPipeline(new_df)
    # Check that even if Engine is Ray, if the new df is not backed by Ray, the Pipeline does not allow an update.
    with pytest.raises(
        NotImplementedError,
        match="Batch Pipeline API is only implemented for Ray Engine.",
    ):
        pipeline.update_df(df)
    Engine.put(eng)
    # Check that pipeline does not allow an update when `Engine` is not Ray.
    with pytest.raises(
        NotImplementedError,
        match="Batch Pipeline API is only implemented for Ray Engine.",
    ):
        pipeline.update_df(df)
