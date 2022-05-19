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

from modin.error_message import ErrorMessage
from modin.core.dataframe.base.dataframe.utils import Axis
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
    PandasOnRayDataframe,
)
from typing import Union, Callable, Optional
from modin.config import Engine, NPartitions
import ray
import numpy as np
from modin.distributed.dataframe.pandas.partitions import from_partitions


class PandasQuery(object):
    """
    Internal representation of a single query in a pipeline.

    This object represents a single function to be pipelined in a batch pipeline.

    Parameters
    ----------
    func : Callable
        The function to apply to the dataframe.
    is_output : bool, default: False
        Whether this query is an output query and should be passed both to the next query, and directly to postprocessing.
    repartition_after : bool, default: False
        Whether to repartition after this query is computed. Currently, repartitioning is only supported if there is 1 partition prior to repartitioning.
    fan_out : bool, default: False
        Whether to fan out this node. If True and only 1 partition is passed as input, the partition is replicated `num_partition` times, and the function is called on each. The `reduce_fn` must also be specified.
    pass_partition_id : bool, default: False
        Whether to pass the numerical partition id to the query.
    reduce_fn : Callable
        The reduce function to apply if `fan_out` is set to True. This takes the `num_partition` partitions that result from this query, and combines them into 1 partition.

    Notes
    -----
    func must be a function that is applied along an axis of the dataframe.
    """

    def __init__(
        self,
        func: Callable,
        is_output: bool = False,
        repartition_after: bool = False,
        fan_out: bool = False,
        pass_partition_id: bool = False,
        reduce_fn: Optional[Callable] = None,
    ):
        self.func = func
        self.is_output = is_output
        self.oids = []
        self.fan_out = fan_out
        self.repartition_after = repartition_after
        self.pass_partition_id = pass_partition_id
        self.operators = None
        self.reduce_fn = reduce_fn


class PandasQueryPipeline(object):
    """
    Internal representation of a query pipeline.

    This object keeps track of the functions that compose to form a query pipeline.

    Parameters
    ----------
    df : modin.pandas.Dataframe
        The dataframe to perform this pipeline on.
    axis : int or modin.core.dataframe.base.dataframe.utils.Axis
        The axis along which to partition the dataframe for this pipeline.
    num_partitions : int, default: `NPartitions.get()`
        The number of partitions to maintain for the batched dataframe.

    Notes
    -----
    Only axis-wide pipelines are supported. All queries will be applied along the same axis.
    """

    def __init__(self, df, axis: Union[int, Axis], num_partitions: int = -1):
        if Engine.get() != "Ray" or (
            not isinstance(df._query_compiler._modin_frame, PandasOnRayDataframe)
        ):  # pragma: no cover
            ErrorMessage.not_implemented(
                "Batch Pipeline API is only implemented for Ray Engine."
            )
        ErrorMessage.single_warning(
            "The Batch Pipeline API is an experimental feature and still under development in Modin."
        )
        num_partitions = num_partitions if num_partitions > 0 else NPartitions.get()
        self.df = df
        self.axis = Axis(axis)
        self.num_partitions = num_partitions
        self.outputs = []
        self.nodes_list = []
        self.node_to_id = None

    def add_query(
        self,
        func: Callable,
        is_output: bool = False,
        repartition_after: bool = False,
        fan_out: bool = False,
        pass_partition_id: bool = False,
        reduce_fn: Optional[Callable] = None,
        output_id: Optional[int] = None,
    ):
        """
        Adds a query to the current pipeline.

        Parameters
        ----------
        func : Callable
            DataFrame query to perform.
        is_output : bool, default: False
            Whether this query should be designated as an output query. If `True`, the output of
            this query is passed both to the next query and directly to postprocessing.
        repartition_after : bool, default: False
            Whether the dataframe should be repartitioned after this query. Currently,
            repartitioning is only supported if there is 1 partition prior.
        fan_out : bool, default: False
            Whether to fan out this node. If True and only 1 partition is passed as input, the
            partition is replicated `num_partition` times, and the function is called on each.
            The `reduce_fn` must also be specified.
        pass_partition_id : bool, default: False
            Whether to pass the numerical partition id to the query.
        reduce_fn : Callable
            The reduce function to apply if `fan_out` is set to True. This takes the `num_partition`
            partitions that result from this query, and combines them into 1 partition.
        output_id : int, default None
            An id to assign to this node if it is an output.
        """
        self.nodes_list.append(
            PandasQuery(
                func,
                is_output,
                repartition_after,
                fan_out,
                pass_partition_id,
                reduce_fn,
            )
        )
        if is_output:
            self.outputs.append(self.nodes_list[-1])
            if output_id:
                if self.node_to_id is None:
                    if len(self.outputs) == 1:
                        self.node_to_id = {}
                    else:
                        raise ValueError("Output ID must be specified for all nodes.")
                self.node_to_id[self.outputs[-1]] = output_id
            if output_id is None and self.node_to_id is not None:
                raise ValueError("Output ID must be specified for all nodes.")
            curr_node = self.outputs[-1]
            curr_node.operators = self.nodes_list[:-1]
            self.nodes_list = []

    def _complete_nodes(self, list_of_nodes, ptns):
        """
        Run a sub-query end to end.

        Parameters
        ----------
        list_of_nodes: list of PandasQuery
            The functions that compose this query.
        ptns: list of PandasOnRayDataframeVirtualPartition
            The ptns that compose the dataframe that is input to this sub-query.
        """
        for node in list_of_nodes:
            if node.fan_out:
                if len(ptns) > 1:
                    ErrorMessage.not_implemented(
                        "Fan out is only supported with DataFrames with 1 partition."
                    )
                ptns[0] = ptns[0].force_materialization()
                ptn_list = ptns[0].list_of_partitions_to_combine
                ptns[0] = ptns[0].add_to_apply_calls(node.func, 0)
                ptns[0].drain_call_queue(num_splits=1)
                oids = []
                new_dfs = []
                for i in range(1, self.num_partitions):
                    new_dfs.append(
                        type(ptns[0])(
                            ptn_list,
                            full_axis=ptns[0].full_axis,
                        ).add_to_apply_calls(node.func, i)
                    )
                    new_dfs[-1].drain_call_queue(num_splits=1)
                    oids.extend(new_dfs[-1].list_of_blocks)
                ray.wait(oids, num_returns=len(oids))

                def reducer(df):
                    df_inputs = [df]
                    for df in new_dfs:
                        df_inputs.append(df.to_pandas())
                    return node.reduce_fn(df_inputs)

                ptns = [ptns[0].add_to_apply_calls(reducer)]
            elif node.repartition_after:
                if len(ptns) > 1:
                    ErrorMessage.not_implemented(
                        "Dynamic repartitioning is currently only supported for DataFrames with 1 partition."
                    )
                ptns[0] = ptns[0].add_to_apply_calls(node.func).force_materialization()
                new_dfs = []

                def masker(df, i):
                    new_length = len(df.index) // self.num_partitions
                    if i == (self.num_partitions - 1):
                        return df.iloc[i * new_length :]
                    return df.iloc[i * new_length : (i + 1) * new_length]

                for i in range(self.num_partitions):
                    new_dfs.append(
                        type(ptns[0])(
                            ptns[0].list_of_partitions_to_combine,
                            full_axis=ptns[0].full_axis,
                        ).add_to_apply_calls(masker, i)
                    )
                ptns = new_dfs
            else:
                if node.pass_partition_id:
                    ptns = [
                        ptn.add_to_apply_calls(node.func, i)
                        for i, ptn in enumerate(ptns)
                    ]
                else:
                    ptns = [ptn.add_to_apply_calls(node.func) for ptn in ptns]
        return ptns

    def compute_batch(
        self,
        postprocessor: Optional[Callable] = None,
        pass_partition_id: Optional[bool] = False,
        final_result_func: Optional[Callable] = None,
        pass_output_id: Optional[bool] = False,
    ):
        """
        Run the completed pipeline + any postprocessing steps end to end.

        Parameters
        ----------
        postprocessor : Callable
            A postprocessing function to be applied to each output partition.
        pass_partition_id : bool
            Whether or not to pass the numerical partition id to the postprocessing function.
        final_result_func : Callable
            A final result function that generates a final result for each output. It takes the
            first partition as input.
        pass_output_id : bool
            Whether or not to pass the output ID associated with output queries to the
            postprocessing function.

        Returns
        -------
        list or dict or DataFrame
            If output ids are specified, a dictionary mapping output id to the result of
            `final_result_func` is returned, otherwise, a list of the results of `final_result_func`
            is returned. If `final_result_func` is not specified, the resulting dataframes are
            returned in the format specified above.
        """
        if self.node_to_id is None and pass_output_id:
            raise ValueError(
                "`pass_output_id` is set to True, but output ids have not been specified. "
                + "To pass output ids, please specify them using the `output_id` kwarg with pipeline.add_query"
            )
        if self.node_to_id:
            outs = {}
        else:
            outs = []
        modin_frame = self.df._query_compiler._modin_frame
        ptns = (
            modin_frame._partition_mgr_cls.row_partitions(modin_frame._partitions)
            if self.axis == Axis.ROW_WISE
            else modin_frame._partition_mgr_cls.column_partitions(
                modin_frame._partitions
            )
        )
        for i, node in enumerate(self.outputs):
            ptns = self._complete_nodes(node.operators + [node], ptns)
            for ptn in ptns:
                ptn.drain_call_queue(num_splits=1)
            out_ptns = ptns
            if postprocessor:
                args = []
                if pass_output_id:
                    args.append(self.node_to_id[node])
                if pass_partition_id:
                    args.append(0)
                    new_ptns = []
                    for i, p in enumerate(ptns):
                        new_ptns.append(p.add_to_apply_calls(postprocessor, *args))
                        args[-1] = i + 1
                    out_ptns = new_ptns
                else:
                    out_ptns = [
                        ptn.add_to_apply_calls(postprocessor, *args) for ptn in ptns
                    ]
                if final_result_func is None:
                    [ptn.drain_call_queue(num_splits=1) for ptn in out_ptns]
            if self.node_to_id is None:
                outs.append(out_ptns)
            else:
                outs[self.node_to_id[node]] = out_ptns
        wait_parts = []
        if self.node_to_id is None:
            final_results = []
            for df in outs:
                if final_result_func:
                    d = df[0].to_pandas()
                    final_results.append(final_result_func(d))
                else:
                    ptns = np.array(df).flatten()
                    ptns = np.array([ptn.list_of_blocks for ptn in df]).flatten()
                    final_results.append(from_partitions(ptns, self.axis.value))
                for d in df:
                    wait_parts.extend(d.list_of_blocks)
        else:
            final_results = {}
            for id, df in outs.items():
                if final_result_func:
                    d = df[0].to_pandas()
                    final_results[id] = final_result_func(d)
                else:
                    ptns = np.array(df).flatten()
                    ptns = np.array([ptn.list_of_blocks for ptn in df]).flatten()
                    final_results[id] = from_partitions(ptns, self.axis.value)
                for d in df:
                    wait_parts.extend(d.list_of_blocks)
        ray.wait(wait_parts, num_returns=len(wait_parts))
        return final_results
