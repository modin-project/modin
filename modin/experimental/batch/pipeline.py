from modin.error_message import ErrorMessage
from modin.core.dataframe.base.dataframe.utils import Axis
from typing import Union, Callable, Optional
from modin.config import NPartitions
import ray
import numpy as np
from modin.distributed.dataframe.pandas.partitions import from_partitions


class PandasQuery(object):
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
        self.reptn = repartition_after
        self.pass_num_ptn = pass_partition_id
        self.operators = None
        self.reduce_fn = reduce_fn

    @classmethod
    def print_options(cls):
        """
        Prints the kwargs that Query objects accept.
        """
        print(
            """is_output : bool
    Whether or not this query is an output query and should be passed both to the next query, and directly to postprocessing.
repartition_after : bool
    Whether or not to repartition after this query is computed. Currently, repartitioning is only supported if there is 1 partition prior to repartitioning.
fan_out : bool
    Whether or not to fan out this node. If True and only 1 partition is passed as input, the partition is replicated `num_partition` times, and the function is called on each. The `reduce_fn` must also be specified.
pass_partition_id : bool
    Whether or not to pass the numerical partition id to the query.
reduce_fn : Calable
    The reduce function to apply if `fan_out` is set to True. This takes the `num_partition` partitions that result from this query, and combines them into 1 partition."""
        )


class PandasQueryPipeline(object):
    def __init__(self, df, axis: Union[int, Axis], num_partitions: int = 16):
        self.df = df
        self.axis = Axis(axis)
        self.num_partitions = num_partitions
        NPartitions.put(self.num_partitions)
        NPartitions.get = lambda: self.num_partitions
        self.unfinished = False
        self.outputs = []
        self.nodes_list = []
        self.node_to_id = None
        from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.virtual_partition import (
            PandasOnRayDataframeVirtualPartition,
        )

        self.dcq = PandasOnRayDataframeVirtualPartition.drain_call_queue

        def dr(self):
            """Execute all operations stored in this partition's call queue."""

            def drain(df):
                for func, args, kwargs in self.call_queue:
                    df = func(df, *args, **kwargs)
                return df

            drained = super(PandasOnRayDataframeVirtualPartition, self).apply(
                drain, num_splits=1
            )
            self.list_of_partitions_to_combine = drained
            self.call_queue = []

        PandasOnRayDataframeVirtualPartition.drain_call_queue = dr

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
                    self.node_to_id = {}
                self.node_to_id[self.outputs[-1]] = output_id
            curr_node = self.outputs[-1]
            curr_node.operators = self.nodes_list[:-1]
            self.nodes_list = []

    def _complete_nodes(self, list_of_nodes, ptns):
        for node in list_of_nodes:
            if node.fan_out:
                ptns[0] = ptns[0].force_materialization()
                ptn_list = ptns[0].list_of_partitions_to_combine
                ptns[0] = ptns[0].add_to_apply_calls(node.func, 0)
                ptns[0].drain_call_queue()
                oids = []
                new_dfs = []
                for i in range(1, self.num_partitions):
                    new_dfs.append(
                        type(ptns[0])(
                            ptn_list,
                            full_axis=ptns[0].full_axis,
                        ).add_to_apply_calls(node.func, i)
                    )
                    new_dfs[-1].drain_call_queue()
                    oids.extend(new_dfs[-1].list_of_blocks)
                ray.wait(oids, num_returns=len(oids))

                def reducer(df):
                    df_inputs = [df]
                    for df in new_dfs:
                        df_inputs.append(df.to_pandas())
                    return node.reduce_fn(df_inputs)

                ptns = [ptns[0].add_to_apply_calls(reducer)]
            elif node.reptn and len(ptns) == 1:
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
                if node.pass_num_ptn:
                    ptns = [
                        ptn.add_to_apply_calls(node.func, i)
                        for i, ptn in enumerate(ptns)
                    ]
                else:
                    ptns = [ptn.add_to_apply_calls(node.func) for ptn in ptns]
        return ptns

    def get_results(
        self,
        postprocessor: Optional[Callable] = None,
        pass_partition_id: Optional[bool] = False,
        final_result_func: Optional[Callable] = None,
        pass_output_id: Optional[bool] = False,
    ):
        if self.node_to_id is None and pass_output_id:
            ErrorMessage.single_warning(
                "`pass_output_id` is set to True, but output ids have not been specified. "
                + "To pass output ids, please specify them using the `output_id` kwarg with df._batch_api"
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
                ptn.drain_call_queue()
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
                # [ptn.drain_call_queue() for ptn in out_ptns]
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
                    final_results.append(from_partitions(np.array(df), self.axis))
                for d in df:
                    wait_parts.extend(d.list_of_blocks)

        else:
            final_results = {}
            for id, df in outs.items():
                if final_result_func:
                    d = df[0].to_pandas()
                    final_results[id] = final_result_func(d)
                else:
                    final_results[id] = from_partitions(np.array(df), self.axis)
                for d in df:
                    wait_parts.extend(d.list_of_blocks)
        ray.wait(wait_parts, num_returns=len(wait_parts))
        from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.virtual_partition import (
            PandasOnRayDataframeVirtualPartition,
        )

        PandasOnRayDataframeVirtualPartition.drain_call_queue = self.dcq
        return final_results
