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

"""Module houses ``PandasQueryPipeline`` and ``PandasQuery`` classes, that implement a batch pipeline protocol for Modin Dataframes."""

from typing import Callable, Optional

import numpy as np

import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
    PandasOnRayDataframe,
)
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution


class PandasQuery(object):
    """
    Internal representation of a single query in a pipeline.

    This object represents a single function to be pipelined in a batch pipeline.

    Parameters
    ----------
    func : Callable
        The function to apply to the dataframe.
    is_output : bool, default: False
        Whether this query is an output query and should be passed both to the next query, and
        directly to postprocessing.
    repartition_after : bool, default: False
        Whether to repartition after this query is computed. Currently, repartitioning is only
        supported if there is 1 partition prior to repartitioning.
    fan_out : bool, default: False
        Whether to fan out this node. If True and only 1 partition is passed as input, the partition
        is replicated `PandasQueryPipeline.num_partitions` (default: `NPartitions.get`) times, and
        the function is called on each. The `reduce_fn` must also be specified.
    pass_partition_id : bool, default: False
        Whether to pass the numerical partition id to the query.
    reduce_fn : Callable, default: None
        The reduce function to apply if `fan_out` is set to True. This takes the
        `PandasQueryPipeline.num_partitions` (default: `NPartitions.get`) partitions that result from
        this query, and combines them into 1 partition.
    output_id : int, default: None
            An id to assign to this node if it is an output.

    Notes
    -----
    `func` must be a function that is applied along an axis of the dataframe.

    Use `pandas` for any module level functions inside `func` since it operates directly on
    partitions.
    """

    def __init__(
        self,
        func: Callable,
        is_output: bool = False,
        repartition_after: bool = False,
        fan_out: bool = False,
        pass_partition_id: bool = False,
        reduce_fn: Optional[Callable] = None,
        output_id: Optional[int] = None,
    ):
        self.func = func
        self.is_output = is_output
        self.repartition_after = repartition_after
        self.fan_out = fan_out
        self.pass_partition_id = pass_partition_id
        self.reduce_fn = reduce_fn
        self.output_id = output_id
        # List of sub-queries to feed into this query, if this query is an output node.
        self.operators = None


class PandasQueryPipeline(object):
    """
    Internal representation of a query pipeline.

    This object keeps track of the functions that compose to form a query pipeline.

    Parameters
    ----------
    df : modin.pandas.Dataframe
        The dataframe to perform this pipeline on.
    num_partitions : int, optional
        The number of partitions to maintain for the batched dataframe.
        If not specified, the value is assumed equal to ``NPartitions.get()``.

    Notes
    -----
    Only row-parallel pipelines are supported. All queries will be applied along the row axis.
    """

    def __init__(self, df, num_partitions: Optional[int] = None):
        if get_current_execution() != "PandasOnRay" or (
            not isinstance(df._query_compiler._modin_frame, PandasOnRayDataframe)
        ):  # pragma: no cover
            ErrorMessage.not_implemented(
                "Batch Pipeline API is only implemented for `PandasOnRay` execution."
            )
        ErrorMessage.single_warning(
            "The Batch Pipeline API is an experimental feature and still under development in Modin."
        )
        self.df = df
        self.num_partitions = num_partitions if num_partitions else NPartitions.get()
        self.outputs = []  # List of output queries.
        self.query_list = []  # List of all queries.
        self.is_output_id_specified = (
            False  # Flag to indicate that `output_id` has been specified for a node.
        )

    def update_df(self, df):
        """
        Update the dataframe to perform this pipeline on.

        Parameters
        ----------
        df : modin.pandas.DataFrame
            The new dataframe to perform this pipeline on.
        """
        if get_current_execution() != "PandasOnRay" or (
            not isinstance(df._query_compiler._modin_frame, PandasOnRayDataframe)
        ):  # pragma: no cover
            ErrorMessage.not_implemented(
                "Batch Pipeline API is only implemented for `PandasOnRay` execution."
            )
        self.df = df

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
        Add a query to the current pipeline.

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
            partition is replicated `self.num_partitions` (default: `NPartitions.get`) times,
            and the function is called on each. The `reduce_fn` must also be specified.
        pass_partition_id : bool, default: False
            Whether to pass the numerical partition id to the query.
        reduce_fn : Callable, default: None
            The reduce function to apply if `fan_out` is set to True. This takes the
            `self.num_partitions` (default: `NPartitions.get`) partitions that result from this
            query, and combines them into 1 partition.
        output_id : int, default: None
            An id to assign to this node if it is an output.

        Notes
        -----
        Use `pandas` for any module level functions inside `func` since it operates directly on
        partitions.
        """
        if not is_output and output_id is not None:
            raise ValueError("Output ID cannot be specified for non-output node.")
        if is_output:
            if not self.is_output_id_specified and output_id is not None:
                if len(self.outputs) != 0:
                    raise ValueError("Output ID must be specified for all nodes.")
            if output_id is None and self.is_output_id_specified:
                raise ValueError("Output ID must be specified for all nodes.")
        self.query_list.append(
            PandasQuery(
                func,
                is_output,
                repartition_after,
                fan_out,
                pass_partition_id,
                reduce_fn,
                output_id,
            )
        )
        if is_output:
            self.outputs.append(self.query_list[-1])
            if output_id is not None:
                self.is_output_id_specified = True
            self.outputs[-1].operators = self.query_list[:-1]
            self.query_list = []

    def _complete_nodes(self, list_of_nodes, partitions):
        """
        Run a sub-query end to end.

        Parameters
        ----------
        list_of_nodes : list of PandasQuery
            The functions that compose this query.
        partitions : list of PandasOnRayDataframeVirtualPartition
            The partitions that compose the dataframe that is input to this sub-query.

        Returns
        -------
        list of PandasOnRayDataframeVirtualPartition
            The partitions that result from computing the functions represented by `list_of_nodes`.
        """
        for node in list_of_nodes:
            if node.fan_out:
                if len(partitions) > 1:
                    ErrorMessage.not_implemented(
                        "Fan out is only supported with DataFrames with 1 partition."
                    )
                partitions[0] = partitions[0].force_materialization()
                partition_list = partitions[0].list_of_block_partitions
                partitions[0] = partitions[0].add_to_apply_calls(node.func, 0)
                partitions[0].drain_call_queue(num_splits=1)
                new_dfs = []
                for i in range(1, self.num_partitions):
                    new_dfs.append(
                        type(partitions[0])(
                            partition_list,
                            full_axis=partitions[0].full_axis,
                        ).add_to_apply_calls(node.func, i)
                    )
                    new_dfs[-1].drain_call_queue(num_splits=1)

                def reducer(df):
                    df_inputs = [df]
                    for df in new_dfs:
                        df_inputs.append(df.to_pandas())
                    return node.reduce_fn(df_inputs)

                partitions = [partitions[0].add_to_apply_calls(reducer)]
            elif node.repartition_after:
                if len(partitions) > 1:
                    ErrorMessage.not_implemented(
                        "Dynamic repartitioning is currently only supported for DataFrames with 1 partition."
                    )
                partitions[0] = (
                    partitions[0].add_to_apply_calls(node.func).force_materialization()
                )
                new_dfs = []

                def mask_partition(df, i):  # pragma: no cover
                    new_length = len(df.index) // self.num_partitions
                    if i == self.num_partitions - 1:
                        return df.iloc[i * new_length :]
                    return df.iloc[i * new_length : (i + 1) * new_length]

                for i in range(self.num_partitions):
                    new_dfs.append(
                        type(partitions[0])(
                            partitions[0].list_of_block_partitions,
                            full_axis=partitions[0].full_axis,
                        ).add_to_apply_calls(mask_partition, i)
                    )
                partitions = new_dfs
            else:
                if node.pass_partition_id:
                    partitions = [
                        part.add_to_apply_calls(node.func, i)
                        for i, part in enumerate(partitions)
                    ]
                else:
                    partitions = [
                        part.add_to_apply_calls(node.func) for part in partitions
                    ]
        return partitions

    def compute_batch(
        self,
        postprocessor: Optional[Callable] = None,
        pass_partition_id: Optional[bool] = False,
        pass_output_id: Optional[bool] = False,
    ):
        """
        Run the completed pipeline + any postprocessing steps end to end.

        Parameters
        ----------
        postprocessor : Callable, default: None
            A postprocessing function to be applied to each output partition.
            The order of arguments passed is `df` (the partition), `output_id`
            (if `pass_output_id=True`), and `partition_id` (if `pass_partition_id=True`).
        pass_partition_id : bool, default: False
            Whether or not to pass the numerical partition id to the postprocessing function.
        pass_output_id : bool, default: False
            Whether or not to pass the output ID associated with output queries to the
            postprocessing function.

        Returns
        -------
        list or dict or DataFrame
            If output ids are specified, a dictionary mapping output id to the resulting dataframe
            is returned, otherwise, a list of the resulting dataframes is returned.
        """
        if len(self.outputs) == 0:
            ErrorMessage.single_warning(
                "No outputs to compute. Returning an empty list. Please specify outputs by calling `add_query` with `is_output=True`."
            )
            return []
        if not self.is_output_id_specified and pass_output_id:
            raise ValueError(
                "`pass_output_id` is set to True, but output ids have not been specified. "
                + "To pass output ids, please specify them using the `output_id` kwarg with pipeline.add_query"
            )
        if self.is_output_id_specified:
            outs = {}
        else:
            outs = []
        modin_frame = self.df._query_compiler._modin_frame
        partitions = modin_frame._partition_mgr_cls.row_partitions(
            modin_frame._partitions
        )
        for node in self.outputs:
            partitions = self._complete_nodes(node.operators + [node], partitions)
            for part in partitions:
                part.drain_call_queue(num_splits=1)
            if postprocessor:
                output_partitions = []
                for partition_id, partition in enumerate(partitions):
                    args = []
                    if pass_output_id:
                        args.append(node.output_id)
                    if pass_partition_id:
                        args.append(partition_id)
                    output_partitions.append(
                        partition.add_to_apply_calls(postprocessor, *args)
                    )
            else:
                output_partitions = [
                    part.add_to_apply_calls(lambda df: df) for part in partitions
                ]
            [
                part.drain_call_queue(num_splits=self.num_partitions)
                for part in output_partitions
            ]  # Ensures our result df is block partitioned.
            if not self.is_output_id_specified:
                outs.append(output_partitions)
            else:
                outs[node.output_id] = output_partitions
        if self.is_output_id_specified:
            final_results = {}
            id_df_iter = outs.items()
        else:
            final_results = [None] * len(outs)
            id_df_iter = enumerate(outs)

        for id, df in id_df_iter:
            partitions = []
            for row_partition in df:
                partitions.append(row_partition.list_of_block_partitions)
            partitions = np.array(partitions)
            partition_mgr_class = PandasOnRayDataframe._partition_mgr_cls
            index, internal_rows = partition_mgr_class.get_indices(0, partitions)
            columns, internal_cols = partition_mgr_class.get_indices(1, partitions)
            result_modin_frame = PandasOnRayDataframe(
                partitions,
                index,
                columns,
                row_lengths=list(map(len, internal_rows)),
                column_widths=list(map(len, internal_cols)),
            )
            query_compiler = PandasQueryCompiler(result_modin_frame)
            result_df = pd.DataFrame(query_compiler=query_compiler)
            final_results[id] = result_df

        return final_results
