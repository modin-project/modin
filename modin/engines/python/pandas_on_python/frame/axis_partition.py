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

import pandas

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnPythonFramePartition


class PandasOnPythonFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.data for obj in list_of_blocks]

    partition_type = PandasOnPythonFramePartition
    instance_type = pandas.DataFrame

    @classmethod
    def deploy_axis_func(
        cls,
        axis,
        func,
        num_splits,
        num_objs,
        kwargs,
        maintain_partitioning,
        *partitions,
    ):
        return deploy_python_func(
            PandasFrameAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            num_objs,
            kwargs,
            maintain_partitioning,
            *partitions,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls,
        axis,
        func,
        num_splits,
        num_objs,
        len_of_left,
        other_shape,
        kwargs,
        *partitions,
    ):
        return deploy_python_func(
            PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            num_objs,
            len_of_left,
            other_shape,
            kwargs,
            *partitions,
        )


class PandasOnPythonFrameColumnPartition(PandasOnPythonFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 0


class PandasOnPythonFrameRowPartition(PandasOnPythonFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 1


def deploy_python_func(func, *args):
    result = func(*args)

    def compute_result(result):
        if isinstance(result, pandas.DataFrame):
            return [result, len(result), len(result.columns)]
        elif all(isinstance(r, pandas.DataFrame) for r in result):
            return [i for r in result for i in [r, len(r), len(r.columns)]]
        else:
            return [i for r in result for i in [r, None, None]]

    if isinstance(result, tuple):
        whole_result = []
        for partial_result in result:
            whole_result += compute_result(partial_result)
        return whole_result
    else:
        return compute_result(result)
