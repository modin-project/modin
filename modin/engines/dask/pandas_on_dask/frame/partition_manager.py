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

import numpy as np

from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import (
    PandasOnDaskFrameColumnPartition,
    PandasOnDaskFrameRowPartition,
)
from .partition import PandasOnDaskFramePartition
from modin.error_message import ErrorMessage
import pandas

from distributed.client import _get_global_client
import cloudpickle as pkl


def deploy_func(df, apply_func, call_queue_df=None, call_queues_other=None, *others):
    if call_queue_df is not None and len(call_queue_df) > 0:
        for call, kwargs in call_queue_df:
            if isinstance(call, bytes):
                call = pkl.loads(call)
            if isinstance(kwargs, bytes):
                kwargs = pkl.loads(kwargs)
            df = call(df, **kwargs)
    new_others = np.empty(shape=len(others), dtype=object)
    for i, call_queue_other in enumerate(call_queues_other):
        other = others[i]
        if call_queue_other is not None and len(call_queue_other) > 0:
            for call, kwargs in call_queue_other:
                if isinstance(call, bytes):
                    call = pkl.loads(call)
                if isinstance(kwargs, bytes):
                    kwargs = pkl.loads(kwargs)
                other = call(other, **kwargs)
        new_others[i] = other
    if isinstance(apply_func, bytes):
        apply_func = pkl.loads(apply_func)
    return apply_func(df, new_others)


class DaskFrameManager(BaseFrameManager):
    """This class implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnDaskFramePartition
    _column_partitions_class = PandasOnDaskFrameColumnPartition
    _row_partition_class = PandasOnDaskFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func):
        """
        This gets the internal indices stored in the partitions.

        Parameters
        ----------
            axis : 0 or 1
                This axis to extract the labels (0 - index, 1 - columns).
            partitions : NumPy array
                The array of partitions from which need to extract the labels.
            index_func : callable
                The function to be used to extract the function.

        Returns
        -------
        Index
            A Pandas Index object.

        Notes
        -----
        These are the global indices of the object. This is mostly useful
        when you have deleted rows/columns internally, but do not know
        which ones were deleted.
        """
        client = _get_global_client()
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            # We grab the first column of blocks and extract the indices
            new_idx = (
                [idx.apply(func).future for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func).future for idx in partitions[0]]
                if len(partitions)
                else []
            )
        new_idx = client.gather(new_idx)
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
        def mapper(df, others):
            other = pandas.concat(others, axis=axis ^ 1)
            return apply_func(df, **{other_name: other})

        client = _get_global_client()
        return np.array(
            [
                [
                    PandasOnDaskFramePartition(
                        client.submit(
                            deploy_func,
                            part.future,
                            mapper,
                            part.call_queue,
                            [obj[col_idx].call_queue for obj in right]
                            if axis
                            else [obj.call_queue for obj in right[row_idx]],
                            *(
                                [obj[col_idx].future for obj in right]
                                if axis
                                else [obj.future for obj in right[row_idx]]
                            ),
                            pure=False,
                        )
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )
