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

"""Module houses class that implements ``GenericUnidistDataframePartitionManager`` using Unidist."""

from modin.core.execution.modin_aqp import progress_bar_wrapper
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.generic.partitioning import (
    GenericUnidistDataframePartitionManager,
)

from .partition import PandasOnUnidistDataframePartition
from .virtual_partition import (
    PandasOnUnidistDataframeColumnPartition,
    PandasOnUnidistDataframeRowPartition,
)


class PandasOnUnidistDataframePartitionManager(GenericUnidistDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    # This object uses PandasOnUnidistDataframePartition objects as the underlying store.
    _partition_class = PandasOnUnidistDataframePartition
    _column_partitions_class = PandasOnUnidistDataframeColumnPartition
    _row_partition_class = PandasOnUnidistDataframeRowPartition
    _execution_wrapper = UnidistWrapper

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions` in parallel, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.
        """
        UnidistWrapper.wait(
            [block for partition in partitions for block in partition.list_of_blocks]
        )


def _make_wrapped_method(name: str):
    """
    Define new attribute that should work with progress bar.

    Parameters
    ----------
    name : str
        Name of `GenericUnidistDataframePartitionManager` attribute that should be reused.

    Notes
    -----
    - `classmethod` decorator shouldn't be applied twice, so we refer to `__func__` attribute.
    - New attribute is defined for `PandasOnUnidistDataframePartitionManager`.
    """
    setattr(
        PandasOnUnidistDataframePartitionManager,
        name,
        classmethod(
            progress_bar_wrapper(
                getattr(GenericUnidistDataframePartitionManager, name).__func__
            )
        ),
    )


for method in (
    "map_partitions",
    "lazy_map_partitions",
    "map_axis_partitions",
    "_apply_func_to_list_of_partitions",
    "apply_func_to_select_indices",
    "apply_func_to_select_indices_along_full_axis",
    "apply_func_to_indices_both_axis",
    "n_ary_operation",
):
    _make_wrapped_method(method)
