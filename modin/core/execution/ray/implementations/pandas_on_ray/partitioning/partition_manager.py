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

"""Module houses class that implements ``GenericRayDataframePartitionManager`` using Ray."""

import numpy as np
from pandas.core.dtypes.common import is_numeric_dtype

from modin.config import AsyncReadMode
from modin.core.execution.modin_aqp import progress_bar_wrapper
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.generic.partitioning import (
    GenericRayDataframePartitionManager,
)
from modin.logging import get_logger
from modin.utils import _inherit_docstrings

from .partition import PandasOnRayDataframePartition
from .virtual_partition import (
    PandasOnRayDataframeColumnPartition,
    PandasOnRayDataframeRowPartition,
)


class PandasOnRayDataframePartitionManager(GenericRayDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayDataframePartition
    _column_partitions_class = PandasOnRayDataframeColumnPartition
    _row_partition_class = PandasOnRayDataframeRowPartition
    _execution_wrapper = RayWrapper
    materialize_futures = RayWrapper.materialize

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
        RayWrapper.wait(
            [block for partition in partitions for block in partition.list_of_blocks]
        )

    @classmethod
    @_inherit_docstrings(
        GenericRayDataframePartitionManager.split_pandas_df_into_partitions
    )
    def split_pandas_df_into_partitions(
        cls, df, row_chunksize, col_chunksize, update_bar
    ):
        # it was found out, that with the following condition it's more beneficial
        # to use the distributed splitting, let's break them down:
        #   1. The distributed splitting is used only when there's more than 6mln elements
        #   in the `df`, as with fewer data it's better to use the sequential splitting
        #   2. Only used with numerical data, as with other dtypes, putting the whole big
        #   dataframe into the storage takes too much time.
        #   3. The distributed splitting consumes more memory that the sequential one.
        #   It was estimated that it requires ~2.5x of the dataframe size, for now there
        #   was no good way found to automatically fall back to the sequential
        #   implementation in case of not enough memory, so currently we're enabling
        #   the distributed version only if 'AsyncReadMode' is set to True. Follow this
        #   discussion for more info on why automatical dispatching is hard:
        #   https://github.com/modin-project/modin/pull/6640#issuecomment-1759932664
        enough_elements = (len(df) * len(df.columns)) > 6_000_000
        all_numeric_types = all(is_numeric_dtype(dtype) for dtype in df.dtypes)
        async_mode_on = AsyncReadMode.get()

        distributed_splitting = enough_elements and all_numeric_types and async_mode_on

        log = get_logger()

        if not distributed_splitting:
            log.info(
                "Using sequential splitting in '.from_pandas()' because of some of the conditions are False: "
                + f"{enough_elements=}; {all_numeric_types=}; {async_mode_on=}"
            )
            return super().split_pandas_df_into_partitions(
                df, row_chunksize, col_chunksize, update_bar
            )

        log.info("Using distributed splitting in '.from_pandas()'")
        put_func = cls._partition_class.put

        def mask(part, row_loc, col_loc):
            # 2D iloc works surprisingly slow, so doing this chained iloc calls:
            # https://github.com/pandas-dev/pandas/issues/55202
            return part.apply(lambda df: df.iloc[row_loc, :].iloc[:, col_loc])

        main_part = put_func(df)
        parts = [
            [
                update_bar(
                    mask(
                        main_part,
                        slice(i, i + row_chunksize),
                        slice(j, j + col_chunksize),
                    ),
                )
                for j in range(0, len(df.columns), col_chunksize)
            ]
            for i in range(0, len(df), row_chunksize)
        ]
        return np.array(parts)


def _make_wrapped_method(name: str):
    """
    Define new attribute that should work with progress bar.

    Parameters
    ----------
    name : str
        Name of `GenericRayDataframePartitionManager` attribute that should be reused.

    Notes
    -----
    - `classmethod` decorator shouldn't be applied twice, so we refer to `__func__` attribute.
    - New attribute is defined for `PandasOnRayDataframePartitionManager`.
    """
    setattr(
        PandasOnRayDataframePartitionManager,
        name,
        classmethod(
            progress_bar_wrapper(
                getattr(GenericRayDataframePartitionManager, name).__func__
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
