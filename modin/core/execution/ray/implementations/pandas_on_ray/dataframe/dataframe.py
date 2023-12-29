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

"""Module houses class that implements ``PandasDataframe`` using Ray."""

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.pandas.utils import apply_function_on_selected_items

from ..partitioning.partition_manager import PandasOnRayDataframePartitionManager


class PandasOnRayDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe`` using Ray.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a ``pandas.Index``.
    columns : sequence
        The columns object for the dataframe. Converted to a ``pandas.Index``.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """

    _partition_mgr_cls = PandasOnRayDataframePartitionManager
    _materialize_in_loop = False

    def _get_dimensions(self, parts, dim_name):
        """
        Get list of  dimensions for all the provided parts.

        Parameters
        ----------
        parts : list
            List of parttions.
        dim_name : string
            Dimension name could be "length" or "width".

        Returns
        -------
        list
        """
        dims = [getattr(part, dim_name)(False) for part in parts]
        filter_condition = self._partition_mgr_cls._execution_wrapper.check_is_future
        apply_function_on_selected_items(dims, filter_condition, self.materialize_func)
        return dims
