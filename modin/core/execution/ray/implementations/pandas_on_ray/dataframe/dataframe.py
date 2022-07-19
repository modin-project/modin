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

from ..partitioning.partition_manager import PandasOnRayDataframePartitionManager
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe


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

    def _get_partition_size_along_axis(self, partition, axis=0):
        """
        Compute the length along the specified axis of the specified partition.

        Parameters
        ----------
        partition : ``PandasOnRayDataframeVirtualPartition`` or ``PandasOnRayDataframePartition``
            The partition whose size to compute.
        axis : int, default: 0
            The axis along which to compute size.

        Returns
        -------
        list
            A list of Ray object IDs representing lengths along the specified axis that sum to the overall length of the partition
            along the specified axis.

        Notes
        -----
        This utility function is used to ensure that computation occurs asynchronously across all partitions
        whether the partitions are virtual or physical partitions.
        """

        def len_fn(df):
            return len(df) if not axis else len(df.columns)

        if isinstance(partition, self._partition_mgr_cls._partition_class):
            return [partition.apply(len_fn)._data]
        elif partition.axis == axis:
            return [
                ptn.apply(len_fn)._data
                for ptn in partition.list_of_partitions_to_combine
            ]
        return [partition.list_of_partitions_to_combine[0].apply(len_fn)._data]

    def _ray_get_nested(self, ray_list):
        """
        Get the result of computations of a nested list of Ray object IDs, and returns a nested same dimension.

        For example, calling `_ray_get_nested([id1, [id2], id3])` would return a list of the form `[val1, [val2], val3]`.

        This function does not work for lists that are nested more than 3 layers (e.g. `[[[id]]]`).

        Parameters
        ----------
        ray_list : list
            A 2D list of Ray object IDs.

        Returns
        -------
        list
            A 2D list of computed values corresponding to the passed in object IDs, with the same
            structure as the list that was passed in.
        """
        # Lengths of lists in original `ray_list`, or -1 if just a single item and not a list
        lens = []
        flat_obj_ids = []
        for lst_or_id in ray_list:
            if isinstance(lst_or_id, list):
                lens.append(len(lst_or_id))
                flat_obj_ids.extend(lst_or_id)
            else:
                lens.append(-1)
                flat_obj_ids.append(lst_or_id)
        flat_values = ray.get(flat_obj_ids)
        nested_values = []
        flat_index = 0
        for length in lens:
            if length == -1:
                # Original list had a single element here
                nested_values.append(flat_values[flat_index])
                flat_index += 1
            else:
                # Original list had a nested list here
                nested_values.append(flat_values[flat_index : flat_index + length])
                flat_index += length
        return nested_values

    @property
    def _row_lengths(self):
        """
        Compute the row partitions lengths if they are not cached.

        Returns
        -------
        list
            A list of row partitions lengths.
        """
        if self._row_lengths_cache is None:
            row_lengths_list = self._ray_get_nested(
                [
                    self._get_partition_size_along_axis(obj, axis=0)
                    for obj in self._partitions.T[0]
                ]
            )
            self._row_lengths_cache = [sum(len_list) for len_list in row_lengths_list]
        return self._row_lengths_cache

    @property
    def _column_widths(self):
        """
        Compute the column partitions widths if they are not cached.

        Returns
        -------
        list
            A list of column partitions widths.
        """
        if self._column_widths_cache is None:
            col_widths_list = self._ray_get_nested(
                [
                    self._get_partition_size_along_axis(obj, axis=1)
                    for obj in self._partitions[0]
                ]
            )
            self._column_widths_cache = [
                sum(width_list) for width_list in col_widths_list
            ]
        return self._column_widths_cache
