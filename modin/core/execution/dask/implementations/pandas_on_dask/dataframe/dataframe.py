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

"""Module houses class that implements ``PandasDataframe``."""

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from ..partitioning.partition_manager import PandasOnDaskDataframePartitionManager


class PandasOnDaskDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe``.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a pandas.Index.
    columns : sequence
        The columns object for the dataframe. Converted to a pandas.Index.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """

    _partition_mgr_cls = PandasOnDaskDataframePartitionManager

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
            self._row_lengths_cache = (
                self._partition_mgr_cls.get_objects_from_partitions(
                    [obj.apply(lambda df: len(df)) for obj in self._partitions.T[0]]
                )
            )
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
            self._column_widths_cache = (
                self._partition_mgr_cls.get_objects_from_partitions(
                    [
                        obj.apply(lambda df: len(df.columns))
                        for obj in self._partitions[0]
                    ]
                )
            )
        return self._column_widths_cache
