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

"""
Module contains class ``PyarrowOnRayDataframe``.

``PyarrowOnRayDataframe`` is a dataframe class with PyArrow storage format and Ray engine.
"""

from ..partitioning.partition_manager import PyarrowOnRayDataframePartitionManager
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe


class PyarrowOnRayDataframe(PandasDataframe):
    """
    Class for dataframes with PyArrow storage format and Ray engine.

    ``PyarrowOnRayDataframe`` implements interfaces specific for PyArrow and Ray,
    other functionality is inherited from the ``PandasDataframe`` class.

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

    _partition_mgr_cls = PyarrowOnRayDataframePartitionManager

    def synchronize_labels(self, axis=None):
        """
        Synchronize labels by applying the index object (Index or Columns) to the partitions lazily.

        Parameters
        ----------
        axis : {0, 1}, optional
            Parameter is deprecated and affects nothing.
        """
        self._filter_empties()

    def to_pandas(self):
        """
        Convert frame object to a ``pandas.DataFrame``.

        Returns
        -------
        pandas.DataFrame
        """
        df = super(PyarrowOnRayDataframe, self).to_pandas()
        df.index = self.index
        df.columns = self.columns
        return df
