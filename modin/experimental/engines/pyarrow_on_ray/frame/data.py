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
Module contains class `PyarrowOnRayFrame`.

`PyarrowOnRayFrame` is dataframe class with PyArrow backend and Ray engine.
"""

import pandas
from pandas.core.dtypes.cast import find_common_type

from .partition_manager import PyarrowOnRayFrameManager
from modin.engines.base.frame.data import BasePandasFrame

import ray


class PyarrowOnRayFrame(BasePandasFrame):
    """
    Class for dataframes with PyArrow backend and Ray engine.

    `PyarrowOnRayFrame` implements specific for PyArrow and Ray interfaces,
    other functionality is inherited from the `BasePandasFrame` class.

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

    _frame_mgr_cls = PyarrowOnRayFrameManager

    def _apply_index_objs(self, axis=None):
        """
        Lazily apply the index object (Index or Columns) to the partitions.

        Parameters
        ----------
        axis : int, optional
            Parameter is deprecated and affects nothing.
        """
        self._filter_empties()

    @classmethod
    def combine_dtypes(cls, list_of_dtypes, column_names):
        """
        Get common for all partitions dtype for each of the columns.

        Parameters
        ----------
        list_of_dtypes : array-like
            Array with references to the partitions dtypes objects.
        column_names : array-like or pandas.Index
            Columns names to use for resulting Series.

        Returns
        -------
        pandas.Series
            pandas.Series where index is columns names and values are
            columns dtypes.

        Notes
        -----
        The reported dtypes from differing rows can be different based
        on the inference in the limited data seen by each worker. We
        use pandas to compute the exact dtype over the whole column for
        each column.
        """
        dtypes = (
            pandas.concat(ray.get(list_of_dtypes), axis=1)
            .apply(lambda row: find_common_type(row.values), axis=1)
            .squeeze(axis=0)
        )
        dtypes.index = column_names
        return dtypes

    def to_pandas(self):
        """
        Convert frame object to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        df = super(PyarrowOnRayFrame, self).to_pandas()
        df.index = self.index
        df.columns = self.columns
        return df
