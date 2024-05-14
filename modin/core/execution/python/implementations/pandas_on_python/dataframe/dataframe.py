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
Module contains class ``PandasOnPythonDataframe``.

``PandasOnPythonDataframe`` is dataframe class with pandas storage format and Python engine.
"""

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe

from ..partitioning.partition_manager import PandasOnPythonDataframePartitionManager


class PandasOnPythonDataframe(PandasDataframe):
    """
    Class for dataframes with pandas storage format and Python engine.

    ``PandasOnPythonDataframe`` doesn't implement any specific interfaces,
    all functionality is inherited from the ``PandasDataframe`` class.

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
    pandas_backend : {"pyarrow", None}, optional
        Backend used by pandas. None - means default NumPy backend.
    """

    _partition_mgr_cls = PandasOnPythonDataframePartitionManager
