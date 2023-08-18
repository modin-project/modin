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
Module contains ``PyarrowQueryCompiler`` class.

``PyarrowQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for the ``PyarrowOnRayDataframe``.
"""

import pandas

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.utils import _inherit_docstrings


class FakeSeries:
    """
    Series metadata class.

    Parameters
    ----------
    dtype : dtype
        Data-type of the represented Series.
    """

    def __init__(self, dtype):
        self.dtype = dtype


@_inherit_docstrings(PandasQueryCompiler)
class PyarrowQueryCompiler(PandasQueryCompiler):
    """
    Query compiler for the PyArrow storage format.

    This class translates common query compiler API into the DataFrame Algebra
    queries, that is supposed to be executed by
    :py:class:`~modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.dataframe.dataframe.PyarrowOnRayDataframe`.

    Parameters
    ----------
    modin_frame : PyarrowOnRayDataframe
        Modin Frame to query with the compiled queries.
    shape_hint : {"row", "column", None}, default: None
        Shape hint for frames known to be a column or a row, otherwise None.
    """

    def _compute_index(self, axis, data_object, compute_diff=True):
        """
        Compute index labels of the passed Modin Frame along specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to compute index labels along. 0 is for index and 1 is for column.
        data_object : PyarrowOnRayDataframe
            Modin Frame object to build indices from.
        compute_diff : bool, default: True
            Whether to cut the resulted indices to a subset of the self indices.

        Returns
        -------
        pandas.Index
        """

        def arrow_index_extraction(table, axis):
            """Extract index labels from the passed pyarrow table the along specified axis."""
            if not axis:
                return pandas.Index(table.column(table.num_columns - 1))
            else:
                try:
                    return pandas.Index(table.columns)
                except AttributeError:
                    return []

        index_obj = self.index if not axis else self.columns
        old_blocks = self.data if compute_diff else None
        # FIXME: `PandasDataframe.get_indices` was deprecated, this call should be
        # replaced either by `PandasDataframe._compute_axis_label` or by `PandasDataframe.axes`.
        new_indices, _ = data_object.get_indices(
            axis=axis,
            index_func=lambda df: arrow_index_extraction(df, axis),
            old_blocks=old_blocks,
        )
        return index_obj[new_indices] if compute_diff else new_indices
