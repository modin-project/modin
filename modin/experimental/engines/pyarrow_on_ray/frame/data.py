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

import pandas
from pandas.core.dtypes.cast import find_common_type

from .partition_manager import PyarrowOnRayFrameManager
from modin.engines.base.frame.data import BasePandasFrame

import ray


class PyarrowOnRayFrame(BasePandasFrame):

    _frame_mgr_cls = PyarrowOnRayFrameManager

    def _apply_index_objs(self, axis=None):
        """Lazily applies the index object (Index or Columns) to the partitions.

        Args:
            axis: The axis to apply to, None applies to both axes.

        Returns:
            A new 2D array of partitions that have the index assignment added to the
            call queue.
        """
        self._filter_empties()

    @classmethod
    def combine_dtypes(cls, list_of_dtypes, column_names):
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column.
        dtypes = (
            pandas.concat(ray.get(list_of_dtypes), axis=1)
            .apply(lambda row: find_common_type(row.values), axis=1)
            .squeeze(axis=0)
        )
        dtypes.index = column_names
        return dtypes

    def to_pandas(self):
        df = super(PyarrowOnRayFrame, self).to_pandas()
        df.index = self.index
        df.columns = self.columns
        return df
