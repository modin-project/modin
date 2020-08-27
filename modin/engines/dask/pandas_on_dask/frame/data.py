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

from modin.engines.base.frame.data import BasePandasFrame
from .partition_manager import DaskFrameManager

from distributed.client import _get_global_client


class PandasOnDaskFrame(BasePandasFrame):

    _frame_mgr_cls = DaskFrameManager

    @property
    def _row_lengths(self):
        """Compute the row lengths if they are not cached.

        Returns:
            A list of row lengths.
        """
        client = _get_global_client()
        if self._row_lengths_cache is None:
            self._row_lengths_cache = client.gather(
                [obj.apply(lambda df: len(df)).future for obj in self._partitions.T[0]]
            )
        return self._row_lengths_cache

    @property
    def _column_widths(self):
        """Compute the column widths if they are not cached.

        Returns:
            A list of column widths.
        """
        client = _get_global_client()
        if self._column_widths_cache is None:
            self._column_widths_cache = client.gather(
                [
                    obj.apply(lambda df: len(df.columns)).future
                    for obj in self._partitions[0]
                ]
            )
        return self._column_widths_cache
