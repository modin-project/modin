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
from modin.experimental.backends.omnisci.query_compiler import DFAlgQueryCompiler
from .partition_manager import OmnisciOnRayFrameManager

from pandas.core.index import ensure_index

from .algebra import MaskNode, FrameNode

import ray


class OmnisciOnRayFrame(BasePandasFrame):

    _query_compiler_cls = DFAlgQueryCompiler
    _frame_mgr_cls = OmnisciOnRayFrameManager

    def __init__(
        self,
        partitions=None,
        index=None,
        columns=None,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
        op=None,
    ):
        if index is not None:
            index = ensure_index(index)
        columns = ensure_index(columns)
        self._op = op
        self._partitions = partitions
        self._index_cache = index
        self._columns_cache = columns
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        self._dtypes = dtypes
        if self._op is None:
            self._op = FrameNode(self)
        if partitions is not None:
            self._filter_empties()

    def mask(
        self,
        row_indices=None,
        row_numeric_idx=None,
        col_indices=None,
        col_numeric_idx=None,
    ):
        if col_indices:
            new_columns = col_indices
        elif col_numeric_idx is not None:
            new_columns = [self.columns[col_numeric_idx]]
        else:
            new_columns = self.columns

        new_frame = self.__constructor__(columns=new_columns)
        op = MaskNode(
            self,
            row_indices=row_indices,
            row_numeric_idx=row_numeric_idx,
            col_indices=new_columns,
        )

        return self.__constructor__(columns=new_columns, op=op)

    def _execute(self):
        if isinstance(self._op, FrameNode):
            return

        new_partitions = self._frame_mgr_cls.run_exec_plan(self._op)
        self._partitions = new_partitions
        self._op = FrameNode(self)

    def _build_index_cache(self):
        assert isinstance(self._op, FrameNode)
        assert self._partitions.size == 1
        self._index_cache = ray.get(self._partitions[0][0].oid).index

    def _get_index(self):
        self._execute()
        if self._index_cache is None:
            self._build_index_cache()
        return self._index_cache

    def _set_index(self, new_index):
        raise NotImplementedError("OmnisciOnRayFrame._set_index is not yet suported")

    def _set_columns(self, new_columns):
        raise NotImplementedError("OmnisciOnRayFrame._set_columns is not yet suported")

    # columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    # @classmethod
    # def from_pandas(cls, df):
    #    return super().from_pandas(df)
