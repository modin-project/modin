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

from pandas.core.index import ensure_index, Index

from .df_algebra import (
    MaskNode,
    FrameNode,
    GroupbyAggNode,
    TransformNode,
    UnionNode,
    JoinNode,
)
from .expr import (
    InputRefExpr,
    LiteralExpr,
    OpExprType,
    build_if_then_else,
)
from collections import OrderedDict

import ray
import numpy as np


class OmnisciOnRayFrame(BasePandasFrame):

    _query_compiler_cls = DFAlgQueryCompiler
    _frame_mgr_cls = OmnisciOnRayFrameManager

    _next_id = [1]

    def __init__(
        self,
        partitions=None,
        index=None,
        columns=None,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
        op=None,
        index_cols=None,
    ):
        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1

        if index is not None:
            index = ensure_index(index)
        columns = ensure_index(columns)
        self._op = op
        self._index_cols = index_cols
        self._partitions = partitions
        self._index_cache = index
        self._columns_cache = columns
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        self._dtypes = dtypes
        if self._op is None:
            self._op = FrameNode(self)

        self._table_cols = columns.tolist()
        if self._index_cols is not None:
            self._table_cols = self._index_cols + self._table_cols

        if partitions is not None:
            self._filter_empties()

    def copy(self):
        """Copy this object.

        Returns:
            A copied version of this object.
        """
        index_copy = None
        if self._index_cache is not None:
            index_copy = self._index_cache.copy()

        return self.__constructor__(
            self._partitions,
            index_copy,
            self.columns.copy(),
            self._row_lengths,
            self._column_widths,
            self._dtypes,
            op=self._op,
            index_cols=self._index_cols,
        )

    def id_str(self):
        return f"frame${self.id}"

    def ref(self, col):
        return InputRefExpr(self, col)

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
            new_columns = self.columns[col_numeric_idx]
        else:
            new_columns = self.columns

        op = MaskNode(
            self,
            row_indices=row_indices,
            row_numeric_idx=row_numeric_idx,
            col_indices=new_columns,
        )

        return self.__constructor__(
            columns=new_columns, op=op, index_cols=self._index_cols
        )

    def groupby_agg(self, by, axis, agg, groupby_args, **kwargs):
        # Currently we only expect by to be a projection of the same frame
        if not isinstance(by, DFAlgQueryCompiler):
            raise NotImplementedError("unsupported groupby args")

        if axis != 0:
            raise NotImplementedError("groupby is supported for axis = 0 only")

        mask = by._modin_frame._op
        if not isinstance(mask, MaskNode):
            raise NotImplementedError("unsupported groupby args")

        if mask.input[0] != self:
            raise NotImplementedError("unsupported groupby args")

        if mask.row_indices is not None or mask.row_numeric_idx is not None:
            raise NotImplementedError("unsupported groupby args")

        if groupby_args["level"] is not None:
            raise NotImplementedError("levels are not supported for groupby")

        groupby_cols = by._modin_frame.columns
        new_columns = []
        index_cols = None

        if groupby_args["as_index"]:
            index_cols = groupby_cols.tolist()
        else:
            new_columns = groupby_cols.tolist()

        if isinstance(agg, str):
            new_agg = {}
            for col in self.columns:
                if col not in groupby_cols:
                    new_agg[col] = agg
                    new_columns.append(col)
            agg = new_agg
        else:
            for k, v in agg.items():
                if isinstance(v, list):
                    # TODO: support levels
                    new_columns.append(k + " " + v)
                else:
                    new_columns.append(k)
        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)

        new_op = GroupbyAggNode(self, groupby_cols, agg, groupby_args)
        new_frame = self.__constructor__(
            columns=new_columns, op=new_op, index_cols=index_cols
        )

        return new_frame

    def fillna(
        self, value=None, method=None, axis=None, limit=None, downcast=None,
    ):
        if axis != 0:
            raise NotImplementedError("fillna is supported for axis = 0 only")

        if limit is not None:
            raise NotImplementedError("fillna doesn't support limit yet")

        if downcast is not None:
            raise NotImplementedError("fillna doesn't support downcast yet")

        if method is not None:
            raise NotImplementedError("fillna doesn't support method yet")

        exprs = {}
        if isinstance(value, dict):
            for col in self.columns:
                col_expr = self.ref(col)
                if col in value:
                    value_expr = LiteralExpr(value[col])
                    res_type = OpExprType(type(value[col]), False)
                    exprs[col] = build_if_then_else(
                        col_expr.is_null(), value_expr, col_expr, res_type
                    )
                else:
                    exprs[col] = col_expr
        elif np.isscalar(value):
            value_expr = LiteralExpr(value)
            res_type = OpExprType(type(value), False)
            for col in self.columns:
                col_expr = self.ref(col)
                exprs[col] = build_if_then_else(
                    col_expr.is_null(), value_expr, col_expr, res_type
                )
        else:
            raise NotImplementedError("unsupported value for fillna")

        new_op = TransformNode(self, exprs)
        new_frame = self.__constructor__(
            columns=self.columns, op=new_op, index_cols=self._index_cols
        )

        return new_frame

    def join(self, other, how="inner", on=None, sort=False, suffixes=("_x", "_y")):
        assert (
            on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        assert (
            on in self.columns and on in other.columns
        ), "Only cases when both frames contain key column are supported"

        new_columns = []
        new_columns.append(on)

        conflicting_list = list(set(self.columns) & set(other.columns))
        for c in self.columns:
            if c != on:
                suffix = suffixes[0] if c in conflicting_list else ""
                new_columns.append(c + suffix)
        for c in other.columns:
            if c != on:
                suffix = suffixes[1] if c in conflicting_list else ""
                new_columns.append(c + suffix)

        op = JoinNode(self, other, how=how, on=on, sort=sort, suffixes=suffixes,)

        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)
        return self.__constructor__(columns=new_columns, op=op)

    def _index_width(self):
        if self._index_cols is None:
            return 1
        return len(self._index_cols)

    def _concat(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        assert axis == 0, "unsupported concat on axis = 1"

        # determine output columns
        new_columns = OrderedDict()
        for col in self.columns:
            new_columns[col] = 1
        for frame in other_modin_frames:
            if join == "inner":
                for col in list(new_columns):
                    if col not in frame.columns:
                        del new_columns[col]
            else:
                for col in frame.columns:
                    if col not in new_columns:
                        new_columns[col] = 1
        new_columns = list(new_columns.keys())

        if sort:
            new_columns = sorted(new_columns)

        # determine how many index components are going into
        # the resulting table
        if not ignore_index:
            index_width = self._index_width()
            for frame in other_modin_frames:
                index_width = min(index_width, frame._index_width())

        # build projections to align all frames
        aligned_frames = []
        for frame in [self] + other_modin_frames:
            aligned_index = None
            exprs = {}
            if not ignore_index:
                if frame._index_cols:
                    aligned_index = frame._index_cols[0 : index_width + 1]
                    for i in range(0, index_width):
                        col = frame._index_cols[i]
                        exprs[col] = frame.ref(col)
                else:
                    assert index_width == 1, "unexpected index width"
                    aligned_index = ["__index__"]
                    exprs["__index__"] = frame.ref("__rowid__")
            for col in new_columns:
                if col in frame._table_cols:
                    exprs[col] = frame.ref(col)
                else:
                    exprs[col] = LiteralExpr(None)
            aligned_frame_op = TransformNode(frame, exprs, False)
            aligned_frames.append(
                self.__constructor__(
                    columns=new_columns, op=aligned_frame_op, index_cols=aligned_index
                )
            )

        new_op = UnionNode(aligned_frames)
        return self.__constructor__(
            columns=new_columns, op=new_op, index_cols=aligned_frames[0]._index_cols,
        )

    def _execute(self):
        if isinstance(self._op, FrameNode):
            return

        new_partitions = self._frame_mgr_cls.run_exec_plan(self._op, self._index_cols)
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

    def to_pandas(self):
        self._execute()
        return super(OmnisciOnRayFrame, self).to_pandas()

    # @classmethod
    # def from_pandas(cls, df):
    #    return super().from_pandas(df)
