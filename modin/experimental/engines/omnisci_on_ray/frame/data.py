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

from pandas.core.index import ensure_index, Index, MultiIndex
from pandas.core.dtypes.common import _get_dtype
import pandas as pd

from .df_algebra import (
    MaskNode,
    FrameNode,
    GroupbyAggNode,
    TransformNode,
    UnionNode,
    JoinNode,
    translate_exprs_to_base,
)
from .expr import (
    AggregateExpr,
    InputRefExpr,
    LiteralExpr,
    OpExpr,
    build_if_then_else,
    build_dt_expr,
    _get_common_dtype,
    is_cmp_op,
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
        uses_rowid=False,
    ):
        assert dtypes is not None

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
        if self._op is None:
            self._op = FrameNode(self)

        self._table_cols = columns.tolist()
        if self._index_cols is not None:
            self._table_cols = self._index_cols + self._table_cols

        assert len(dtypes) == len(self._table_cols)
        if isinstance(dtypes, list):
            self._dtypes = pd.Series(dtypes, index=self._table_cols)
        else:
            self._dtypes = dtypes

        if partitions is not None and self._index_cols is not None:
            self._filter_empties()

        self._uses_rowid = uses_rowid

    def id_str(self):
        return f"frame${self.id}"

    def ref(self, col):
        if col == "__rowid__":
            return InputRefExpr(self, col, _get_dtype(int))
        return InputRefExpr(self, col, self._dtypes[col])

    def mask(
        self,
        row_indices=None,
        row_numeric_idx=None,
        col_indices=None,
        col_numeric_idx=None,
    ):
        base = self

        if col_indices is not None or col_numeric_idx is not None:
            if col_indices is not None:
                new_columns = col_indices
            elif col_numeric_idx is not None:
                new_columns = base.columns[col_numeric_idx]
            exprs = self._index_exprs()
            for col in new_columns:
                exprs[col] = base.ref(col)
            dtypes = self._dtypes_for_exprs(exprs)
            base = self.__constructor__(
                columns=new_columns,
                dtypes=dtypes,
                op=TransformNode(base, exprs),
                index_cols=self._index_cols,
            )

        if row_indices is not None or row_numeric_idx is not None:
            op = MaskNode(
                base, row_indices=row_indices, row_numeric_idx=row_numeric_idx,
            )
            return self.__constructor__(
                columns=base.columns,
                dtypes=base._dtypes,
                op=op,
                index_cols=self._index_cols,
            )

        return base

    def _dtypes_for_cols(self, new_index, new_columns):
        if new_index is not None:
            res = self._dtypes[
                new_index
                + (
                    new_columns
                    if isinstance(new_columns, list)
                    else new_columns.to_list()
                )
            ]
        else:
            res = self._dtypes[new_columns]
        return res

    def _dtypes_for_exprs(self, exprs):
        return [expr._dtype for expr in exprs.values()]

    def groupby_agg(self, by, axis, agg, groupby_args, **kwargs):
        # Currently we only expect by to be a projection of the same frame
        if not isinstance(by, DFAlgQueryCompiler):
            raise NotImplementedError("unsupported groupby args")

        if axis != 0:
            raise NotImplementedError("groupby is supported for axis = 0 only")

        by_frame = by._modin_frame
        base = by_frame._find_common_projections_base(self)
        if base is None:
            raise NotImplementedError("unsupported groupby args")

        if groupby_args["level"] is not None:
            raise NotImplementedError("levels are not supported for groupby")

        groupby_cols = by.columns.tolist()
        agg_cols = [col for col in self.columns if col not in by.columns]

        # Create new base where all required columns are computed. We don't allow
        # complex expressions to be a group key or an aggeregate operand.
        assert isinstance(by_frame._op, TransformNode), "unexpected by_frame"
        exprs = OrderedDict(((col, by_frame.ref(col)) for col in groupby_cols))
        exprs.update(((col, self.ref(col)) for col in agg_cols))
        exprs = translate_exprs_to_base(exprs, base)
        base_cols = Index.__new__(
            Index, data=list(exprs.keys()), dtype=self.columns.dtype
        )
        base = self.__constructor__(
            columns=base_cols,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(base, exprs, fold=True),
            index_cols=None,
        )

        new_columns = []
        index_cols = None

        if groupby_args["as_index"]:
            index_cols = groupby_cols.copy()
        else:
            new_columns = groupby_cols.copy()

        new_dtypes = by_frame._dtypes[groupby_cols].tolist()

        agg_exprs = OrderedDict()
        if isinstance(agg, str):
            for col in agg_cols:
                agg_exprs[col] = AggregateExpr(agg, base.ref(col))
        else:
            assert isinstance(agg, dict), "unsupported aggregate type"
            for k, v in agg.items():
                if isinstance(v, list):
                    # TODO: support levels
                    for item in v:
                        agg_exprs[k + " " + item] = AggregateExpr(item, base.ref(k))
                else:
                    agg_exprs[k] = AggregateExpr(v, base.ref(k))
        new_columns.extend(agg_exprs.keys())
        new_dtypes.extend((x._dtype for x in agg_exprs.values()))
        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)

        new_op = GroupbyAggNode(base, groupby_cols, agg_exprs, groupby_args)
        new_frame = self.__constructor__(
            columns=new_columns, dtypes=new_dtypes, op=new_op, index_cols=index_cols
        )

        # When 'by' columns do not become a new index, we need to filter out those
        # columns, which are not simple input refs.
        if not groupby_args["as_index"] and any(
            (not by_frame._op.is_original_ref(col) for col in groupby_cols)
        ):
            output_columns = [
                col
                for col in new_columns
                if col not in by_frame.columns or by_frame._op.is_original_ref(col)
            ]
            return new_frame.mask(col_indices=output_columns)

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

        exprs = self._index_exprs()
        if isinstance(value, dict):
            for col in self.columns:
                col_expr = self.ref(col)
                if col in value:
                    value_expr = LiteralExpr(value[col])
                    res_type = _get_common_dtype(value_expr._dtype, col_expr._dtype)
                    exprs[col] = build_if_then_else(
                        col_expr.is_null(), value_expr, col_expr, res_type
                    )
                else:
                    exprs[col] = col_expr
        elif np.isscalar(value):
            value_expr = LiteralExpr(value)
            for col in self.columns:
                col_expr = self.ref(col)
                res_type = _get_common_dtype(value_expr._dtype, col_expr._dtype)
                exprs[col] = build_if_then_else(
                    col_expr.is_null(), value_expr, col_expr, res_type
                )
        else:
            raise NotImplementedError("unsupported value for fillna")

        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(exprs)
        new_frame = self.__constructor__(
            columns=self.columns, dtypes=dtypes, op=new_op, index_cols=self._index_cols
        )

        return new_frame

    def dt_extract(self, obj):
        exprs = self._index_exprs()
        for col in self.columns:
            col_expr = self.ref(col)
            exprs[col] = build_dt_expr(obj, self.ref(col))
        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(exprs)
        return self.__constructor__(
            columns=self.columns, dtypes=dtypes, op=new_op, index_cols=self._index_cols
        )

    def astype(self, col_dtypes, **kwargs):
        columns = col_dtypes.keys()
        new_dtypes = self.dtypes.copy()
        for column in columns:
            dtype = col_dtypes[column]
            if (
                not isinstance(dtype, type(self.dtypes[column]))
                or dtype != self.dtypes[column]
            ):
                # Update the new dtype series to the proper pandas dtype
                try:
                    new_dtype = np.dtype(dtype)
                except TypeError:
                    new_dtype = dtype

                if dtype != np.int32 and new_dtype == np.int32:
                    new_dtypes[column] = np.dtype("int64")
                elif dtype != np.float32 and new_dtype == np.float32:
                    new_dtypes[column] = np.dtype("float64")
                # We cannot infer without computing the dtype if
                elif isinstance(new_dtype, str) and new_dtype == "category":
                    raise NotImplementedError("unsupported type conversion")
                else:
                    new_dtypes[column] = new_dtype
        exprs = self._index_exprs()
        for col in self.columns:
            col_expr = self.ref(col)
            if col in columns:
                exprs[col] = col_expr.cast(new_dtypes[col])
            else:
                exprs[col] = col_expr

        new_op = TransformNode(self, exprs)
        return self.__constructor__(
            columns=self.columns,
            dtypes=new_dtypes,
            op=new_op,
            index_cols=self._index_cols,
        )

    def join(self, other, how="inner", on=None, sort=False, suffixes=("_x", "_y")):
        assert (
            on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        for col in on:
            assert (
                col in self.columns and col in other.columns
            ), "Only cases when both frames contain key column are supported"

        new_columns = []
        new_dtypes = []

        conflicting_cols = set(self.columns) & set(other.columns) - set(on)
        for c in self.columns:
            suffix = suffixes[0] if c in conflicting_cols else ""
            new_columns.append(c + suffix)
            new_dtypes.append(self._dtypes[c])
        for c in other.columns:
            if c not in on:
                suffix = suffixes[1] if c in conflicting_cols else ""
                new_columns.append(c + suffix)
                new_dtypes.append(other._dtypes[c])

        op = JoinNode(self, other, how=how, on=on, sort=sort, suffixes=suffixes,)

        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)
        return self.__constructor__(dtypes=new_dtypes, columns=new_columns, op=op)

    def _index_width(self):
        if self._index_cols is None:
            return 1
        return len(self._index_cols)

    def _union_all(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        # determine output columns
        new_cols_map = OrderedDict()
        for col in self.columns:
            new_cols_map[col] = self._dtypes[col]
        for frame in other_modin_frames:
            if join == "inner":
                for col in list(new_cols_map):
                    if col not in frame.columns:
                        del new_cols_map[col]
            else:
                for col in frame.columns:
                    if col not in new_cols_map:
                        new_cols_map[col] = frame._dtypes[col]
        new_columns = list(new_cols_map.keys())

        if sort:
            new_columns = sorted(new_columns)

        # determine how many index components are going into
        # the resulting table
        if not ignore_index:
            index_width = self._index_width()
            for frame in other_modin_frames:
                index_width = min(index_width, frame._index_width())

        # compute resulting dtypes
        if sort:
            new_dtypes = [new_cols_map[col] for col in new_columns]
        else:
            new_dtypes = list(new_cols_map.values())

        # build projections to align all frames
        aligned_frames = []
        for frame in [self] + other_modin_frames:
            aligned_index = None
            exprs = OrderedDict()
            uses_rowid = False

            if not ignore_index:
                if frame._index_cols:
                    aligned_index = frame._index_cols[0 : index_width + 1]
                    aligned_index_dtypes = frame._dtypes[aligned_index].tolist()
                    for i in range(0, index_width):
                        col = frame._index_cols[i]
                        exprs[col] = frame.ref(col)
                else:
                    assert index_width == 1, "unexpected index width"
                    aligned_index = ["__index__"]
                    exprs["__index__"] = frame.ref("__rowid__")
                    aligned_index_dtypes = [_get_dtype(int)]
                    uses_rowid = True
                aligned_dtypes = aligned_index_dtypes + new_dtypes
            else:
                aligned_dtypes = new_dtypes

            for col in new_columns:
                if col in frame._table_cols:
                    exprs[col] = frame.ref(col)
                else:
                    exprs[col] = LiteralExpr(None)

            aligned_frame_op = TransformNode(frame, exprs)
            aligned_frames.append(
                self.__constructor__(
                    columns=new_columns,
                    dtypes=aligned_dtypes,
                    op=aligned_frame_op,
                    index_cols=aligned_index,
                    uses_rowid=uses_rowid,
                )
            )

        new_frame = aligned_frames[0]
        for frame in aligned_frames[1:]:
            new_frame = self.__constructor__(
                columns=new_columns,
                dtypes=new_frame._dtypes,
                op=UnionNode([new_frame, frame]),
                index_cols=new_frame._index_cols,
            )

        return new_frame

    def _concat(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        if axis == 0:
            return self._union_all(axis, other_modin_frames, join, sort, ignore_index)

        base = self
        for frame in other_modin_frames:
            base = base._find_common_projections_base(frame)
            if base is None:
                raise NotImplementedError("concat requiring join is not supported yet")

        exprs = self._index_exprs()
        new_columns = self.columns.tolist()
        for col in self.columns:
            exprs[col] = self.ref(col)
        for frame in other_modin_frames:
            for col in frame.columns:
                if col == "" or col in exprs:
                    new_col = f"__col{len(exprs)}__"
                else:
                    new_col = col
                exprs[new_col] = frame.ref(col)
                new_columns.append(new_col)

        exprs = translate_exprs_to_base(exprs, base)
        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)
        new_frame = self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(base, exprs),
            index_cols=self._index_cols,
        )
        return new_frame

    def bin_op(self, other, op_name, **kwargs):
        if isinstance(other, (int, float)):
            value_expr = LiteralExpr(other)
            exprs = self._index_exprs()
            for col in self.columns:
                exprs[col] = self.ref(col).bin_op(value_expr, op_name)
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=self._index_cols,
            )
        elif isinstance(other, list):
            if len(other) != len(self.columns):
                raise ValueError(
                    f"length must be {len(self.columns)}: given {len(other)}"
                )
            exprs = self._index_exprs()
            for col, val in zip(self.columns, other):
                exprs[col] = self.ref(col).bin_op(LiteralExpr(val), op_name)
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=self._index_cols,
            )
        elif isinstance(other, type(self)):
            # For now we only support binary operations on
            # projections of the same frame, because we have
            # no support for outer join.
            base = self._find_common_projections_base(other)
            if base is None:
                raise NotImplementedError(
                    "unsupported binary op args (outer join is not supported)"
                )

            new_columns = self.columns.tolist()
            for col in other.columns:
                if col not in self.columns:
                    new_columns.append(col)
            new_columns = sorted(new_columns)

            fill_value = kwargs.get("fill_value", None)
            if fill_value is not None:
                fill_value = LiteralExpr(fill_value)
            if is_cmp_op(op_name):
                null_value = LiteralExpr(op_name == "ne")
            else:
                null_value = LiteralExpr(None)

            exprs = self._index_exprs()
            for col in new_columns:
                lhs = self.ref(col) if col in self.columns else fill_value
                rhs = other.ref(col) if col in other.columns else fill_value
                if lhs is None or rhs is None:
                    exprs[col] = null_value
                else:
                    exprs[col] = lhs.bin_op(rhs, op_name)

            exprs = translate_exprs_to_base(exprs, base)
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(base, exprs),
                index_cols=self._index_cols,
            )

    def insert(self, loc, column, value):
        assert column not in self._table_cols
        assert 0 <= loc <= len(self.columns)

        exprs = self._index_exprs()
        for i in range(0, loc):
            col = self.columns[i]
            exprs[col] = self.ref(col)
        exprs[column] = LiteralExpr(value)
        for i in range(loc, len(self.columns)):
            col = self.columns[i]
            exprs[col] = self.ref(col)

        new_columns = self.columns.insert(loc, column)

        return self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
        )

    def cat_codes(self):
        assert len(self.columns) == 1
        assert self._dtypes[-1] == "category"

        col = self.columns[-1]
        exprs = self._index_exprs()
        col_expr = self.ref(col)
        code_expr = OpExpr("KEY_FOR_STRING", [col_expr], _get_dtype("int32"))
        null_val = LiteralExpr(np.int32(-1))
        exprs[col] = build_if_then_else(
            col_expr.is_null(), null_val, code_expr, _get_dtype("int32")
        )

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes,
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
        )

    def _index_exprs(self):
        exprs = OrderedDict()
        if self._index_cols:
            for col in self._index_cols:
                exprs[col] = self.ref(col)
        return exprs

    def _find_common_projections_base(self, rhs):
        bases = {self}
        while self._is_projection():
            self = self._op.input[0]
            bases.add(self)

        while rhs not in bases and rhs._is_projection():
            rhs = rhs._op.input[0]

        if rhs in bases:
            return rhs

        return None

    def _is_projection(self):
        return isinstance(self._op, TransformNode)

    def _execute(self):
        if isinstance(self._op, FrameNode):
            return

        # Some frames require rowid which is available for executed frames only.
        # Also there is a common pattern when MaskNode is executed to print
        # frame. If we run the whole tree then any following frame usage will
        # require re-compute. So we just execute MaskNode's operands.
        self._run_sub_queries()

        new_partitions = self._frame_mgr_cls.run_exec_plan(
            self._op, self._index_cols, self._dtypes
        )
        self._partitions = new_partitions
        self._op = FrameNode(self)

    def _require_executed_base(self):
        if isinstance(self._op, MaskNode):
            return True
        return self._uses_rowid

    def _run_sub_queries(self):
        if isinstance(self._op, FrameNode):
            return

        if self._require_executed_base():
            for op in self._op.input:
                op._execute()
        else:
            for frame in self._op.input:
                frame._run_sub_queries()

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

    def reset_index(self, drop):
        if drop:
            exprs = OrderedDict()
            for c in self.columns:
                exprs[c] = self.ref(c)
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=None,
            )
        else:
            if self._index_cols is None:
                raise NotImplementedError(
                    "default index reset with no drop is not supported"
                )
            new_columns = Index.__new__(
                Index, data=self._table_cols, dtype=self.columns.dtype
            )
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_cols(None, new_columns),
                op=self._op,
                index_cols=None,
            )

    def _set_columns(self, new_columns):
        exprs = self._index_exprs()
        for old, new in zip(self.columns, new_columns):
            exprs[new] = self.ref(old)
        return self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes.tolist(),
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
        )

    def _get_columns(self):
        return super(OmnisciOnRayFrame, self)._get_columns()

    columns = property(_get_columns)
    index = property(_get_index, _set_index)

    def has_multiindex(self):
        if self._index_cache is not None:
            return isinstance(self._index_cache, MultiIndex)
        return self._index_cols is not None and len(self._index_cols) > 1

    def to_pandas(self):
        self._execute()
        return super(OmnisciOnRayFrame, self).to_pandas()

    # @classmethod
    # def from_pandas(cls, df):
    #    return super().from_pandas(df)
