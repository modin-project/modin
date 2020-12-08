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

from pandas.core.index import ensure_index, Index, MultiIndex, RangeIndex
from pandas.core.dtypes.common import _get_dtype, is_list_like, is_bool_dtype
from modin.error_message import ErrorMessage
import pandas as pd

from .df_algebra import (
    MaskNode,
    FrameNode,
    GroupbyAggNode,
    TransformNode,
    UnionNode,
    JoinNode,
    SortNode,
    FilterNode,
    translate_exprs_to_base,
    replace_frame_in_exprs,
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

import numpy as np
import pyarrow
import re


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
        force_execution_mode=None,
        has_unsupported_data=False,
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
        self._has_unsupported_data = has_unsupported_data
        if self._op is None:
            self._op = FrameNode(self)

        self._table_cols = columns.tolist()
        if self._index_cols is not None:
            self._table_cols = self._index_cols + self._table_cols

        assert len(dtypes) == len(
            self._table_cols
        ), f"unaligned dtypes ({dtypes}) and table columns ({self._table_cols})"
        if isinstance(dtypes, list):
            if self._index_cols is not None:
                # Table stores both index and data columns but those are accessed
                # differently if we have a MultiIndex for columns. To unify access
                # to dtype we extend index column names to tuples to have a MultiIndex
                # of dtypes.
                if isinstance(columns, MultiIndex):
                    tail = [""] * (columns.nlevels - 1)
                    index_tuples = [(col, *tail) for col in self._index_cols]
                    dtype_index = MultiIndex.from_tuples(index_tuples).append(columns)
                    self._dtypes = pd.Series(dtypes, index=dtype_index)
                else:
                    self._dtypes = pd.Series(dtypes, index=self._table_cols)
            else:
                self._dtypes = pd.Series(dtypes, index=columns)
        else:
            self._dtypes = dtypes

        if partitions is not None:
            self._filter_empties()

        # This frame uses encoding for column names to support exotic
        # (e.g. non-string and reserved words) column names. Encoded
        # names are used in OmniSci tables and corresponding Arrow tables.
        # If we import Arrow table, we have to rename its columns for
        # proper processing.
        if self._has_arrow_table() and self._partitions.size > 0:
            assert self._partitions.size == 1
            table = self._partitions[0][0].get()
            if table.column_names[0] != f"F_{self._table_cols[0]}":
                new_names = [f"F_{col}" for col in table.column_names]
                new_table = table.rename_columns(new_names)
                self._partitions[0][0] = self._frame_mgr_cls._partition_class.put_arrow(
                    new_table
                )

        self._uses_rowid = uses_rowid
        # Tests use forced execution mode to take control over frame
        # execution process. Supported values:
        #  "lazy" - RuntimeError is raised if execution is triggered for the frame
        #  "arrow" - RuntimeError is raised if execution is triggered, but we cannot
        #  execute it using Arrow API (have to use OmniSci for execution)
        self._force_execution_mode = force_execution_mode

    def id_str(self):
        return f"frame${self.id}"

    def _get_dtype(self, col):
        # If we search for an index column type in a MultiIndex then we need to
        # extend index column names to tuples.
        if isinstance(self._dtypes, MultiIndex) and not isinstance(col, tuple):
            return self._dtypes[(col, *([""] * (self._dtypes.nlevels - 1)))]
        return self._dtypes[col]

    def ref(self, col):
        if col == "__rowid__":
            return InputRefExpr(self, col, _get_dtype(int))
        return InputRefExpr(self, col, self._get_dtype(col))

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
                force_execution_mode=self._force_execution_mode,
            )

        if row_indices is not None or row_numeric_idx is not None:
            op = MaskNode(
                base,
                row_indices=row_indices,
                row_numeric_idx=row_numeric_idx,
            )
            return self.__constructor__(
                columns=base.columns,
                dtypes=base._dtypes,
                op=op,
                index_cols=self._index_cols,
                force_execution_mode=self._force_execution_mode,
            )

        return base

    def _has_arrow_table(self):
        if not isinstance(self._op, FrameNode):
            return False
        return all(p.arrow_table for p in self._partitions.flatten())

    def _dtypes_for_cols(self, new_index, new_columns):
        if new_index is not None:
            if isinstance(self._dtypes, MultiIndex):
                new_index = [
                    (col, *([""] * (self._dtypes.nlevels - 1))) for col in new_index
                ]
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
        # Currently we only expect 'by' to be a projection of the same frame.
        # If 'by' holds a list of columns/series, then we create such projection
        # to re-use code.
        if not isinstance(by, DFAlgQueryCompiler):
            if is_list_like(by):
                by_cols = []
                by_frames = []
                for obj in by:
                    if isinstance(obj, str):
                        by_cols.append(obj)
                    elif hasattr(obj, "_modin_frame"):
                        by_frames.append(obj._modin_frame)
                    else:
                        raise NotImplementedError("unsupported groupby args")
                by_cols = Index.__new__(Index, data=by_cols, dtype=self.columns.dtype)
                by_frame = self.mask(col_indices=by_cols)
                if by_frames:
                    by_frame = by_frame._concat(
                        axis=1, other_modin_frames=by_frames, ignore_index=True
                    )
            else:
                raise NotImplementedError("unsupported groupby args")
        else:
            by_frame = by._modin_frame

        if axis != 0:
            raise NotImplementedError("groupby is supported for axis = 0 only")

        base = by_frame._find_common_projections_base(self)
        if base is None:
            raise NotImplementedError("unsupported groupby args")

        if groupby_args["level"] is not None:
            raise NotImplementedError("levels are not supported for groupby")

        groupby_cols = by_frame.columns.tolist()
        agg_cols = [col for col in self.columns if col not in by_frame.columns]

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
            force_execution_mode=self._force_execution_mode,
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
            multiindex = any(isinstance(v, list) for v in agg.values())
            for k, v in agg.items():
                if isinstance(v, list):
                    for item in v:
                        agg_exprs[(k, item)] = AggregateExpr(item, base.ref(k))
                else:
                    col_name = (k, v) if multiindex else k
                    agg_exprs[col_name] = AggregateExpr(v, base.ref(k))
        new_columns.extend(agg_exprs.keys())
        new_dtypes.extend((x._dtype for x in agg_exprs.values()))
        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)

        new_op = GroupbyAggNode(base, groupby_cols, agg_exprs, groupby_args)
        new_frame = self.__constructor__(
            columns=new_columns,
            dtypes=new_dtypes,
            op=new_op,
            index_cols=index_cols,
            force_execution_mode=self._force_execution_mode,
        )

        return new_frame

    def agg(self, agg):
        assert isinstance(agg, str)

        agg_exprs = OrderedDict()
        for col in self.columns:
            agg_exprs[col] = AggregateExpr(agg, self.ref(col))

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(agg_exprs),
            op=GroupbyAggNode(self, [], agg_exprs, {"sort": False}),
            index_cols=None,
            force_execution_mode=self._force_execution_mode,
        )

    def value_counts(self, dropna, columns, sort, ascending):
        by = [col for col in self.columns if columns is None or col in columns]

        if not by:
            raise ValueError("invalid columns subset is specified")

        base = self
        if dropna:
            checks = [base.ref(col).is_not_null() for col in by]
            condition = (
                checks[0]
                if len(checks) == 1
                else OpExpr("AND", [checks], np.dtype("bool"))
            )
            base = self.__constructor__(
                columns=Index.__new__(Index, data=by, dtype="O"),
                dtypes=base._dtypes[by],
                op=FilterNode(base, condition),
                index_cols=None,
                force_execution_mode=base._force_execution_mode,
            )

        agg_exprs = OrderedDict()
        agg_exprs[""] = AggregateExpr("size", None)
        dtypes = base._dtypes[by].tolist()
        dtypes.append(np.dtype("int64"))

        new_columns = Index.__new__(Index, data=[""], dtype="O")

        res = self.__constructor__(
            columns=new_columns,
            dtypes=dtypes,
            op=GroupbyAggNode(base, by, agg_exprs, {"sort": False}),
            index_cols=by.copy(),
            force_execution_mode=base._force_execution_mode,
        )

        if sort or ascending:
            res = self.__constructor__(
                columns=res.columns,
                dtypes=res._dtypes,
                op=SortNode(res, [""], [ascending], "last"),
                index_cols=res._index_cols,
                force_execution_mode=res._force_execution_mode,
            )

        # If a single column is used then it keeps its name.
        # TODO: move it to upper levels when index renaming is in place.
        if len(by) == 1:
            exprs = OrderedDict()
            exprs["__index__"] = res.ref(by[0])
            exprs[by[0]] = res.ref("")

            res = self.__constructor__(
                columns=Index.__new__(Index, data=by, dtype="O"),
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(res, exprs),
                index_cols=["__index__"],
                force_execution_mode=res._force_execution_mode,
            )

        return res

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        limit=None,
        downcast=None,
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
            columns=self.columns,
            dtypes=dtypes,
            op=new_op,
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

        return new_frame

    def dt_extract(self, obj):
        exprs = self._index_exprs()
        for col in self.columns:
            exprs[col] = build_dt_expr(obj, self.ref(col))
        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(exprs)
        return self.__constructor__(
            columns=self.columns,
            dtypes=dtypes,
            op=new_op,
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
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
            force_execution_mode=self._force_execution_mode,
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

        op = JoinNode(
            self,
            other,
            how=how,
            on=on,
            sort=sort,
            suffixes=suffixes,
        )

        new_columns = Index.__new__(Index, data=new_columns, dtype=self.columns.dtype)
        return self.__constructor__(
            dtypes=new_dtypes,
            columns=new_columns,
            op=op,
            force_execution_mode=self._force_execution_mode,
        )

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
                    force_execution_mode=self._force_execution_mode,
                )
            )

        new_frame = aligned_frames[0]
        for frame in aligned_frames[1:]:
            new_frame = self.__constructor__(
                columns=new_columns,
                dtypes=new_frame._dtypes,
                op=UnionNode([new_frame, frame]),
                index_cols=new_frame._index_cols,
                force_execution_mode=self._force_execution_mode,
            )

        return new_frame

    def _concat(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        if not other_modin_frames:
            return self

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
            force_execution_mode=self._force_execution_mode,
        )
        return new_frame

    def bin_op(self, other, op_name, **kwargs):
        if isinstance(other, (int, float, str)):
            value_expr = LiteralExpr(other)
            exprs = self._index_exprs()
            for col in self.columns:
                exprs[col] = self.ref(col).bin_op(value_expr, op_name)
            return self.__constructor__(
                columns=self.columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=self._index_cols,
                force_execution_mode=self._force_execution_mode,
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
                force_execution_mode=self._force_execution_mode,
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
                force_execution_mode=self._force_execution_mode,
            )
        else:
            raise NotImplementedError(f"unsupported operand type: {type(other)}")

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
            force_execution_mode=self._force_execution_mode,
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
            force_execution_mode=self._force_execution_mode,
        )

    def sort_rows(self, columns, ascending, ignore_index, na_position):
        if na_position != "first" and na_position != "last":
            raise ValueError(f"Unsupported na_position value '{na_position}'")

        if not isinstance(columns, list):
            columns = [columns]
        columns = [self._find_index_or_col(col) for col in columns]

        if isinstance(ascending, list):
            if len(ascending) != len(columns):
                raise ValueError("ascending list length doesn't match columns list")
        else:
            if not isinstance(ascending, bool):
                raise ValueError("unsupported ascending value")
            ascending = [ascending] * len(columns)

        if ignore_index:
            # If index is ignored then we might need to drop some columns.
            # At the same time some of dropped index columns can be used
            # for sorting and should be droped after sorting is done.
            if self._index_cols is not None:
                base = self

                drop_index_cols_before = [
                    col for col in self._index_cols if col not in columns
                ]
                drop_index_cols_after = [
                    col for col in self._index_cols if col in columns
                ]
                if not drop_index_cols_after:
                    drop_index_cols_after = None

                if drop_index_cols_before:
                    exprs = OrderedDict()
                    index_cols = (
                        drop_index_cols_after if drop_index_cols_after else None
                    )
                    for col in drop_index_cols_after:
                        exprs[col] = base.ref(col)
                    for col in base.columns:
                        exprs[col] = base.ref(col)
                    base = self.__constructor__(
                        columns=base.columns,
                        dtypes=self._dtypes_for_exprs(exprs),
                        op=TransformNode(base, exprs),
                        index_cols=index_cols,
                        force_execution_mode=self._force_execution_mode,
                    )

                base = self.__constructor__(
                    columns=base.columns,
                    dtypes=base._dtypes,
                    op=SortNode(base, columns, ascending, na_position),
                    index_cols=base._index_cols,
                    force_execution_mode=self._force_execution_mode,
                )

                if drop_index_cols_after:
                    exprs = OrderedDict()
                    for col in base.columns:
                        exprs[col] = base.ref(col)
                    base = self.__constructor__(
                        columns=base.columns,
                        dtypes=self._dtypes_for_exprs(exprs),
                        op=TransformNode(base, exprs),
                        index_cols=None,
                        force_execution_mode=self._force_execution_mode,
                    )

                return base
            else:
                return self.__constructor__(
                    columns=self.columns,
                    dtypes=self._dtypes,
                    op=SortNode(self, columns, ascending, na_position),
                    index_cols=None,
                    force_execution_mode=self._force_execution_mode,
                )
        else:
            base = self

            # If index is preserved and we have no index columns then we
            # need to create one using __rowid__ virtual column.
            if self._index_cols is None:
                base = base._materialize_rowid()

            return self.__constructor__(
                columns=base.columns,
                dtypes=base._dtypes,
                op=SortNode(base, columns, ascending, na_position),
                index_cols=base._index_cols,
                force_execution_mode=self._force_execution_mode,
            )

    def filter(self, key):
        if not isinstance(key, type(self)):
            raise NotImplementedError("Unsupported key type in filter")

        if not isinstance(key._op, TransformNode) or len(key.columns) != 1:
            raise NotImplementedError("Unsupported key in filter")

        key_col = key.columns[0]
        if not is_bool_dtype(key._dtypes[key_col]):
            raise NotImplementedError("Unsupported key in filter")

        base = self._find_common_projections_base(key)
        if base is None:
            raise NotImplementedError("Unsupported key in filter")

        # We build the resulting frame by applying the filter to the
        # base frame and then using the filtered result as a new base.
        # If base frame has no index columns, then we need to create
        # one.
        key_exprs = translate_exprs_to_base(key._op.exprs, base)
        if base._index_cols is None:
            filter_base = base._materialize_rowid()
            key_exprs = replace_frame_in_exprs(key_exprs, base, filter_base)
        else:
            filter_base = base
        condition = key_exprs[key_col]
        filtered_base = self.__constructor__(
            columns=filter_base.columns,
            dtypes=filter_base._dtypes,
            op=FilterNode(filter_base, condition),
            index_cols=filter_base._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

        if self is base:
            exprs = OrderedDict()
            for col in filtered_base._table_cols:
                exprs[col] = filtered_base.ref(col)
        else:
            assert isinstance(
                self._op, TransformNode
            ), f"unexpected op: {self._op.dumps()}"
            exprs = translate_exprs_to_base(self._op.exprs, base)
            exprs = replace_frame_in_exprs(exprs, base, filtered_base)
            if base._index_cols is None:
                exprs["__index__"] = filtered_base.ref("__index__")
                exprs.move_to_end("__index__", last=False)

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(filtered_base, exprs),
            index_cols=filtered_base._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def _materialize_rowid(self):
        exprs = OrderedDict()
        exprs["__index__"] = self.ref("__rowid__")
        for col in self._table_cols:
            exprs[col] = self.ref(col)
        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index_cols=["__index__"],
            uses_rowid=True,
            force_execution_mode=self._force_execution_mode,
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

        if self._force_execution_mode == "lazy":
            raise RuntimeError("unexpected execution triggered on lazy frame")

        # Some frames require rowid which is available for executed frames only.
        # Also there is a common pattern when MaskNode is executed to print
        # frame. If we run the whole tree then any following frame usage will
        # require re-compute. So we just execute MaskNode's operands.
        self._run_sub_queries()

        if self._can_execute_arrow():
            new_table = self._execute_arrow()
            new_partitions = np.empty((1, 1), dtype=np.dtype(object))
            new_partitions[0][0] = self._frame_mgr_cls._partition_class.put_arrow(
                new_table
            )
        else:
            if self._force_execution_mode == "arrow":
                raise RuntimeError("forced arrow execution failed")

            new_partitions = self._frame_mgr_cls.run_exec_plan(
                self._op, self._index_cols, self._dtypes, self._table_cols
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

    def _can_execute_arrow(self):
        if isinstance(self._op, FrameNode):
            return self._has_arrow_table()
        elif isinstance(self._op, MaskNode):
            return (
                self._op.row_indices is None and self._op.input[0]._can_execute_arrow()
            )
        elif isinstance(self._op, TransformNode):
            return self._op.is_drop() and self._op.input[0]._can_execute_arrow()
        elif isinstance(self._op, UnionNode):
            return all(frame._can_execute_arrow() for frame in self._op.input)
        else:
            return False

    def _execute_arrow(self):
        if isinstance(self._op, FrameNode):
            if self._partitions.size == 0:
                return pyarrow.Table()
            else:
                assert self._partitions.size == 1
                return self._partitions[0][0].get()
        elif isinstance(self._op, MaskNode):
            return self._op.input[0]._arrow_row_slice(self._op.row_numeric_idx)
        elif isinstance(self._op, TransformNode):
            return self._op.input[0]._arrow_col_slice(set(self._op.exprs.keys()))
        elif isinstance(self._op, UnionNode):
            return self._arrow_concat(self._op.input)
        else:
            raise RuntimeError(f"Unexpected op ({type(self._op)}) in _execute_arrow")

    def _arrow_col_slice(self, new_columns):
        table = self._execute_arrow()
        return table.drop(
            [f"F_{col}" for col in self._table_cols if col not in new_columns]
        )

    def _arrow_row_slice(self, row_numeric_idx):
        table = self._execute_arrow()
        if isinstance(row_numeric_idx, slice):
            start = 0 if row_numeric_idx.start is None else row_numeric_idx.start
            if start < 0:
                start = table.num_rows - start
            end = (
                table.num_rows if row_numeric_idx.stop is None else row_numeric_idx.stop
            )
            if end < 0:
                end = table.num_rows - end
            if row_numeric_idx.step is None or row_numeric_idx.step == 1:
                length = 0 if start >= end else end - start
                return table.slice(start, length)
            else:
                parts = []
                for i in range(start, end, row_numeric_idx.step):
                    parts.append(table.slice(i, 1))
                return pyarrow.concat_tables(parts)

        start = None
        end = None
        parts = []
        for idx in row_numeric_idx:
            if start is None:
                start = idx
                end = idx
            elif idx == end + 1:
                end = idx
            else:
                if start:
                    parts.append(table.slice(start, end - start + 1))
                start = idx
                end = idx
        parts.append(table.slice(start, end - start + 1))

        return pyarrow.concat_tables(parts)

    @classmethod
    def _arrow_concat(cls, frames):
        return pyarrow.concat_tables(frame._execute_arrow() for frame in frames)

    def _build_index_cache(self):
        assert isinstance(self._op, FrameNode)

        if self._partitions.size == 0:
            self._index_cache = Index.__new__(Index)
        else:
            assert self._partitions.size == 1
            obj = self._partitions[0][0].get()
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                self._index_cache = obj.index
            else:
                assert isinstance(obj, pyarrow.Table)
                if self._index_cols is None:
                    self._index_cache = Index.__new__(
                        RangeIndex, data=range(obj.num_rows)
                    )
                else:
                    index_at = obj.drop([f"F_{col}" for col in self.columns])
                    index_df = index_at.to_pandas()
                    index_df.set_index(
                        [f"F_{col}" for col in self._index_cols], inplace=True
                    )
                    index_df.index.rename(
                        self._index_names(self._index_cols), inplace=True
                    )
                    self._index_cache = index_df.index

    def _get_index(self):
        self._execute()
        if self._index_cache is None:
            self._build_index_cache()
        return self._index_cache

    def _set_index(self, new_index):
        if not isinstance(new_index, (Index, MultiIndex)):
            raise NotImplementedError(
                "OmnisciOnRayFrame._set_index is not yet suported"
            )

        self._execute()

        assert self._partitions.size == 1
        obj = self._partitions[0][0].get()
        if isinstance(obj, pd.DataFrame):
            raise NotImplementedError(
                "OmnisciOnRayFrame._set_index is not yet suported"
            )
        else:
            assert isinstance(obj, pyarrow.Table)

            at = obj
            if self._index_cols:
                at = at.drop(self._index_cols)

            index_df = pd.DataFrame(data={}, index=new_index.copy())
            index_df = index_df.reset_index()

            index_at = pyarrow.Table.from_pandas(index_df)

            for i, field in enumerate(at.schema):
                index_at = index_at.append_column(field, at.column(i))

            index_names = self._mangle_index_names(new_index.names)
            index_at = index_at.rename_columns(index_names + list(self.columns))

            return self.from_arrow(index_at, index_names, new_index)

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
                force_execution_mode=self._force_execution_mode,
            )
        else:
            if self._index_cols is None:
                raise NotImplementedError(
                    "default index reset with no drop is not supported"
                )
            # Need to demangle index names.
            exprs = OrderedDict()
            for i, c in enumerate(self._index_cols):
                name = self._index_name(c)
                if name is None:
                    name = f"level_{i}"
                if name in exprs:
                    raise ValueError(f"cannot insert {name}, already exists")
                exprs[name] = self.ref(c)
            for c in self.columns:
                if c in exprs:
                    raise ValueError(f"cannot insert {c}, already exists")
                exprs[c] = self.ref(c)
            new_columns = Index.__new__(Index, data=exprs.keys(), dtype="O")
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=None,
                force_execution_mode=self._force_execution_mode,
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
            force_execution_mode=self._force_execution_mode,
        )

    def _get_columns(self):
        return super(OmnisciOnRayFrame, self)._get_columns()

    columns = property(_get_columns)
    index = property(_get_index)

    def has_multiindex(self):
        if self._index_cache is not None:
            return isinstance(self._index_cache, MultiIndex)
        return self._index_cols is not None and len(self._index_cols) > 1

    def get_index_name(self):
        if self._index_cols is None:
            return None
        if len(self._index_cols) > 1:
            return None
        return self._index_cols[0]

    def set_index_name(self, name):
        if self.has_multiindex():
            ErrorMessage.single_warning("Scalar name for MultiIndex is not supported!")
            return self

        if self._index_cols is None and name is None:
            return self

        names = self._mangle_index_names([name])
        exprs = OrderedDict()
        if self._index_cols is None:
            exprs[names[0]] = self.ref("__rowid__")
        else:
            exprs[names[0]] = self.ref(self._index_cols[0])

        for col in self.columns:
            exprs[col] = self.ref(col)

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index_cols=names,
            uses_rowid=self._index_cols is None,
            force_execution_mode=self._force_execution_mode,
        )

    def get_index_names(self):
        if self.has_multiindex():
            return self._index_cols.copy()
        return [self.get_index_name()]

    def set_index_names(self, names):
        if not self.has_multiindex():
            raise ValueError("Can set names for MultiIndex only")

        if len(names) != len(self._index_cols):
            raise ValueError(
                f"Unexpected names count: expected {len(self._index_cols)} got {len(names)}"
            )

        names = self._mangle_index_names(names)
        exprs = OrderedDict()
        for old, new in zip(self._index_cols, names):
            exprs[new] = self.ref(old)
        for col in self.columns:
            exprs[col] = self.ref(col)

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index_cols=names,
            force_execution_mode=self._force_execution_mode,
        )

    def to_pandas(self):
        self._execute()

        if self._force_execution_mode == "lazy":
            raise RuntimeError("unexpected to_pandas triggered on lazy frame")

        df = self._frame_mgr_cls.to_pandas(self._partitions)

        # If we make dataframe from Arrow table then we might need to set
        # index columns.
        if len(df.columns) != len(self.columns):
            assert self._index_cols
            df.set_index([f"F_{col}" for col in self._index_cols], inplace=True)
            df.index.rename(self._index_names(self._index_cols), inplace=True)
            assert len(df.columns) == len(self.columns)
        else:
            assert self._index_cols is None
            assert df.index.name is None, f"index name '{df.index.name}' is not None"

        # Restore original column labels encoded in OmniSci to meet its
        # restirctions on column names.
        df.columns = self.columns

        return df

    def _index_names(self, cols):
        if len(cols) == 1:
            return self._index_name(cols[0])
        return [self._index_name(n) for n in cols]

    def _index_name(self, col):
        if col == "__index__":
            return None

        match = re.search("__index__\\d+_(.*)", col)
        if match:
            name = match.group(1)
            if name == "__None__":
                return None
            return name

        return col

    def _find_index_or_col(self, col):
        """For given column or index name return a column name"""
        if col in self.columns:
            return col

        if self._index_cols is not None:
            for idx_col in self._index_cols:
                if re.match(f"__index__\\d+_{col}", idx_col):
                    return idx_col

        raise ValueError(f"Unknown column '{col}'")

    @classmethod
    def from_pandas(cls, df):
        new_index = df.index
        new_columns = df.columns
        # If there is non-trivial index, we put it into columns.
        # That's what we usually have for arrow tables and execution
        # result. Unnamed index is renamed to __index__. Also all
        # columns get 'F_' prefix to handle names unsupported in
        # OmniSci.
        if cls._is_trivial_index(df.index):
            index_cols = None
        else:
            orig_index_names = df.index.names
            orig_df = df

            index_cols = cls._mangle_index_names(df.index.names)
            df.index.names = index_cols
            df = df.reset_index()

            orig_df.index.names = orig_index_names
        new_dtypes = df.dtypes
        df = df.add_prefix("F_")

        (
            new_parts,
            new_lengths,
            new_widths,
            unsupported_cols,
        ) = cls._frame_mgr_cls.from_pandas(df, True)

        if len(unsupported_cols) > 0:
            ErrorMessage.single_warning(
                f"Frame contain columns with unsupported data-types: {unsupported_cols}. "
                "All operations with this frame will be default to pandas!"
            )

        return cls(
            new_parts,
            new_index,
            new_columns,
            new_lengths,
            new_widths,
            dtypes=new_dtypes,
            index_cols=index_cols,
            has_unsupported_data=len(unsupported_cols) > 0,
        )

    @classmethod
    def _mangle_index_names(cls, names):
        return [
            f"__index__{i}_{'__None__' if n is None else n}"
            for i, n in enumerate(names)
        ]

    @classmethod
    def from_arrow(cls, at, index_cols=None, index=None):
        (
            new_frame,
            new_lengths,
            new_widths,
            unsupported_cols,
        ) = cls._frame_mgr_cls.from_arrow(at, return_dims=True)

        if index_cols:
            data_cols = [col for col in at.column_names if col not in index_cols]
            new_index = index
        else:
            data_cols = at.column_names
            assert index is None
            new_index = pd.RangeIndex(at.num_rows)

        new_columns = pd.Index(data=data_cols, dtype="O")
        new_dtypes = pd.Series(
            [cls._arrow_type_to_dtype(col.type) for col in at.columns],
            index=at.column_names,
        )

        if len(unsupported_cols) > 0:
            ErrorMessage.single_warning(
                f"Frame contain columns with unsupported data-types: {unsupported_cols}. "
                "All operations with this frame will be default to pandas!"
            )

        return cls(
            partitions=new_frame,
            index=new_index,
            columns=new_columns,
            row_lengths=new_lengths,
            column_widths=new_widths,
            dtypes=new_dtypes,
            index_cols=index_cols,
            has_unsupported_data=len(unsupported_cols) > 0,
        )

    @classmethod
    def _is_trivial_index(cls, index):
        """Return true if index is a range [0..N]"""
        if isinstance(index, pd.RangeIndex):
            return index.start == 0 and index.step == 1
        if not isinstance(index, pd.Int64Index):
            return False
        return (
            index.is_monotonic_increasing
            and index.unique
            and index.min == 0
            and index.max == len(index) - 1
        )
