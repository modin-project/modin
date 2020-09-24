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

from .expr import (
    InputRefExpr,
    LiteralExpr,
    OpExpr,
    AggregateExpr,
    build_if_then_else,
    build_row_idx_filter_expr,
)
from .calcite_algebra import (
    CalciteBaseNode,
    CalciteInputRefExpr,
    CalciteInputIdxExpr,
    CalciteScanNode,
    CalciteProjectionNode,
    CalciteFilterNode,
    CalciteAggregateNode,
    CalciteCollation,
    CalciteSortNode,
    CalciteJoinNode,
    CalciteUnionNode,
)
from .df_algebra import (
    FrameNode,
    MaskNode,
    GroupbyAggNode,
    TransformNode,
    JoinNode,
    UnionNode,
    SortNode,
    FilterNode,
)

from collections import abc
from pandas.core.dtypes.common import _get_dtype


class CalciteBuilder:
    class CompoundAggregate:
        def __init__(self, builder, arg):
            self._builder = builder
            self._arg = arg

        def gen_proj_exprs(self):
            return []

        def gen_agg_exprs(self):
            pass

        def gen_reduce_expr(self):
            pass

    class StdAggregate(CompoundAggregate):
        def __init__(self, builder, arg):
            assert isinstance(arg, InputRefExpr)
            super().__init__(builder, arg)

            self._quad_name = self._arg.column + "__quad__"
            self._sum_name = self._arg.column + "__sum__"
            self._quad_sum_name = self._arg.column + "__quad_sum__"
            self._count_name = self._arg.column + "__count__"

        def gen_proj_exprs(self):
            expr = self._builder._translate(self._arg.mul(self._arg))
            return {self._quad_name: expr}

        def gen_agg_exprs(self):
            count_expr = self._builder._translate(AggregateExpr("count", self._arg))
            sum_expr = self._builder._translate(AggregateExpr("sum", self._arg))
            self._sum_dtype = sum_expr._dtype
            qsum_expr = AggregateExpr(
                "SUM",
                self._builder._ref_idx(self._arg.modin_frame, self._quad_name),
                dtype=sum_expr._dtype,
            )

            return {
                self._sum_name: sum_expr,
                self._quad_sum_name: qsum_expr,
                self._count_name: count_expr,
            }

        def gen_reduce_expr(self):
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = _get_dtype(int)
            sum_expr = self._builder._ref(self._arg.modin_frame, self._sum_name)
            sum_expr._dtype = self._sum_dtype
            qsum_expr = self._builder._ref(self._arg.modin_frame, self._quad_sum_name)
            qsum_expr._dtype = self._sum_dtype

            null_expr = LiteralExpr(None)
            count_or_null = build_if_then_else(
                count_expr.eq(LiteralExpr(0)), null_expr, count_expr, count_expr._dtype
            )
            count_m_1_or_null = build_if_then_else(
                count_expr.eq(LiteralExpr(1)),
                null_expr,
                count_expr.sub(LiteralExpr(1)),
                count_expr._dtype,
            )

            # sqrt((sum(x * x) - sum(x) * sum(x) / n) / (n - 1))
            return (
                qsum_expr.sub(sum_expr.mul(sum_expr).truediv(count_or_null))
                .truediv(count_m_1_or_null)
                .pow(LiteralExpr(0.5))
            )

    class SkewAggregate(CompoundAggregate):
        def __init__(self, builder, arg):
            assert isinstance(arg, InputRefExpr)
            super().__init__(builder, arg)

            self._quad_name = self._arg.column + "__quad__"
            self._cube_name = self._arg.column + "__cube__"
            self._sum_name = self._arg.column + "__sum__"
            self._quad_sum_name = self._arg.column + "__quad_sum__"
            self._cube_sum_name = self._arg.column + "__cube_sum__"
            self._count_name = self._arg.column + "__count__"

        def gen_proj_exprs(self):
            quad_expr = self._builder._translate(self._arg.mul(self._arg))
            cube_expr = self._builder._translate(
                self._arg.mul(self._arg).mul(self._arg)
            )
            return {self._quad_name: quad_expr, self._cube_name: cube_expr}

        def gen_agg_exprs(self):
            count_expr = self._builder._translate(AggregateExpr("count", self._arg))
            sum_expr = self._builder._translate(AggregateExpr("sum", self._arg))
            self._sum_dtype = sum_expr._dtype
            qsum_expr = AggregateExpr(
                "SUM",
                self._builder._ref_idx(self._arg.modin_frame, self._quad_name),
                dtype=sum_expr._dtype,
            )
            csum_expr = AggregateExpr(
                "SUM",
                self._builder._ref_idx(self._arg.modin_frame, self._cube_name),
                dtype=sum_expr._dtype,
            )

            return {
                self._sum_name: sum_expr,
                self._quad_sum_name: qsum_expr,
                self._cube_sum_name: csum_expr,
                self._count_name: count_expr,
            }

        def gen_reduce_expr(self):
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = _get_dtype(int)
            sum_expr = self._builder._ref(self._arg.modin_frame, self._sum_name)
            sum_expr._dtype = self._sum_dtype
            qsum_expr = self._builder._ref(self._arg.modin_frame, self._quad_sum_name)
            qsum_expr._dtype = self._sum_dtype
            csum_expr = self._builder._ref(self._arg.modin_frame, self._cube_sum_name)
            csum_expr._dtype = self._sum_dtype

            mean_expr = sum_expr.truediv(count_expr)

            # n * sqrt(n - 1) / (n - 2)
            #  * (sum(x ** 3) - 3 * mean * sum(x * x) + 2 * mean * mean * sum(x))
            #  / (sum(x * x) - mean * sum(x)) ** 1.5
            part1 = count_expr.mul(
                count_expr.sub(LiteralExpr(1)).pow(LiteralExpr(0.5))
            ).truediv(count_expr.sub(LiteralExpr(2)))
            part2 = csum_expr.sub(mean_expr.mul(qsum_expr).mul(LiteralExpr(3.0))).add(
                mean_expr.mul(mean_expr).mul(sum_expr).mul(LiteralExpr(2.0))
            )
            part3 = qsum_expr.sub(mean_expr.mul(sum_expr)).pow(LiteralExpr(1.5))
            skew_expr = part1.mul(part2).truediv(part3)

            # The result is NULL if n <= 2
            return build_if_then_else(
                count_expr.le(LiteralExpr(2)),
                LiteralExpr(None),
                skew_expr,
                skew_expr._dtype,
            )

    _compound_aggregates = {"std": StdAggregate, "skew": SkewAggregate}

    class InputContext:
        _simple_aggregates = {
            "sum": "SUM",
            "mean": "AVG",
            "max": "MAX",
            "min": "MIN",
            "size": "COUNT",
            "count": "COUNT",
        }
        _no_arg_aggregates = {"size"}

        def __init__(self, input_frames, input_nodes):
            self.input_nodes = input_nodes
            self.frame_to_node = {x: y for x, y in zip(input_frames, input_nodes)}
            self.input_offsets = {}
            self.replacements = {}
            offs = 0
            for frame in input_frames:
                self.input_offsets[frame] = offs
                offs += len(frame._table_cols)
                # Materialized frames have additional 'rowid' column
                if isinstance(frame._op, FrameNode):
                    offs += 1

        def replace_input_node(self, frame, node, new_cols):
            self.replacements[frame] = new_cols

        def _idx(self, frame, col):
            assert (
                frame in self.input_offsets
            ), f"unexpected reference to {frame.id_str()}"

            offs = self.input_offsets[frame]

            if frame in self.replacements:
                return self.replacements[frame].index(col) + offs

            if col == "__rowid__":
                if not isinstance(self.frame_to_node[frame], CalciteScanNode):
                    raise NotImplementedError(
                        "rowid can be accessed in materialized frames only"
                    )
                return len(frame._table_cols) + offs

            assert (
                col in frame._table_cols
            ), f"unexpected reference to '{col}' in {frame.id_str()}"
            return frame._table_cols.index(col) + offs

        def ref(self, frame, col):
            return CalciteInputRefExpr(self._idx(frame, col))

        def ref_idx(self, frame, col):
            return CalciteInputIdxExpr(self._idx(frame, col))

        def input_ids(self):
            return [x.id for x in self.input_nodes]

        def translate(self, expr):
            """Copy those parts of expr tree that have input references
            and translate all references into CalciteInputRefExr"""
            return self._maybe_copy_and_translate_expr(expr)

        def _maybe_copy_and_translate_expr(self, expr, ref_idx=False):
            if isinstance(expr, InputRefExpr):
                if ref_idx:
                    return self.ref_idx(expr.modin_frame, expr.column)
                else:
                    return self.ref(expr.modin_frame, expr.column)

            if isinstance(expr, AggregateExpr):
                expr = expr.copy()
                if expr.agg in self._no_arg_aggregates:
                    expr.operands = []
                else:
                    expr.operands[0] = self._maybe_copy_and_translate_expr(
                        expr.operands[0], True
                    )
                expr.agg = self._simple_aggregates[expr.agg]
                return expr

            copied = False
            for i, op in enumerate(getattr(expr, "operands", [])):
                new_op = self._maybe_copy_and_translate_expr(op)
                if new_op != op:
                    if not copied:
                        expr = expr.copy()
                    expr.operands[i] = new_op
            return expr

    class InputContextMgr:
        def __init__(self, builder, input_frames, input_nodes):
            self.builder = builder
            self.input_frames = input_frames
            self.input_nodes = input_nodes

        def __enter__(self):
            self.builder._input_ctx_stack.append(
                self.builder.InputContext(self.input_frames, self.input_nodes)
            )
            return self.builder._input_ctx_stack[-1]

        def __exit__(self, type, value, traceback):
            self.builder._input_ctx_stack.pop()

    type_strings = {
        int: "INTEGER",
        bool: "BOOLEAN",
    }

    def __init__(self):
        self._input_ctx_stack = []

    def build(self, op):
        CalciteBaseNode.reset_id()
        self.res = []
        self._to_calcite(op)
        return self.res

    def _input_ctx(self):
        return self._input_ctx_stack[-1]

    def _set_input_ctx(self, op):
        input_frames = getattr(op, "input", [])
        input_nodes = [self._to_calcite(x._op) for x in input_frames]
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _set_tmp_ctx(self, input_frames, input_nodes):
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _ref(self, frame, col):
        return self._input_ctx().ref(frame, col)

    def _ref_idx(self, frame, col):
        return self._input_ctx().ref_idx(frame, col)

    def _translate(self, exprs):
        if isinstance(exprs, abc.Iterable):
            return [self._input_ctx().translate(x) for x in exprs]
        return self._input_ctx().translate(exprs)

    def _push(self, node):
        self.res.append(node)

    def _last(self):
        return self.res[-1]

    def _input_nodes(self):
        return self._input_ctx().input_nodes

    def _input_node(self, idx):
        return self._input_nodes()[idx]

    def _input_ids(self):
        return self._input_ctx().input_ids()

    def _to_calcite(self, op):
        # This context translates input operands and setup current
        # input context to translate input references (recursion
        # over tree happens here).
        with self._set_input_ctx(op):
            if isinstance(op, FrameNode):
                self._process_frame(op)
            elif isinstance(op, MaskNode):
                self._process_mask(op)
            elif isinstance(op, GroupbyAggNode):
                self._process_groupby(op)
            elif isinstance(op, TransformNode):
                self._process_transform(op)
            elif isinstance(op, JoinNode):
                self._process_join(op)
            elif isinstance(op, UnionNode):
                self._process_union(op)
            elif isinstance(op, SortNode):
                self._process_sort(op)
            elif isinstance(op, FilterNode):
                self._process_filter(op)
            else:
                raise NotImplementedError(
                    f"CalciteBuilder doesn't support {type(op).__name__}"
                )
        return self.res[-1]

    def _process_frame(self, op):
        self._push(CalciteScanNode(op.modin_frame))

    def _process_mask(self, op):
        if op.row_indices is not None:
            raise NotImplementedError("row indices masking is not yet supported")

        frame = op.input[0]

        # select rows by rowid
        rowid_col = self._ref(frame, "__rowid__")
        condition = build_row_idx_filter_expr(op.row_numeric_idx, rowid_col)
        self._push(CalciteFilterNode(condition))

        # mask is currently always applied over scan, it means
        # we need additional projection to remove rowid column
        fields = frame._table_cols
        exprs = [self._ref(frame, col) for col in frame._table_cols]
        self._push(CalciteProjectionNode(fields, exprs))

    def _process_groupby(self, op):
        frame = op.input[0]

        # Aggregation's input should always be a projection and
        # group key columns should always go first
        proj_cols = op.by.copy()
        for col in frame._table_cols:
            if col not in op.by:
                proj_cols.append(col)
        proj_exprs = [self._ref(frame, col) for col in proj_cols]
        # Add expressions required for compound aggregates
        compound_aggs = {}
        for agg, expr in op.agg_exprs.items():
            if expr.agg in self._compound_aggregates:
                compound_aggs[agg] = self._compound_aggregates[expr.agg](
                    self, expr.operands[0]
                )
                extra_exprs = compound_aggs[agg].gen_proj_exprs()
                proj_cols.extend(extra_exprs.keys())
                proj_exprs.extend(extra_exprs.values())
        proj = CalciteProjectionNode(proj_cols, proj_exprs)
        self._push(proj)

        self._input_ctx().replace_input_node(frame, proj, proj_cols)

        group = [self._ref_idx(frame, col) for col in op.by]
        fields = op.by.copy()
        aggs = []
        for agg, expr in op.agg_exprs.items():
            if agg in compound_aggs:
                extra_aggs = compound_aggs[agg].gen_agg_exprs()
                fields.extend(extra_aggs.keys())
                aggs.extend(extra_aggs.values())
            else:
                fields.append(agg)
                aggs.append(self._translate(expr))
        node = CalciteAggregateNode(fields, group, aggs)
        self._push(node)

        if compound_aggs:
            self._input_ctx().replace_input_node(frame, node, fields)
            proj_cols = op.by.copy()
            proj_exprs = [self._ref(frame, col) for col in proj_cols]
            proj_cols.extend(op.agg_exprs.keys())
            for agg in op.agg_exprs:
                if agg in compound_aggs:
                    proj_exprs.append(compound_aggs[agg].gen_reduce_expr())
                else:
                    proj_exprs.append(self._ref(frame, agg))
            proj = CalciteProjectionNode(proj_cols, proj_exprs)
            self._push(proj)

        if op.groupby_opts["sort"]:
            collation = [CalciteCollation(col) for col in group]
            self._push(CalciteSortNode(collation))

    def _process_transform(self, op):
        fields = list(op.exprs.keys())
        exprs = self._translate(op.exprs.values())
        self._push(CalciteProjectionNode(fields, exprs))

    def _process_join(self, op):
        left = op.input[0]
        right = op.input[1]

        assert (
            op.on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        for col in op.on:
            assert (
                col in left._table_cols and col in right._table_cols
            ), f"Column '{col}'' is missing in one of merge operands"

        """ Join, only equal-join supported """
        cmps = [self._ref(left, c).eq(self._ref(right, c)) for c in op.on]
        if len(cmps) > 1:
            condition = OpExpr("AND", cmps, _get_dtype(bool))
        else:
            condition = cmps[0]
        node = CalciteJoinNode(
            left_id=self._input_node(0).id,
            right_id=self._input_node(1).id,
            how=op.how,
            condition=condition,
        )
        self._push(node)

        """Projection for both frames"""
        fields = []
        exprs = []
        conflicting_cols = set(left.columns) & set(right.columns) - set(op.on)
        """First goes 'on' column then all left columns(+suffix for conflicting names)
        but 'on' then all right columns(+suffix for conflicting names) but 'on'"""
        on_idx = [-1] * len(op.on)
        for c in left.columns:
            if c in op.on:
                on_idx[op.on.index(c)] = len(fields)
            suffix = op.suffixes[0] if c in conflicting_cols else ""
            fields.append(c + suffix)
            exprs.append(self._ref(left, c))

        for c in right.columns:
            if c not in op.on:
                suffix = op.suffixes[1] if c in conflicting_cols else ""
                fields.append(c + suffix)
                exprs.append(self._ref(right, c))

        self._push(CalciteProjectionNode(fields, exprs))

        # TODO: current input translation system doesn't work here
        # because there is no frame to reference for index computation.
        # We should build calcite tree to keep references to input
        # nodes and keep scheme in calcite nodes. For now just use
        # known index on_idx.
        if op.sort is True:
            """Sort by key column"""
            collation = [CalciteCollation(CalciteInputIdxExpr(x)) for x in on_idx]
            self._push(CalciteSortNode(collation))

    def _process_union(self, op):
        self._push(CalciteUnionNode(self._input_ids(), True))

    def _process_sort(self, op):
        frame = op.input[0]

        # Sort should be applied to projections.
        if not isinstance(self._input_node(0), CalciteProjectionNode):
            proj = CalciteProjectionNode(
                frame._table_cols, [self._ref(frame, col) for col in frame._table_cols]
            )
            self._push(proj)
            self._input_ctx().replace_input_node(frame, proj, frame._table_cols)

        nulls = op.na_position.upper()
        collations = []
        for col, asc in zip(op.columns, op.ascending):
            ascending = "ASCENDING" if asc else "DESCENDING"
            collations.append(
                CalciteCollation(self._ref_idx(frame, col), ascending, nulls)
            )
        self._push(CalciteSortNode(collations))

    def _process_filter(self, op):
        condition = self._translate(op.condition)
        self._push(CalciteFilterNode(condition))
