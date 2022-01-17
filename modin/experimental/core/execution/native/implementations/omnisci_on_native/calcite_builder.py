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

"""Module provides ``CalciteBuilder`` class."""

from .expr import (
    InputRefExpr,
    LiteralExpr,
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
from pandas.core.dtypes.common import get_dtype


class CalciteBuilder:
    """Translator used to transform ``DFAlgNode`` tree into a calcite node sequence."""

    class CompoundAggregate:
        """
        A base class for a compound aggregate translation.

        Translation is done in three steps. Step 1 is an additional
        values generation using a projection. Step 2 is a generation
        of aggregates that will be later used for a compound aggregate
        value computation. Step 3 is a final aggregate value generation
        using another projection.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : BaseExpr
            An aggregated value.
        """

        def __init__(self, builder, arg):
            self._builder = builder
            self._arg = arg

        def gen_proj_exprs(self):
            """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
            return []

        def gen_agg_exprs(self):
            """
            Generate intermediate aggregates required for a compound aggregate computation.

            Returns
            -------
            dict
                New aggregate expressions mapped to their names.
            """
            pass

        def gen_reduce_expr(self):
            """
            Generate an expression for a compound aggregate.

            Returns
            -------
            BaseExpr
                A final compound aggregate expression.
            """
            pass

    class StdAggregate(CompoundAggregate):
        """
        A sample standard deviation aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : BaseExpr
            An aggregated value.
        """

        def __init__(self, builder, arg):
            assert isinstance(arg, InputRefExpr)
            super().__init__(builder, arg)

            self._quad_name = self._arg.column + "__quad__"
            self._sum_name = self._arg.column + "__sum__"
            self._quad_sum_name = self._arg.column + "__quad_sum__"
            self._count_name = self._arg.column + "__count__"

        def gen_proj_exprs(self):
            """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
            expr = self._builder._translate(self._arg.mul(self._arg))
            return {self._quad_name: expr}

        def gen_agg_exprs(self):
            """
            Generate intermediate aggregates required for a compound aggregate computation.

            Returns
            -------
            dict
                New aggregate expressions mapped to their names.
            """
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
            """
            Generate an expression for a compound aggregate.

            Returns
            -------
            BaseExpr
                A final compound aggregate expression.
            """
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = get_dtype(int)
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
        """
        An unbiased skew aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : BaseExpr
            An aggregated value.
        """

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
            """
            Generate values required for intermediate aggregates computation.

            Returns
            -------
            dict
                New column expressions mapped to their names.
            """
            quad_expr = self._builder._translate(self._arg.mul(self._arg))
            cube_expr = self._builder._translate(
                self._arg.mul(self._arg).mul(self._arg)
            )
            return {self._quad_name: quad_expr, self._cube_name: cube_expr}

        def gen_agg_exprs(self):
            """
            Generate intermediate aggregates required for a compound aggregate computation.

            Returns
            -------
            dict
                New aggregate expressions mapped to their names.
            """
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
            """
            Generate an expression for a compound aggregate.

            Returns
            -------
            BaseExpr
                A final compound aggregate expression.
            """
            count_expr = self._builder._ref(self._arg.modin_frame, self._count_name)
            count_expr._dtype = get_dtype(int)
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
        """
        A class to track current input frames and corresponding nodes.

        Used to translate input column references to numeric indices.

        Parameters
        ----------
        input_frames : list of DFAlgNode
            Input nodes of the currently translated node.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Attributes
        ----------
        input_nodes : list of CalciteBaseNode
            Input nodes of the currently translated node.
        frame_to_node : dict
            Maps input frames to corresponding calcite nodes.
        input_offsets : dict
            Maps input frame to an input index used for its first column.
        replacements : dict
            Maps input frame to a new list of columns to use. Used when
            a single `DFAlgNode` is lowered into multiple computation
            steps, e.g. for compound aggregates requiring additional
            projections.
        """

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
            """
            Use `node` as an input node for references to columns of `frame`.

            Parameters
            ----------
            frame : DFAlgNode
                Replaced input frame.
            node : CalciteBaseNode
                A new node to use.
            new_cols : list of str
                A new columns list to use.
            """
            self.replacements[frame] = new_cols

        def _idx(self, frame, col):
            """
            Get a numeric input index for an input column.

            Parameters
            ----------
            frame : DFAlgNode
                An input frame.
            col : str
                An input column.

            Returns
            -------
            int
            """
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
            """
            Translate input column into ``CalciteInputRefExpr``.

            Parameters
            ----------
            frame : DFAlgNode
                An input frame.
            col : str
                An input column.

            Returns
            -------
            CalciteInputRefExpr
            """
            return CalciteInputRefExpr(self._idx(frame, col))

        def ref_idx(self, frame, col):
            """
            Translate input column into ``CalciteInputIdxExpr``.

            Parameters
            ----------
            frame : DFAlgNode
                An input frame.
            col : str
                An input column.

            Returns
            -------
            CalciteInputIdxExpr
            """
            return CalciteInputIdxExpr(self._idx(frame, col))

        def input_ids(self):
            """
            Get ids of all input nodes.

            Returns
            -------
            list of int
            """
            return [x.id for x in self.input_nodes]

        def translate(self, expr):
            """
            Translate an expression.

            Translation is done by replacing ``InputRefExpr`` with
            ``CalciteInputRefExpr`` and ``CalciteInputIdxExpr``.

            Parameters
            ----------
            expr : BaseExpr
                An expression to translate.

            Returns
            -------
            BaseExpr
                Translated expression.
            """
            return self._maybe_copy_and_translate_expr(expr)

        def _maybe_copy_and_translate_expr(self, expr, ref_idx=False):
            """
            Translate an expression.

            Translate an expression replacing ``InputRefExpr`` with ``CalciteInputRefExpr``
            and ``CalciteInputIdxExpr``. An expression tree branches with input columns
            are copied into a new tree, other branches are used as is.

            Parameters
            ----------
            expr : BaseExpr
                An expression to translate.
            ref_idx : bool, default: False
                If True then translate ``InputRefExpr`` to ``CalciteInputIdxExpr``,
                use ``CalciteInputRefExr`` otherwise.

            Returns
            -------
            BaseExpr
                Translated expression.
            """
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
        """
        A helper class to manage an input context stack.

        The class is designed to be used in a recursion with nested
        'with' statements.

        Parameters
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Attributes
        ----------
        builder : CalciteBuilder
            An outer builder.
        input_frames : list of DFAlgNode
            Input nodes for the new context.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.
        """

        def __init__(self, builder, input_frames, input_nodes):
            self.builder = builder
            self.input_frames = input_frames
            self.input_nodes = input_nodes

        def __enter__(self):
            """
            Push new input context into the input context stack.

            Returns
            -------
            InputContext
                New input context.
            """
            self.builder._input_ctx_stack.append(
                self.builder.InputContext(self.input_frames, self.input_nodes)
            )
            return self.builder._input_ctx_stack[-1]

        def __exit__(self, type, value, traceback):
            """
            Pop current input context.

            Parameters
            ----------
            type : Any
                An exception type.
            value : Any
                An exception value.
            traceback : Any
                A traceback.
            """
            self.builder._input_ctx_stack.pop()

    type_strings = {
        int: "INTEGER",
        bool: "BOOLEAN",
    }

    def __init__(self):
        self._input_ctx_stack = []

    def build(self, op):
        """
        Translate a ``DFAlgNode`` tree into a calcite nodes sequence.

        Parameters
        ----------
        op : DFAlgNode
            A tree to translate.

        Returns
        -------
        list of CalciteBaseNode
            The resulting calcite nodes sequence.
        """
        CalciteBaseNode.reset_id()
        self.res = []
        self._to_calcite(op)
        return self.res

    def _add_projection(self, frame):
        """
        Add a projection node to the resulting sequence.

        Added node simply selects all frame's columns. This method can be used
        to discard a virtual 'rowid' column provided by all scan nodes.

        Parameters
        ----------
        frame : OmnisciOnNativeDataframe
            An input frame for a projection.

        Returns
        -------
        CalciteProjectionNode
            Created projection node.
        """
        proj = CalciteProjectionNode(
            frame._table_cols, [self._ref(frame, col) for col in frame._table_cols]
        )
        self._push(proj)
        return proj

    def _input_ctx(self):
        """
        Get current input context.

        Returns
        -------
        InputContext
        """
        return self._input_ctx_stack[-1]

    def _set_input_ctx(self, op):
        """
        Create input context manager for a node translation.

        Parameters
        ----------
        op : DFAlgNode
            A translated node.

        Returns
        -------
        InputContextMgr
            Created input context manager.
        """
        input_frames = getattr(op, "input", [])
        input_nodes = [self._to_calcite(x._op) for x in input_frames]
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _set_tmp_ctx(self, input_frames, input_nodes):
        """
        Create a temporary input context manager.

        This method is deprecated.

        Parameters
        ----------
        input_frames : list of DFAlgNode
            Input nodes of the currently translated node.
        input_nodes : list of CalciteBaseNode
            Translated input nodes.

        Returns
        -------
        InputContextMgr
            Created input context manager.
        """
        return self.InputContextMgr(self, input_frames, input_nodes)

    def _ref(self, frame, col):
        """
        Translate input column into ``CalciteInputRefExpr``.

        Parameters
        ----------
        frame : DFAlgNode
            An input frame.
        col : str
            An input column.

        Returns
        -------
        CalciteInputRefExpr
        """
        return self._input_ctx().ref(frame, col)

    def _ref_idx(self, frame, col):
        """
        Translate input column into ``CalciteInputIdxExpr``.

        Parameters
        ----------
        frame : DFAlgNode
            An input frame.
        col : str
            An input column.

        Returns
        -------
        CalciteInputIdxExpr
        """
        return self._input_ctx().ref_idx(frame, col)

    def _translate(self, exprs):
        """
        Translate expressions.

        Translate expressions replacing ``InputRefExpr`` with ``CalciteInputRefExpr`` and
        ``CalciteInputIdxExpr``.

        Parameters
        ----------
        exprs : BaseExpr or list-like of BaseExpr
            Expressions to translate.

        Returns
        -------
        BaseExpr or list of BaseExpr
            Translated expression.
        """
        if isinstance(exprs, abc.Iterable):
            return [self._input_ctx().translate(x) for x in exprs]
        return self._input_ctx().translate(exprs)

    def _push(self, node):
        """
        Append node to the resulting sequence.

        Parameters
        ----------
        node : CalciteBaseNode
            A node to add.
        """
        self.res.append(node)

    def _last(self):
        """
        Get the last node of the resulting calcite node sequence.

        Returns
        -------
        CalciteBaseNode
        """
        return self.res[-1]

    def _input_nodes(self):
        """
        Get current input calcite nodes.

        Returns
        -------
        list if CalciteBaseNode
        """
        return self._input_ctx().input_nodes

    def _input_node(self, idx):
        """
        Get an input calcite node by index.

        Parameters
        ----------
        idx : int
            An input node's index.

        Returns
        -------
        CalciteBaseNode
        """
        return self._input_nodes()[idx]

    def _input_ids(self):
        """
        Get ids of the current input nodes.

        Returns
        -------
        list of int
        """
        return self._input_ctx().input_ids()

    def _to_calcite(self, op):
        """
        Translate tree to a calcite node sequence.

        Parameters
        ----------
        op : DFAlgNode
            A tree to translate.

        Returns
        -------
        CalciteBaseNode
            The last node of the generated sequence.
        """
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
        """
        Translate ``FrameNode`` node.

        Parameters
        ----------
        op : FrameNode
            A frame to translate.
        """
        self._push(CalciteScanNode(op.modin_frame))

    def _process_mask(self, op):
        """
        Translate ``MaskNode`` node.

        Parameters
        ----------
        op : MaskNode
            An operation to translate.
        """
        if op.row_labels is not None:
            raise NotImplementedError("row indices masking is not yet supported")

        frame = op.input[0]

        # select rows by rowid
        rowid_col = self._ref(frame, "__rowid__")
        condition = build_row_idx_filter_expr(op.row_positions, rowid_col)
        self._push(CalciteFilterNode(condition))

        # mask is currently always applied over scan, it means
        # we need additional projection to remove rowid column
        self._add_projection(frame)

    def _process_groupby(self, op):
        """
        Translate ``GroupbyAggNode`` node.

        Parameters
        ----------
        op : GroupbyAggNode
            An operation to translate.
        """
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
        """
        Translate ``TransformNode`` node.

        Parameters
        ----------
        op : TransformNode
            An operation to translate.
        """
        fields = list(op.exprs.keys())
        exprs = self._translate(op.exprs.values())
        self._push(CalciteProjectionNode(fields, exprs))

    def _process_join(self, op):
        """
        Translate ``JoinNode`` node.

        Parameters
        ----------
        op : JoinNode
            An operation to translate.
        """
        node = CalciteJoinNode(
            left_id=self._input_node(0).id,
            right_id=self._input_node(1).id,
            how=op.how,
            condition=self._translate(op.condition),
        )
        self._push(node)

        self._push(
            CalciteProjectionNode(
                op.exprs.keys(), [self._translate(val) for val in op.exprs.values()]
            )
        )

    def _process_union(self, op):
        """
        Translate ``UnionNode`` node.

        Parameters
        ----------
        op : UnionNode
            An operation to translate.
        """
        self._push(CalciteUnionNode(self._input_ids(), True))

    def _process_sort(self, op):
        """
        Translate ``SortNode`` node.

        Parameters
        ----------
        op : SortNode
            An operation to translate.
        """
        frame = op.input[0]
        if not isinstance(self._input_node(0), CalciteProjectionNode):
            proj = self._add_projection(frame)
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
        """
        Translate ``FilterNode`` node.

        Parameters
        ----------
        op : FilterNode
            An operation to translate.
        """
        condition = self._translate(op.condition)
        self._push(CalciteFilterNode(condition))

        if isinstance(self._input_node(0), CalciteScanNode):
            # if filter was applied over scan, then we need additional
            # projection to remove rowid column
            self._add_projection(op.input[0])
