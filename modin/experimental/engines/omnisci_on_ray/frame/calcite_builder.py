from .expr import *
from .calcite_algebra import *
from .df_algebra import *

from collections import abc


class CalciteBuilder:
    class InputContext:
        def __init__(self, input_frames, input_nodes):
            self.input_nodes = input_nodes
            self.frame_to_node = {x: y for x, y in zip(input_frames, input_nodes)}
            self.input_offsets = {}
            offs = 0
            for frame in input_frames:
                self.input_offsets[frame] = offs
                offs += len(frame._table_cols)
                # Materialized frames have additional 'rowid' column
                if isinstance(frame._op, FrameNode):
                    offs += 1

        def _idx(self, frame, col):
            assert (
                frame in self.input_offsets
            ), f"unexpected reference to {frame.id_str()}"

            offs = self.input_offsets[frame]
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

        def _maybe_copy_and_translate_expr(self, expr):
            if isinstance(expr, InputRefExpr):
                return self.ref(expr.modin_frame, expr.column)
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
        if op.row_numeric_idx is not None:
            rowid_col = self._ref(frame, "__rowid__")
            condition = build_row_idx_filter_expr(op.row_numeric_idx, rowid_col)
            self._push(CalciteFilterNode(condition))

        # make projection
        fields = []
        if frame._index_cols is not None:
            fields += frame._index_cols

        if op.col_indices is not None or op.col_numeric_idx is not None:
            if op.col_indices is not None:
                fields += list(op.col_indices)
            elif op.col_numeric_idx is not None:
                fields += frame.columns[op.col_numeric_idx].tolist()
        else:
            fields += list(frame.columns)
        exprs = [self._ref(frame, col) for col in fields]

        self._push(CalciteProjectionNode(fields, exprs))

    def _process_groupby(self, op):
        frame = op.input[0]

        # We need a projection to be aggregation op
        input_nodes = self._input_nodes()
        if not isinstance(input_nodes[0], CalciteProjectionNode):
            table_cols = frame._table_cols
            self._push(
                CalciteProjectionNode(
                    table_cols, [self._ref(frame, col) for col in table_cols]
                )
            )
            input_nodes = [self._last()]

        with self._set_tmp_ctx(op.input, input_nodes):
            fields = []
            group = []
            aggs = []
            for col in op.by:
                fields.append(col)
                group.append(self._ref_idx(frame, col))
            for col, agg in op.agg.items():
                if isinstance(agg, list):
                    for agg_val in agg:
                        fields.append(col + " " + agg_val)
                        aggs.append(
                            self._create_agg_expr(self._ref_idx(frame, col), agg_val)
                        )
                else:
                    fields.append(col)
                    aggs.append(self._create_agg_expr(self._ref_idx(frame, col), agg))

            node = CalciteAggregateNode(fields, group, aggs)
            self._push(node)

            if op.groupby_opts["sort"]:
                with self._set_tmp_ctx(op.input, [node]):
                    collation = [CalciteCollation(col) for col in group]
                    self._push(CalciteSortNode(collation))

    def _create_agg_expr(self, col_idx, agg):
        # TODO: track column dtype and compute aggregate dtype,
        # actually INTEGER works for floats too with the correct result,
        # so not a big issue right now
        res_type = OpExprType(int, True)
        return AggregateExpr(agg, [col_idx], res_type, False)

    def _process_transform(self, op):
        frame = op.input[0]
        fields = []
        exprs = []
        if op.keep_index and frame._index_cols is not None:
            fields += frame._index_cols
            exprs += [self._ref(frame, col) for col in frame._index_cols]

        fields += op.exprs.keys()
        exprs += self._translate(op.exprs.values())

        self._push(CalciteProjectionNode(fields, exprs))

    def _process_join(self, op):
        left = op.input[0]
        right = op.input[1]

        assert (
            op.on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        assert (
            op.on in left._table_cols and op.on in right._table_cols
        ), "Only cases when both frames contain key column are supported"

        """ Join, only equal-join supported """
        res_type = OpExprType(bool, False)
        condition = OpExpr(
            "=", [self._ref(left, op.on), self._ref(right, op.on)], res_type
        )
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
        conflicting_list = list(set(left.columns) & set(right.columns))
        """First goes 'on' column then all left columns(+suffix for conflicting names)
        but 'on' then all right columns(+suffix for conflicting names) but 'on'"""
        for c in left.columns:
            if c != op.on:
                suffix = op.suffixes[0] if c in conflicting_list else ""
                fields.append(c + suffix)
                exprs.append(self._ref(left, c))
            else:
                fields.insert(0, c)
                exprs.insert(0, self._ref(left, c))

        for c in right.columns:
            if c != op.on:
                suffix = op.suffixes[1] if c in conflicting_list else ""
                fields.append(c + suffix)
                exprs.append(self._ref(right, c))

        self._push(CalciteProjectionNode(fields, exprs))

        # TODO: current input translation system doesn't work here
        # because there is no frame to reference for index computation.
        # We should build calcite tree to keep references to input
        # nodes and keep scheme in calcite nodes. For now just use
        # known index 0.
        if op.sort is True:
            """Sort by key column"""
            collation = [CalciteCollation(CalciteInputIdxExpr(0))]
            self._push(CalciteSortNode(collation))

    def _process_union(self, op):
        self._push(CalciteUnionNode(self._input_ids(), True))
