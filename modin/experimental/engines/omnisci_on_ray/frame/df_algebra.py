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

import abc
from .calcite_algebra import *
from .expr import *


class DFAlgNode(abc.ABC):
    """Base class for all DataFrame Algebra nodes"""

    @abc.abstractmethod
    def copy(self):
        pass

    def to_calcite(self):
        CalciteBaseNode.reset_id()
        res = []
        self._to_calcite(res)
        return res

    @abc.abstractmethod
    def _to_calcite(self, out_nodes):
        pass

    def walk_dfs(self, cb, *args, **kwargs):
        if hasattr(self, "input"):
            for i in self.input:
                i._op.walk_dfs(cb, *args, **kwargs)
        cb(self, *args, **kwargs)

    def collect_partitions(self):
        partitions = []
        self.walk_dfs(lambda a, b: a._append_partitions(b), partitions)
        return partitions

    def collect_frames(self):
        frames = []
        self.walk_dfs(lambda a, b: a._append_frames(b), frames)
        return frames

    def _append_partitions(self, partitions):
        pass

    def _append_frames(self, frames):
        pass

    def _input_to_calcite(self, out_nodes):
        res = []
        if hasattr(self, "input"):
            for i in self.input:
                i._op._to_calcite(out_nodes)
                res.append(out_nodes[-1].id)
        return res

    def dump(self, prefix=""):
        self._print(prefix)

    @abc.abstractmethod
    def _print(self, prefix):
        pass

    def _print_input(self, prefix):
        if hasattr(self, "input"):
            for i, node in enumerate(self.input):
                print("{}input[{}]:".format(prefix, i))
                node._op._print(prefix + "  ")


class FrameNode(DFAlgNode):
    """FrameNode holds a list of Ray object ids for frame partitions"""

    def __init__(self, modin_frame):
        self.modin_frame = modin_frame

    def copy(self):
        return FrameNode(self.modin_frame)

    def _to_calcite(self, out_nodes):
        node = CalciteScanNode(self.modin_frame)
        out_nodes.append(node)

    def _append_partitions(self, partitions):
        partitions += self.modin_frame._partitions.flatten()

    def _append_frames(self, frames):
        frames.append(self.modin_frame)

    def _print(self, prefix):
        print("{}FrameNode({})".format(prefix, self.modin_frame))


class MaskNode(DFAlgNode):
    def __init__(
        self,
        base,
        row_indices=None,
        row_numeric_idx=None,
        col_indices=None,
        col_numeric_idx=None,
    ):
        self.input = [base]
        self.row_indices = row_indices
        self.row_numeric_idx = row_numeric_idx
        self.col_indices = col_indices
        self.col_numeric_idx = col_numeric_idx

    def copy(self):
        return MaskNode(
            self.input[0],
            self.row_indices,
            self.row_numeric_idx,
            self.col_indices,
            self.col_numeric_idx,
        )

    def _to_calcite(self, out_nodes):
        if self.row_indices is not None:
            raise NotImplementedError("row indices masking is not yet supported")

        frame = self.input[0]

        self._input_to_calcite(out_nodes)

        # select rows by rowid
        if self.row_numeric_idx is not None:
            # rowid is an additional virtual column which is always in
            # the end of the list
            rowid_col = InputRefExpr(len(frame._table_cols))
            condition = build_row_idx_filter_expr(self.row_numeric_idx, rowid_col)
            filter_node = CalciteFilterNode(condition)
            out_nodes.append(filter_node)

        # make projection
        fields = []
        exprs = []
        if frame._index_cols is not None:
            fields += frame._index_cols
            exprs += [InputRefExpr(i) for i in range(0, len(frame._index_cols))]

        if self.col_indices is not None or self.col_numeric_idx is not None:
            if self.col_indices is not None:
                fields += to_list(self.col_indices)
                exprs += [
                    InputRefExpr(frame._table_cols.index(col))
                    for col in self.col_indices
                ]
            elif self.col_numeric_idx is not None:
                offs = len(exprs)
                fields += frame.columns[self.col_numeric_idx].tolist()
                exprs += [InputRefExpr(x + offs) for x in self.col_numeric_idx]
        else:
            offs = len(exprs)
            fields += to_list(frame.columns)
            exprs += [InputRefExpr(i + offs) for i in range(0, len(frame.columns))]

        node = CalciteProjectionNode(fields, exprs)
        out_nodes.append(node)

    def _print(self, prefix):
        print("{}MaskNode:".format(prefix))
        if self.row_indices is not None:
            print("{}  row_indices: {}".format(prefix, self.row_indices))
        if self.row_numeric_idx is not None:
            print("{}  row_numeric_idx: {}".format(prefix, self.row_numeric_idx))
        if self.col_indices is not None:
            print("{}  col_indices: {}".format(prefix, self.col_indices))
        if self.col_numeric_idx is not None:
            print("{}  col_numeric_idx: {}".format(prefix, self.col_numeric_idx))
        self._print_input(prefix + "  ")


class GroupbyAggNode(DFAlgNode):
    def __init__(self, base, by, agg, groupby_opts):
        self.by = by
        self.agg = agg
        self.groupby_opts = groupby_opts
        self.input = [base]

    def copy(self):
        return GroupbyAggNode(self.input[0], self.by, self.agg, self.groupby_opts)

    def _create_agg_expr(self, col_idx, agg):
        # TODO: track column dtype and compute aggregate dtype,
        # actually INTEGER works for floats too with the correct result,
        # so not a big issue right now
        res_type = OpExprType(int, True)
        return AggregateExpr(agg, [col_idx], res_type, False)

    def _to_calcite(self, out_nodes):
        self._input_to_calcite(out_nodes)

        orig_cols = self.input[0].columns.tolist()
        table_cols = self.input[0]._table_cols

        # Wee need a projection to be aggregation op
        if not isinstance(out_nodes[-1], CalciteProjectionNode):
            proj = CalciteProjectionNode(
                table_cols, [InputRefExpr(i) for i in range(0, len(table_cols))]
            )
            out_nodes.append(proj)

        fields = []
        group = []
        aggs = []
        for col in self.by:
            fields.append(col)
            group.append(table_cols.index(col))
        for col, agg in self.agg.items():
            if isinstance(agg, list):
                for agg_val in agg:
                    fields.append(col + " " + agg_val)
                    aggs.append(self._create_agg_expr(table_cols.index(col), agg_val))
            else:
                fields.append(col)
                aggs.append(self._create_agg_expr(table_cols.index(col), agg))

        out_nodes.append(CalciteAggregateNode(fields, group, aggs))

        if self.groupby_opts["sort"]:
            collation = [CalciteCollation(col) for col in group]
            out_nodes.append(CalciteSortNode(collation))

    def _print(self, prefix):
        print("{}AggNode:".format(prefix))
        print("{}  by: {}".format(prefix, self.by))
        print("{}  agg: {}".format(prefix, self.agg))
        print("{}  groupby_opts: {}".format(prefix, self.groupby_opts))
        self._print_input(prefix + "  ")


class TransformNode(DFAlgNode):
    """Make simple column transformations.

    Args:
        base - frame to transform
        exprs - dictionary with new column names mapped to expressions
        keep_index - if True then keep all index columns (if any),
            otherwise drop them
    """

    def __init__(self, base, exprs, keep_index=True):
        self.exprs = exprs
        self.input = [base]
        self.keep_index = keep_index

    def copy(self):
        return TransformNode(self.input[0], self.exprs)

    def _to_calcite(self, out_nodes):
        self._input_to_calcite(out_nodes)

        frame = self.input[0]
        fields = []
        exprs = []
        if self.keep_index and frame._index_cols is not None:
            fields += frame._index_cols
            exprs += [InputRefExpr(i) for i in range(0, len(frame._index_cols))]

        fields += self.exprs.keys()
        exprs += self.exprs.values()

        node = CalciteProjectionNode(fields, exprs)
        out_nodes.append(node)

    def _print(self, prefix):
        print("{}TransformNode:".format(prefix))
        for k, v in self.exprs.items():
            print("{}  {}: {}".format(prefix, k, v))
        self._print_input(prefix + "  ")


class JoinNode(DFAlgNode):
    def __init__(
        self, left, right, how="inner", on=None, sort=False, suffixes=("_x", "_y")
    ):
        self.input = [left, right]
        self.how = how
        self.on = on
        self.sort = sort
        self.suffixes = suffixes

    def copy(self):
        return JoinNode(self.input[0], self.input[1], self.how, self.on, self.sort,)

    def _to_calcite(self, out_nodes):

        left = self.input[0]
        right = self.input[1]

        assert (
            self.on is not None
        ), "Merge with unspecified 'on' parameter is not supported in the engine"

        assert (
            self.on in left._table_cols and self.on in right._table_cols
        ), "Only cases when both frames contain key column are supported"

        left_on_pos = left._table_cols.index(self.on)
        right_on_pos = right._table_cols.index(self.on)

        """Frames scan"""
        inputs = self._input_to_calcite(out_nodes)
        assert len(inputs) > 1, "Unexpected number of DFAlgNodes"
        left_node_id = inputs[0]
        right_node_id = inputs[1]

        """ Join, only equal-join supported """
        res_type = OpExprType(bool, True)
        """We should remember about rowid for left's projection, that's why +1"""
        condition = OpExpr(
            "=",
            [
                InputRefExpr(left_on_pos),
                InputRefExpr(right_on_pos + len(left._table_cols) + 1),
            ],
            res_type,
        )
        node = CalciteJoinNode(
            left_id=left_node_id,
            right_id=right_node_id,
            how=self.how,
            condition=condition,
        )
        out_nodes.append(node)

        """Projection for both frames"""
        fields = []
        exprs = []
        conflicting_list = list(set(left._table_cols) & set(right._table_cols))
        """First goes 'on' column then all left columns(+suffix for conflicting names) but 'on' 
        then all right columns(+suffix for conflicting names) but 'on'"""
        expr_index = 0
        for c in left._table_cols:
            if c != self.on:
                suffix = self.suffixes[0] if c in conflicting_list else ""
                fields.append(c + suffix)
                exprs.append(InputRefExpr(expr_index))
            else:
                fields.insert(0, c)
                exprs.insert(0, InputRefExpr(expr_index))
            expr_index += 1

        for c in right._table_cols:
            if c != self.on:
                suffix = self.suffixes[1] if c in conflicting_list else ""
                fields.append(c + suffix)
                exprs.append(InputRefExpr(expr_index + 1))
            expr_index += 1

        node = CalciteProjectionNode(fields, exprs)
        out_nodes.append(node)

        if self.sort is True:
            """Sort by key column"""
            collation = [CalciteCollation(0)]
            out_nodes.append(CalciteSortNode(collation))

    def _print(self, prefix):
        print("{}JoinNode:".format(prefix))
        print("{}  How: {}".format(prefix, self.how))
        print("{}  On: {}".format(prefix, self.on))
        print("{}  Sorting: {}".format(prefix, self.sort))
        self._print_input(prefix + "  ")


class UnionNode(DFAlgNode):
    """Concat frames by axis=0, all frames should be aligned."""

    def __init__(self, frames):
        self.input = frames

    def copy(self):
        return UnionNode(self.input)

    def _to_calcite(self, out_nodes):
        inputs = self._input_to_calcite(out_nodes)
        node = CalciteUnionNode(inputs, True)
        out_nodes.append(node)

    def _print(self, prefix):
        print("{}UnionNode:".format(prefix))
        self._print_input(prefix + "  ")


def to_list(indices):
    if isinstance(indices, list):
        return indices
    return indices.tolist()
