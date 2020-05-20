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
        if hasattr(self, "input"):
            for i in self.input:
                i._op._to_calcite(out_nodes)

    def dump(self):
        self._print("")

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
            rowid_col = CalciteInputRefExpr(len(frame.columns))
            condition = build_calcite_row_idx_filter_expr(
                self.row_numeric_idx, rowid_col
            )
            filter_node = CalciteFilterNode(condition)
            out_nodes.append(filter_node)

        # make projection
        if self.col_indices is not None or self.col_numeric_idx is not None:
            fields = []
            exprs = []

            if self.col_indices is not None:
                fields = to_list(self.col_indices)
                exprs = [
                    CalciteInputRefExpr(frame.columns.tolist().index(col))
                    for col in self.col_indices
                ]
            elif self.col_numeric_idx is not None:
                fields = frame.columns[self.col_numeric_idx].tolist()
                exprs = [CalciteInputRefExpr(x) for x in self.col_numeric_idx]

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


def to_list(indices):
    if isinstance(indices, list):
        return indices
    return indices.tolist()
