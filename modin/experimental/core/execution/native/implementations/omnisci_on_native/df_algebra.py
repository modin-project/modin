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

"""Module provides classes for lazy DataFrame algebra operations."""

import abc
from .expr import InputRefExpr
from collections import OrderedDict


class TransformMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to expressions used
    for their computation. This mapper is used to fold expressions
    from multiple ``TransformNode``-s into a single expression.

    Parameters
    ----------
    op : TransformNode
        Transformation used for mapping.

    Attributes
    ----------
    _op : TransformNode
        Transformation used for mapping.
    """

    def __init__(self, op):
        self._op = op

    def translate(self, col):
        """
        Translate column reference by its name.

        Parameters
        ----------
        col : str
            A name of the column to translate.

        Returns
        -------
        BaseExpr
            Translated expression.
        """
        if col == "__rowid__":
            return self._op.input[0].ref(col)
        return self._op.exprs[col]


class FrameMapper:
    """
    A helper class for ``InputMapper``.

    This class is used to map column references to another frame.
    This mapper is used to replace input frame in expressions.

    Parameters
    ----------
    frame : OmnisciOnNativeDataframe
        Target frame.

    Attributes
    ----------
    _frame : OmnisciOnNativeDataframe
        Target frame.
    """

    def __init__(self, frame):
        self._frame = frame

    def translate(self, col):
        """
        Translate column reference by its name.

        Parameters
        ----------
        col : str
            A name of the column to translate.

        Returns
        -------
        BaseExpr
            Translated expression.
        """
        return self._frame.ref(col)


class InputMapper:
    """
    Input reference mapper.

    This class is used for input translation/replacement in
    expressions via ``BaseExpr.translate_input`` method.

    Translation is performed using column mappers registered via
    `add_mapper` method. Each input frame can have at most one mapper.
    References to frames with no registered mapper are not translated.

    Attributes
    ----------
    _mappers : dict
        Column mappers to use for translation.
    """

    def __init__(self):
        self._mappers = {}

    def add_mapper(self, frame, mapper):
        """
        Register a mapper for a frame.

        Parameters
        ----------
        frame : OmnisciOnNativeDataframe
            A frame for which a mapper is registered.
        mapper : object
            A mapper to register.
        """
        self._mappers[frame] = mapper

    def translate(self, ref):
        """
        Translate column reference by its name.

        Parameters
        ----------
        ref : InputRefExpr
            A column reference to translate.

        Returns
        -------
        BaseExpr
            Translated expression.
        """
        if ref.modin_frame in self._mappers:
            return self._mappers[ref.modin_frame].translate(ref.column)
        return ref


class DFAlgNode(abc.ABC):
    """
    A base class for dataframe algebra tree node.

    A dataframe algebra tree is used to describe how dataframe is computed.

    Attributes
    ----------
    input : list of DFAlgNode, optional
        Holds child nodes.
    """

    @abc.abstractmethod
    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        DFAlgNode
        """
        pass

    def walk_dfs(self, cb, *args, **kwargs):
        """
        Perform a depth-first walk over a tree.

        Walk over an input in the depth-first order and call a callback function
        for each node.

        Parameters
        ----------
        cb : callable
            A callback function.
        *args : list
            Arguments for the callback.
        **kwargs : dict
            Keyword arguments for the callback.
        """
        if hasattr(self, "input"):
            for i in self.input:
                i._op.walk_dfs(cb, *args, **kwargs)
        cb(self, *args, **kwargs)

    def collect_partitions(self):
        """
        Collect all partitions participating in a tree.

        Returns
        -------
        list
            A list of collected partitions.
        """
        partitions = []
        self.walk_dfs(lambda a, b: a._append_partitions(b), partitions)
        return partitions

    def collect_frames(self):
        """
        Collect all frames participating in a tree.

        Returns
        -------
        list
            A list of collected frames.
        """
        frames = []
        self.walk_dfs(lambda a, b: a._append_frames(b), frames)
        return frames

    def _append_partitions(self, partitions):
        """
        Append all used by the node partitions to `partitions` list.

        The default implementation is no-op. This method should be
        overriden by all nodes referencing frame's partitions.

        Parameters
        ----------
        partitions : list
            Output list of partitions.
        """
        pass

    def _append_frames(self, frames):
        """
        Append all used by the node frames to `frames` list.

        The default implementation is no-op. This method should be
        overriden by all nodes referencing frames.

        Parameters
        ----------
        frames : list
            Output list of frames.
        """
        pass

    def __repr__(self):
        """
        Return a string representation of the tree.

        Returns
        -------
        str
        """
        return self.dumps()

    def dump(self, prefix=""):
        """
        Dump the tree.

        Parameters
        ----------
        prefix : str, default: ''
            A prefix to add at each string of the dump.
        """
        print(self.dumps(prefix))  # noqa: T201

    def dumps(self, prefix=""):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str, default: ''
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return self._prints(prefix)

    @abc.abstractmethod
    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        pass

    def _prints_input(self, prefix):
        """
        Return a string representation of node's operands.

        A helper method for `_prints` implementation in derived classes.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        res = ""
        if hasattr(self, "input"):
            for i, node in enumerate(self.input):
                if isinstance(node._op, FrameNode):
                    res += f"{prefix}input[{i}]: {node._op}\n"
                else:
                    res += f"{prefix}input[{i}]:\n" + node._op._prints(prefix + "  ")
        return res


class FrameNode(DFAlgNode):
    """
    A node to reference a materialized frame.

    Parameters
    ----------
    modin_frame : OmnisciOnNativeDataframe
        Referenced frame.

    Attributes
    ----------
    modin_frame : OmnisciOnNativeDataframe
        Referenced frame.
    """

    def __init__(self, modin_frame):
        self.modin_frame = modin_frame

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        FrameNode
        """
        return FrameNode(self.modin_frame)

    def _append_partitions(self, partitions):
        """
        Append all partitions of the referenced frame to `partitions` list.

        Parameters
        ----------
        partitions : list
            Output list of partitions.
        """
        partitions += self.modin_frame._partitions.flatten()

    def _append_frames(self, frames):
        """
        Append the referenced frame to `frames` list.

        Parameters
        ----------
        frames : list
            Output list of frames.
        """
        frames.append(self.modin_frame)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return f"{prefix}{self.modin_frame.id_str()}"


class MaskNode(DFAlgNode):
    """
    A filtering node which filters rows by index values or row id.

    Parameters
    ----------
    base : DFAlgNode
        A filtered frame.
    row_labels : list, optional
        List of row labels to select.
    row_positions : list of int, optional
        List of rows ids to select.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single filtered frame.
    row_labels : list or None
        List of row labels to select.
    row_positions : list of int or None
        List of rows ids to select.
    """

    def __init__(self, base, row_labels=None, row_positions=None):
        self.input = [base]
        self.row_labels = row_labels
        self.row_positions = row_positions

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        MaskNode
        """
        return MaskNode(self.input[0], self.row_labels, self.row_positions)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return (
            f"{prefix}MaskNode:\n"
            + f"{prefix}  row_labels: {self.row_labels}\n"
            + f"{prefix}  row_positions: {self.row_positions}\n"
            + self._prints_input(prefix + "  ")
        )


class GroupbyAggNode(DFAlgNode):
    """
    A node to represent a groupby aggregation operation.

    Parameters
    ----------
    base : DFAlgNode
        An aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single aggregated frame.
    by : list of str
        A list of columns used for grouping.
    agg_exprs : dict
        Aggregates to compute.
    groupby_opts : dict
        Additional groupby parameters.
    """

    def __init__(self, base, by, agg_exprs, groupby_opts):
        self.by = by
        self.agg_exprs = agg_exprs
        self.groupby_opts = groupby_opts
        self.input = [base]

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        GroupbyAggNode
        """
        return GroupbyAggNode(self.input[0], self.by, self.agg_exprs, self.groupby_opts)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return (
            f"{prefix}AggNode:\n"
            + f"{prefix}  by: {self.by}\n"
            + f"{prefix}  aggs: {self.agg_exprs}\n"
            + f"{prefix}  groupby_opts: {self.groupby_opts}\n"
            + self._prints_input(prefix + "  ")
        )


class TransformNode(DFAlgNode):
    """
    A node to represent a projection of a single frame.

    Provides expressions to compute each column of the projection.

    Parameters
    ----------
    base : DFAlgNode
        A transformed frame.
    exprs : dict
        Expressions for frame's columns computation.
    fold : bool, default: True
        If True and `base` is another `TransformNode`, then translate all
        expressions in `expr` to its base.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single projected frame.
    exprs : dict
        Expressions used to compute frame's columns.
    _original_refs : set
        Set of columns expressed with `InputRefExpr` prior folding.
    """

    def __init__(self, base, exprs, fold=True):
        self.exprs = exprs
        self.input = [base]
        self._original_refs = None
        if fold:
            self.fold()

    def fold(self):
        """
        Fold two ``TransformNode``-s.

        If base of this node is another `TransformNode`, then translate all
        expressions in `expr` to its base.
        """
        if isinstance(self.input[0]._op, TransformNode):
            self._original_refs = {
                col for col in self.exprs if isinstance(self.exprs[col], InputRefExpr)
            }
            self.input[0] = self.input[0]._op.input[0]
            self.exprs = translate_exprs_to_base(self.exprs, self.input[0])

    def is_original_ref(self, col):
        """
        Check original column expression type.

        Return True if `col` is an ``InputRefExpr`` expression or originally was
        an ``InputRefExpr`` expression before folding.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        bool
        """
        if self._original_refs is not None:
            return col in self._original_refs
        return isinstance(self.exprs[col], InputRefExpr)

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        TransformNode
        """
        return TransformNode(self.input[0], self.exprs, self.keep_index)

    def is_simple_select(self):
        """
        Check if transform node is a simple selection.

        Simple selection can only use InputRefExpr expressions.

        Returns
        -------
        bool
            True for simple select and False otherwise.
        """
        return all(isinstance(expr, InputRefExpr) for expr in self.exprs.values())

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        res = f"{prefix}TransformNode:\n"
        if self._original_refs is not None:
            res += f"{prefix}  Original refs: {self._original_refs}\n"
        for k, v in self.exprs.items():
            res += f"{prefix}  {k}: {v}\n"
        res += self._prints_input(prefix + "  ")
        return res


class JoinNode(DFAlgNode):
    """
    A node to represent a join of two frames.

    Parameters
    ----------
    left : DFAlgNode
        A left frame to join.
    right : DFAlgNode
        A right frame to join.
    how : str, default: "inner"
        A type of join.
    exprs : dict, default: None
        Expressions for the resulting frame's columns.
    condition : BaseExpr, default: None
        Join condition.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds joined frames. The first frame in the list is considered as
        the left join operand.
    how : str
        A type of join.
    exprs : dict
        Expressions for the resulting frame's columns.
    condition : BaseExpr
        Join condition.
    """

    def __init__(
        self,
        left,
        right,
        how="inner",
        exprs=None,
        condition=None,
    ):
        self.input = [left, right]
        self.how = how
        self.exprs = exprs
        self.condition = condition

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        JoinNode
        """
        return JoinNode(
            self.input[0],
            self.input[1],
            self.how,
            self.exprs,
            self.condition,
        )

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        exprs_str = ""
        for k, v in self.exprs.items():
            exprs_str += f"{prefix}    {k}: {v}\n"
        return (
            f"{prefix}JoinNode:\n"
            + f"{prefix}  Fields:\n"
            + exprs_str
            + f"{prefix}  How: {self.how}\n"
            + f"{prefix}  Condition: {self.condition}\n"
            + self._prints_input(prefix + "  ")
        )


class UnionNode(DFAlgNode):
    """
    A node to represent rows union of input frames.

    Parameters
    ----------
    frames : list of DFAlgNode
        Input frames.

    Attributes
    ----------
    input : list of DFAlgNode
        Input frames.
    """

    def __init__(self, frames):
        self.input = frames

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        UnionNode
        """
        return UnionNode(self.input)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return f"{prefix}UnionNode:\n" + self._prints_input(prefix + "  ")


class SortNode(DFAlgNode):
    """
    A sort node to order frame's rows in a specified order.

    Parameters
    ----------
    frame : DFAlgNode
        Sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single sorted frame.
    columns : list of str
        A list of key columns for a sort.
    ascending : bool
        Ascending or descending sort.
    na_position : {"first", "last"}
        "first" to put NULLs at the start of the result,
        "last" to put NULLs at the end of the result.
    """

    def __init__(self, frame, columns, ascending, na_position):
        self.input = [frame]
        self.columns = columns
        self.ascending = ascending
        self.na_position = na_position

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        SortNode
        """
        return SortNode(self.input[0], self.columns, self.ascending, self.na_position)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return (
            f"{prefix}SortNode:\n"
            + f"{prefix}  Columns: {self.columns}\n"
            + f"{prefix}  Ascending: {self.ascending}\n"
            + f"{prefix}  NULLs position: {self.na_position}\n"
            + self._prints_input(prefix + "  ")
        )


class FilterNode(DFAlgNode):
    """
    A node for generic rows filtering.

    For rows filter by row id a ``MaskNode`` should be preferred.

    Parameters
    ----------
    frame : DFAlgNode
        A filtered frame.
    condition : BaseExpr
        Filter condition.

    Attributes
    ----------
    input : list of DFAlgNode
        Holds a single filtered frame.
    condition : BaseExpr
        Filter condition.
    """

    def __init__(self, frame, condition):
        self.input = [frame]
        self.condition = condition

    def copy(self):
        """
        Make a shallow copy of the node.

        Returns
        -------
        FilterNode
        """
        return FilterNode(self.input[0], self.condition)

    def _prints(self, prefix):
        """
        Return a string representation of the tree.

        Parameters
        ----------
        prefix : str
            A prefix to add at each string of the dump.

        Returns
        -------
        str
        """
        return (
            f"{prefix}FilterNode:\n"
            + f"{prefix}  Condition: {self.condition}\n"
            + self._prints_input(prefix + "  ")
        )


def translate_exprs_to_base(exprs, base):
    """
    Fold expressions.

    Fold expressions with their input nodes until `base`
    frame is the only input frame.

    Parameters
    ----------
    exprs : dict
        Expressions to translate.
    base : OmnisciOnNativeDataframe
        Required input frame for translated expressions.

    Returns
    -------
    dict
        Translated expressions.
    """
    new_exprs = dict(exprs)

    frames = set()
    for k, v in new_exprs.items():
        v.collect_frames(frames)
    frames.discard(base)

    while len(frames) > 0:
        mapper = InputMapper()
        new_frames = set()
        for frame in frames:
            frame_base = frame._op.input[0]
            if frame_base != base:
                new_frames.add(frame_base)
            assert isinstance(frame._op, TransformNode)
            mapper.add_mapper(frame, TransformMapper(frame._op))

        for k, v in new_exprs.items():
            new_expr = new_exprs[k].translate_input(mapper)
            new_expr.collect_frames(new_frames)
            new_exprs[k] = new_expr

        new_frames.discard(base)
        frames = new_frames

    res = OrderedDict()
    for col in exprs.keys():
        res[col] = new_exprs[col]
    return res


def replace_frame_in_exprs(exprs, old_frame, new_frame):
    """
    Translate input expression replacing an input frame in them.

    Parameters
    ----------
    exprs : dict
        Expressions to translate.
    old_frame : OmnisciOnNativeDataframe
        An input frame to replace.
    new_frame : OmnisciOnNativeDataframe
        A new input frame to use.

    Returns
    -------
    dict
        Translated expressions.
    """
    mapper = InputMapper()
    mapper.add_mapper(old_frame, FrameMapper(new_frame))

    res = OrderedDict()
    for col in exprs.keys():
        res[col] = exprs[col].translate_input(mapper)
    return res
