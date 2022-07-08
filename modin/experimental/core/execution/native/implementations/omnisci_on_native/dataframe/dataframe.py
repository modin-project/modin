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

"""Module provides ``OmnisciOnNativeDataframe`` class implementing lazy frame."""

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
from modin.experimental.core.storage_formats.omnisci.query_compiler import (
    DFAlgQueryCompiler,
)
from ..partitioning.partition_manager import OmnisciOnNativeDataframePartitionManager

from pandas.core.indexes.api import ensure_index, Index, MultiIndex, RangeIndex
from pandas.core.dtypes.common import get_dtype, is_list_like, is_bool_dtype
from modin.error_message import ErrorMessage
from modin.pandas.indexing import is_range_like
import pandas as pd
from typing import List, Hashable, Optional, Tuple, Union

from ..df_algebra import (
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
from ..expr import (
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
from modin.pandas.utils import check_both_not_none


class OmnisciOnNativeDataframe(PandasDataframe):
    """
    Lazy dataframe based on Arrow table representation and embedded OmniSci storage format.

    Currently, materialized dataframe always has a single partition. This partition
    can hold either Arrow table or pandas dataframe.

    Operations on a dataframe are not instantly executed and build an operations
    tree instead. When frame's data is accessed this tree is transformed into
    a query which is executed in OmniSci storage format. In case of simple transformations
    Arrow API can be used instead of OmniSci storage format.

    Since frames are used as an input for other frames, all operations produce
    new frames and are not executed in-place.

    Parameters
    ----------
    partitions : np.ndarray, optional
        Partitions of the frame.
    index : pandas.Index, optional
        Index of the frame to be used as an index cache. If None then will be
        computed on demand.
    columns : pandas.Index, optional
        Columns of the frame.
    row_lengths : np.ndarray, optional
        Partition lengths. Should be None if lengths are unknown.
    column_widths : np.ndarray, optional
        Partition widths. Should be None if widths are unknown.
    dtypes : pandas.Index, optional
        Column data types.
    op : DFAlgNode, optional
        A tree describing how frame is computed. For materialized frames it
        is always ``FrameNode``.
    index_cols : list of str, optional
        A list of columns included into the frame's index. None value means
        a default index (row id is used as an index).
    uses_rowid : bool, default: False
        True for frames which require access to the virtual 'rowid' column
        for its execution.
    force_execution_mode : str or None
        Used by tests to control frame's execution process.
    has_unsupported_data : bool
        True for frames holding data not supported by Arrow or OmniSci storage format.

    Attributes
    ----------
    id : int
        ID of the frame. Used for debug prints only.
    _op : DFAlgNode
        A tree to be used to compute the frame. For materialized frames it is
        always ``FrameNode``.
    _partitions : numpy.ndarray or None
        Partitions of the frame. For materialized dataframes it holds a single
        partition. None for frames requiring execution.
    _index_cols : list of str or None
        Names of index columns. None for default index. Index columns have mangled
        names to handle labels which cannot be directly used as an OmniSci table
        column name (e.g. non-string labels, SQL keywords etc.).
    _table_cols : list of str
        A list of all frame's columns. It includes index columns if any. Index
        columns are always in the head of the list.
    _index_cache : pandas.Index or None
        Materialized index of the frame or None when index is not materialized.
    _has_unsupported_data : bool
        True for frames holding data not supported by Arrow or OmniSci storage format.
        Operations on such frames are not allowed and should be defaulted
        to pandas instead.
    _dtypes : pandas.Series
        Column types.
    _uses_rowid : bool
        True for frames which require access to the virtual 'rowid' column
        for its execution.
    _force_execution_mode : str or None
        Used by tests to control frame's execution process. Value "lazy"
        is used to raise RuntimeError if execution is triggered for the frame.
        Value "arrow" is used to raise RuntimeError execution is triggered
        and cannot be done using Arrow API (have to use OmniSci for execution).
    """

    _query_compiler_cls = DFAlgQueryCompiler
    _partition_mgr_cls = OmnisciOnNativeDataframePartitionManager

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
                self._partitions[0][
                    0
                ] = self._partition_mgr_cls._partition_class.put_arrow(new_table)

        self._uses_rowid = uses_rowid
        self._force_execution_mode = force_execution_mode

    def id_str(self):
        """
        Return string identifier of the frame.

        Used for debug dumps.

        Returns
        -------
        str
        """
        return f"frame${self.id}"

    def get_dtype(self, col):
        """
        Get data type for a column.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        dtype
        """
        # If we search for an index column type in a MultiIndex then we need to
        # extend index column names to tuples.
        if isinstance(self._dtypes.index, MultiIndex) and not isinstance(col, tuple):
            return self._dtypes[(col, *([""] * (self._dtypes.index.nlevels - 1)))]
        return self._dtypes[col]

    def ref(self, col):
        """
        Return an expression referencing a frame's column.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        InputRefExpr
        """
        if col == "__rowid__":
            return InputRefExpr(self, col, get_dtype(int))
        return InputRefExpr(self, col, self.get_dtype(col))

    def mask(
        self,
        row_labels: Optional[List[Hashable]] = None,
        row_positions: Optional[List[int]] = None,
        col_labels: Optional[List[Hashable]] = None,
        col_positions: Optional[List[int]] = None,
    ) -> "OmnisciOnNativeDataframe":
        """
        Mask rows and columns in the dataframe.

        Allow users to perform selection and projection on the row and column labels (named notation),
        in addition to the row and column number (positional notation).

        Parameters
        ----------
        row_labels : list of hashable, optional
            The row labels to extract.
        row_positions : list of int, optional
            The row positions to extract.
        col_labels : list of hashable, optional
            The column labels to extract.
        col_positions : list of int, optional
            The column positions to extract.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.

        Notes
        -----
        If both `row_labels` and `row_positions` are provided, a ValueError is raised.
        The same rule applies for `col_labels` and `col_positions`.
        """
        if check_both_not_none(row_labels, row_positions):
            raise ValueError(
                "Both row_labels and row_positions were provided - please provide only one of row_labels and row_positions."
            )
        if check_both_not_none(col_labels, col_positions):
            raise ValueError(
                "Both col_labels and col_positions were provided - please provide only one of col_labels and col_positions."
            )
        base = self

        if col_labels is not None or col_positions is not None:
            if col_labels is not None:
                new_columns = col_labels
            elif col_positions is not None:
                new_columns = base.columns[col_positions]
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

        if row_labels is not None or row_positions is not None:
            op = MaskNode(base, row_labels=row_labels, row_positions=row_positions)
            return self.__constructor__(
                columns=base.columns,
                dtypes=base._dtypes,
                op=op,
                index_cols=self._index_cols,
                force_execution_mode=self._force_execution_mode,
            )

        return base

    def _has_arrow_table(self):
        """
        Return True for materialized frame with Arrow table.

        Returns
        -------
        bool
        """
        if not isinstance(self._op, FrameNode):
            return False
        return all(p.arrow_table for p in self._partitions.flatten())

    def _dtypes_for_cols(self, new_index, new_columns):
        """
        Return dtypes index for a specified set of index and data columns.

        Parameters
        ----------
        new_index : pandas.Index or list
            Index columns.
        new_columns : pandas.Index or list
            Data Columns.

        Returns
        -------
        pandas.Index
        """
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
        """
        Return dtypes for expressions.

        Parameters
        ----------
        exprs : dict
            Expression to get types for.

        Returns
        -------
        list of dtype
        """
        return [expr._dtype for expr in exprs.values()]

    def groupby_agg(self, by, axis, agg, groupby_args, **kwargs):
        """
        Groupby with aggregation operation.

        Parameters
        ----------
        by : DFAlgQueryCompiler or list-like of str
            Grouping keys.
        axis : {0, 1}
            Only rows groupby is supported, so should be 0.
        agg : str or dict
            Aggregates to compute.
        groupby_args : dict
            Additional groupby args.
        **kwargs : dict
            Keyword args. Currently ignored.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
                by_frame = self.mask(col_labels=by_cols)
                if by_frames:
                    by_frame = by_frame.concat(
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

        drop = kwargs.get("drop", True)
        as_index = groupby_args.get("as_index", True)
        groupby_cols = by_frame.columns
        if isinstance(agg, dict):
            agg_cols = agg.keys()
        elif not drop:
            # If 'by' data came from a different frame then 'self-aggregation'
            # columns are more prioritized.
            agg_cols = self.columns
        else:
            agg_cols = [col for col in self.columns if col not in groupby_cols]

        # Mimic pandas behavior: pandas does not allow for aggregation to be empty
        # in case of multi-column 'by'.
        if not as_index and len(agg_cols) == 0 and len(groupby_cols) > 1:
            agg_cols = self.columns

        # Create new base where all required columns are computed. We don't allow
        # complex expressions to be a group key or an aggeregate operand.
        allowed_nodes = (FrameNode, TransformNode)
        if not isinstance(by_frame._op, allowed_nodes):
            raise NotImplementedError(
                "OmniSci doesn't allow complex expression to be a group key. "
                + f"The only allowed frame nodes are: {tuple(o.__name__ for o in allowed_nodes)}, "
                + f"met '{type(by_frame._op).__name__}'."
            )

        col_to_delete_template = "__delete_me_{name}"

        def generate_by_name(by):
            """Generate unuqie name for `by` column in the resulted frame."""
            if as_index:
                return f"__index__0_{by}"
            elif by in agg_cols:
                # Aggregation columns are more prioritized than the 'by' cols,
                # so in case of naming conflicts, we drop 'by' cols.
                return col_to_delete_template.format(name=by)
            else:
                return by

        exprs = OrderedDict(
            ((generate_by_name(col), by_frame.ref(col)) for col in groupby_cols)
        )
        groupby_cols = list(exprs.keys())
        exprs.update(((col, self.ref(col)) for col in agg_cols))
        exprs = translate_exprs_to_base(exprs, base)
        base_cols = Index.__new__(Index, data=exprs.keys(), dtype=self.columns.dtype)
        base = self.__constructor__(
            columns=base_cols,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(base, exprs, fold=True),
            index_cols=None,
            force_execution_mode=self._force_execution_mode,
        )

        new_columns = []
        index_cols = None

        # TODO: check performance changes after enabling 'dropna' and decide
        # is it worth it or not.
        # if groupby_args["dropna"]:
        #     base = base.dropna(subset=groupby_cols, how="any")

        if as_index:
            index_cols = groupby_cols.copy()
        else:
            new_columns = groupby_cols.copy()

        new_dtypes = base._dtypes[groupby_cols].tolist()

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

        if not as_index:
            col_to_delete = col_to_delete_template.format(name=".*")
            filtered_columns = [
                col
                for col in new_frame.columns
                if not (isinstance(col, str) and re.match(col_to_delete, col))
            ]
            if len(filtered_columns) != len(new_frame.columns):
                new_frame = new_frame.mask(col_labels=filtered_columns)
        return new_frame

    def agg(self, agg):
        """
        Perform specified aggregation along columns.

        Parameters
        ----------
        agg : str
            Name of the aggregation function to perform.

        Returns
        -------
        OmnisciOnNativeDataframe
            New frame containing the result of aggregation.
        """
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

    def fillna(self, value=None, method=None, axis=None, limit=None, downcast=None):
        """
        Replace NULLs operation.

        Parameters
        ----------
        value : dict or scalar, optional
            A value to replace NULLs with. Can be a dictionary to assign
            different values to columns.
        method : None, optional
            Should be None.
        axis : {0, 1}, optional
            Should be 0.
        limit : None, optional
            Should be None.
        downcast : None, optional
            Should be None.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        if axis != 0:
            raise NotImplementedError("fillna is supported for axis = 0 only")

        if limit is not None:
            raise NotImplementedError("fillna doesn't support limit yet")

        if downcast is not None:
            raise NotImplementedError("fillna doesn't support downcast yet")

        if method is not None:
            raise NotImplementedError("fillna doesn't support method yet")

        try:
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
        except TypeError:
            raise NotImplementedError(
                "Heterogenous data is not supported in OmniSci storage format"
            )

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

    def dropna(self, subset, how="any"):
        """
        Drop rows with NULLs.

        Parameters
        ----------
        subset : list of str
            Columns to check.
        how : {"any", "all"}, default: "any"
            Determine if row is removed from DataFrame, when we have
            at least one NULL or all NULLs.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        how_to_merge = {"any": "AND", "all": "OR"}

        # If index columns are not presented in the frame, then we have to create them
        # based on "rowid". This is needed because 'dropna' preserves index.
        if self._index_cols is None:
            base = self._materialize_rowid()
        else:
            base = self

        checks = [base.ref(col).is_not_null() for col in subset]
        condition = (
            checks[0]
            if len(checks) == 1
            else OpExpr(how_to_merge[how], checks, np.dtype("bool"))
        )
        result = base.__constructor__(
            columns=base.columns,
            dtypes=base._dtypes,
            op=FilterNode(base, condition),
            index_cols=base._index_cols,
            force_execution_mode=base._force_execution_mode,
        )
        return result

    def dt_extract(self, obj):
        """
        Extract a date or a time unit from a datetime value.

        Parameters
        ----------
        obj : str
            Datetime unit to extract.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Cast frame columns to specified types.

        Parameters
        ----------
        col_dtypes : dict
            Maps column names to new data types.
        **kwargs : dict
            Keyword args. Not used.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        columns = col_dtypes.keys()
        new_dtypes = self._dtypes.copy()
        for column in columns:
            dtype = col_dtypes[column]
            if (
                not isinstance(dtype, type(self._dtypes[column]))
                or dtype != self._dtypes[column]
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

    def join(
        self,
        other: "OmnisciOnNativeDataframe",
        how: Optional[Union[str, JoinType]] = JoinType.INNER,
        left_on: Optional[List[str]] = None,
        right_on: Optional[List[str]] = None,
        sort: Optional[bool] = False,
        suffixes: Optional[Tuple[str]] = ("_x", "_y"),
    ):
        """
        Join operation.

        Parameters
        ----------
        other : OmnisciOnNativeDataframe
            A frame to join with.
        how : str or modin.core.dataframe.base.utils.JoinType, default: JoinType.INNER
            A type of join.
        left_on : list of str, optional
            A list of columns for the left frame to join on.
        right_on : list of str, optional
            A list of columns for the right frame to join on.
        sort : bool, default: False
            Sort the result by join keys.
        suffixes : list-like of str, default: ("_x", "_y")
            A length-2 sequence of suffixes to add to overlapping column names
            of left and right operands respectively.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        how = JoinType(how)
        assert (
            left_on is not None and right_on is not None
        ), "Merge with unspecified 'left_on' or 'right_on' parameter is not supported in the engine"
        assert len(left_on) == len(
            right_on
        ), "'left_on' and 'right_on' lengths don't match"

        for col in left_on:
            assert col in self.columns, f"'left_on' references unknown column {col}"
        for col in right_on:
            assert col in other.columns, f"'right_on' references unknown column {col}"

        new_columns = []
        new_dtypes = []
        exprs = OrderedDict()

        left_conflicts = set(self.columns) & (set(other.columns) - set(right_on))
        right_conflicts = set(other.columns) & (set(self.columns) - set(left_on))
        conflicting_cols = left_conflicts | right_conflicts
        for c in self.columns:
            new_name = f"{c}{suffixes[0]}" if c in conflicting_cols else c
            new_columns.append(new_name)
            new_dtypes.append(self._dtypes[c])
            exprs[new_name] = self.ref(c)
        for c in other.columns:
            if c not in left_on or c not in right_on:
                new_name = f"{c}{suffixes[1]}" if c in conflicting_cols else c
                new_columns.append(new_name)
                new_dtypes.append(other._dtypes[c])
                exprs[new_name] = other.ref(c)

        condition = self._build_equi_join_condition(other, left_on, right_on)

        op = JoinNode(
            self,
            other,
            how=how.value,
            exprs=exprs,
            condition=condition,
        )

        new_columns = Index.__new__(Index, data=new_columns)
        res = self.__constructor__(
            dtypes=new_dtypes,
            columns=new_columns,
            op=op,
            force_execution_mode=self._force_execution_mode,
        )

        if sort:
            res = res.sort_rows(
                left_on, ascending=True, ignore_index=True, na_position="last"
            )

        return res

    def _build_equi_join_condition(self, rhs, lhs_cols, rhs_cols):
        """
        Build condition for equi-join.

        Parameters
        ----------
        rhs : OmnisciOnNativeDataframe
            Joined frame.
        lhs_cols : list
            Left frame columns to join by.
        rhs_cols : list
            Right frame columns to join by.

        Returns
        -------
        BaseExpr
        """
        condition = [
            self.ref(lhs_col).eq(rhs.ref(rhs_col))
            for lhs_col, rhs_col in zip(lhs_cols, rhs_cols)
        ]
        condition = (
            condition[0]
            if len(condition) == 1
            else OpExpr("AND", condition, get_dtype(bool))
        )
        return condition

    def _index_width(self):
        """
        Return a number of columns in the frame's index.

        Returns
        -------
        int
        """
        if self._index_cols is None:
            return 1
        return len(self._index_cols)

    def _union_all(
        self, axis, other_modin_frames, join="outer", sort=False, ignore_index=False
    ):
        """
        Concatenate frames' rows.

        Parameters
        ----------
        axis : {0, 1}
            Should be 0.
        other_modin_frames : list of OmnisciOnNativeDataframe
            Frames to concat.
        join : {"outer", "inner"}, default: "outer"
            How to handle columns with mismatched names.
            "inner" - drop such columns. "outer" - fill
            with NULLs.
        sort : bool, default: False
            Sort unaligned columns for 'outer' join.
        ignore_index : bool, default: False
            Ignore index columns.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
                    aligned_index_dtypes = [get_dtype(int)]
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

    def _join_by_index(self, other_modin_frames, how, sort, ignore_index):
        """
        Perform equi-join operation for multiple frames by index columns.

        Parameters
        ----------
        other_modin_frames : list of OmnisciOnNativeDataframe
            Frames to join with.
        how : str
            A type of join.
        sort : bool
            Sort the result by join keys.
        ignore_index : bool
            If True then reset column index for the resulting frame.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        if how == "outer":
            raise NotImplementedError("outer join is not supported in OmniSci engine")

        lhs = self._maybe_materialize_rowid()
        reset_index_names = False
        for rhs in other_modin_frames:
            rhs = rhs._maybe_materialize_rowid()
            if len(lhs._index_cols) != len(rhs._index_cols):
                raise NotImplementedError(
                    "join by indexes with different sizes is not supported"
                )

            reset_index_names = reset_index_names or lhs._index_cols != rhs._index_cols

            condition = lhs._build_equi_join_condition(
                rhs, lhs._index_cols, rhs._index_cols
            )

            exprs = lhs._index_exprs()
            new_columns = lhs.columns.to_list()
            for col in lhs.columns:
                exprs[col] = lhs.ref(col)
            for col in rhs.columns:
                # Handle duplicating column names here. When user specifies
                # suffixes to make a join, actual renaming is done in front-end.
                new_col_name = col
                rename_idx = 0
                while new_col_name in exprs:
                    new_col_name = f"{col}{rename_idx}"
                    rename_idx += 1
                exprs[new_col_name] = rhs.ref(col)
                new_columns.append(new_col_name)

            op = JoinNode(
                lhs,
                rhs,
                how=how,
                exprs=exprs,
                condition=condition,
            )

            new_columns = Index.__new__(
                Index, data=new_columns, dtype=self.columns.dtype
            )
            lhs = lhs.__constructor__(
                dtypes=lhs._dtypes_for_exprs(exprs),
                columns=new_columns,
                index_cols=lhs._index_cols,
                op=op,
                force_execution_mode=self._force_execution_mode,
            )

        if sort:
            lhs = lhs.sort_rows(
                lhs._index_cols,
                ascending=True,
                ignore_index=False,
                na_position="last",
            )

        if reset_index_names:
            lhs = lhs._reset_index_names()

        if ignore_index:
            new_columns = Index.__new__(RangeIndex, data=range(len(lhs.columns)))
            lhs = lhs._set_columns(new_columns)

        return lhs

    def concat(
        self,
        axis: Union[int, Axis],
        other_modin_frames: List["OmnisciOnNativeDataframe"],
        join: Optional[str] = "outer",
        sort: Optional[bool] = False,
        ignore_index: Optional[bool] = False,
    ):
        """
        Concatenate frames along a particular axis.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to concatenate along.
        other_modin_frames : list of OmnisciOnNativeDataframe
            Frames to concat.
        join : {"outer", "inner"}, default: "outer"
            How to handle mismatched indexes on other axis.
        sort : bool, default: False
            Sort non-concatenation axis if it is not already aligned
            when join is 'outer'.
        ignore_index : bool, default: False
            Ignore index along the concatenation axis.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        axis = Axis(axis)
        if not other_modin_frames:
            return self

        if axis == Axis.ROW_WISE:
            return self._union_all(
                axis.value, other_modin_frames, join, sort, ignore_index
            )

        base = self
        for frame in other_modin_frames:
            base = base._find_common_projections_base(frame)
            if base is None:
                return self._join_by_index(
                    other_modin_frames, how=join, sort=sort, ignore_index=ignore_index
                )

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
        """
        Perform binary operation.

        An arithmetic binary operation or a comparison operation to
        perform on columns.

        Parameters
        ----------
        other : scalar, list-like, or OmnisciOnNativeDataframe
            The second operand.
        op_name : str
            An operation to perform.
        **kwargs : dict
            Keyword args.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Insert a constant column.

        Parameters
        ----------
        loc : int
            Inserted column location.
        column : str
            Inserted column name.
        value : scalar
            Inserted column value.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Extract codes for a category column.

        The frame should have a single data column.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        assert len(self.columns) == 1
        assert self._dtypes[-1] == "category"

        col = self.columns[-1]
        exprs = self._index_exprs()
        col_expr = self.ref(col)
        code_expr = OpExpr("KEY_FOR_STRING", [col_expr], get_dtype("int32"))
        null_val = LiteralExpr(np.int32(-1))
        exprs[col] = build_if_then_else(
            col_expr.is_null(), null_val, code_expr, get_dtype("int32")
        )

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes,
            op=TransformNode(self, exprs),
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def sort_rows(self, columns, ascending, ignore_index, na_position):
        """
        Sort rows of the frame.

        Parameters
        ----------
        columns : str or list of str
            Sorting keys.
        ascending : bool or list of bool
            Sort order.
        ignore_index : bool
            Drop index columns.
        na_position : {"first", "last"}
            NULLs position.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Filter rows by a boolean key column.

        Parameters
        ----------
        key : OmnisciOnNativeDataframe
            A frame with a single bool data column used as a filter.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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

    def _maybe_materialize_rowid(self):
        """
        Materialize virtual 'rowid' column if frame uses it as an index.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        if self._index_cols is None:
            return self._materialize_rowid()
        return self

    def _materialize_rowid(self):
        """
        Materialize virtual 'rowid' column.

        Make a projection with a virtual 'rowid' column materialized
        as '__index__' column.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Build index column expressions.

        Build dictionary with references to all index columns
        mapped to index column names.

        Returns
        -------
        dict
        """
        exprs = OrderedDict()
        if self._index_cols:
            for col in self._index_cols:
                exprs[col] = self.ref(col)
        return exprs

    def _find_common_projections_base(self, rhs):
        """
        Try to find a common base for projections.

        Check if two frames can be expressed as `TransformNode`
        operations from the same input frame.

        Parameters
        ----------
        rhs : OmnisciOnNativeDataframe
            The second frame.

        Returns
        -------
        OmnisciOnNativeDataframe
            The found common projection base or None.
        """
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
        """
        Check if frame is a ``TranformNode`` operation.

        Returns
        -------
        bool
        """
        return isinstance(self._op, TransformNode)

    def _execute(self):
        """
        Materialize lazy frame.

        After this call frame always has ``FrameNode`` operation.
        """
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
            new_partitions[0][0] = self._partition_mgr_cls._partition_class.put_arrow(
                new_table
            )
        else:
            if self._force_execution_mode == "arrow":
                raise RuntimeError("forced arrow execution failed")

            new_partitions = self._partition_mgr_cls.run_exec_plan(
                self._op, self._index_cols, self._dtypes, self._table_cols
            )
        self._partitions = new_partitions
        self._op = FrameNode(self)

    def _require_executed_base(self):
        """
        Check if materialization of input frames is required.

        Returns
        -------
        bool
        """
        if isinstance(self._op, MaskNode):
            return True
        return self._uses_rowid

    def _run_sub_queries(self):
        """
        Run sub-queries for materialization.

        Materialize all frames in the execution tree which have to
        be materialized to materialize this frame.
        """
        if isinstance(self._op, FrameNode):
            return

        if self._require_executed_base():
            for op in self._op.input:
                op._execute()
        else:
            for frame in self._op.input:
                frame._run_sub_queries()

    def _can_execute_arrow(self):
        """
        Check for possibility of Arrow execution.

        Check if operation's tree for the frame can be executed using
        Arrow API instead of OmniSci query.

        Returns
        -------
        bool
        """
        if isinstance(self._op, FrameNode):
            return self._has_arrow_table()
        elif isinstance(self._op, MaskNode):
            return (
                self._op.row_labels is None and self._op.input[0]._can_execute_arrow()
            )
        elif isinstance(self._op, TransformNode):
            return (
                self._op.is_simple_select() and self._op.input[0]._can_execute_arrow()
            )
        elif isinstance(self._op, UnionNode):
            return all(frame._can_execute_arrow() for frame in self._op.input)
        else:
            return False

    def _execute_arrow(self):
        """
        Compute the frame data using Arrow API.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        if isinstance(self._op, FrameNode):
            if self._partitions.size == 0:
                return pyarrow.Table.from_pandas(pd.DataFrame({}))
            else:
                assert self._partitions.size == 1
                return self._partitions[0][0].get()
        elif isinstance(self._op, MaskNode):
            return self._op.input[0]._arrow_row_slice(self._op.row_positions)
        elif isinstance(self._op, TransformNode):
            return self._op.input[0]._arrow_select(self._op.exprs)
        elif isinstance(self._op, UnionNode):
            return self._arrow_concat(self._op.input)
        else:
            raise RuntimeError(f"Unexpected op ({type(self._op)}) in _execute_arrow")

    def _arrow_select(self, exprs):
        """
        Perform column selection on the frame using Arrow API.

        Parameters
        ----------
        exprs : dict
            Select expressions.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        table = self._execute_arrow()

        new_fields = []
        new_columns = []

        for col, expr in exprs.items():
            if expr.column == "__rowid__" and "F___rowid__" not in table.schema.names:
                arr = pyarrow.array(np.arange(0, table.num_rows))
                table = table.append_column("F___rowid__", arr)

            field = table.schema.field(f"F_{expr.column}")
            if col != expr.column:
                field = field.with_name(f"F_{col}")
            new_fields.append(field)
            new_columns.append(table.column(f"F_{expr.column}"))

        new_schema = pyarrow.schema(new_fields)

        return pyarrow.Table.from_arrays(new_columns, schema=new_schema)

    def _arrow_row_slice(self, row_positions):
        """
        Perform row selection on the frame using Arrow API.

        Parameters
        ----------
        row_positions : list of int
            Row positions to select.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        table = self._execute_arrow()

        if not isinstance(row_positions, slice) and not is_range_like(row_positions):
            if not isinstance(row_positions, (pyarrow.Array, np.ndarray, list)):
                row_positions = pyarrow.array(row_positions)
            return table.take(row_positions)

        if isinstance(row_positions, slice):
            row_positions = range(*row_positions.indices(table.num_rows))

        start, stop, step = (
            row_positions.start,
            row_positions.stop,
            row_positions.step,
        )

        if step == 1:
            return table.slice(start, len(row_positions))
        else:
            indices = np.arange(start, stop, step)
            return table.take(indices)

    @classmethod
    def _arrow_concat(cls, frames):
        """
        Concat frames' rows using Arrow API.

        Parameters
        ----------
        frames : list of OmnisciOnNativeDataframe
            Frames to concat.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        return pyarrow.concat_tables(frame._execute_arrow() for frame in frames)

    def _build_index_cache(self):
        """
        Materialize index and store it in the cache.

        Can only be called for materialized frames.
        """
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
        """
        Get the index of the frame in pandas format.

        Materializes the frame if required.

        Returns
        -------
        pandas.Index
        """
        self._execute()
        if self._index_cache is None:
            self._build_index_cache()
        return self._index_cache

    def _set_index(self, new_index):
        """
        Set new index for the frame.

        Parameters
        ----------
        new_index : pandas.Index
            New index.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        if not isinstance(new_index, (Index, MultiIndex)):
            raise NotImplementedError(
                "OmnisciOnNativeDataframe._set_index is not yet suported"
            )

        self._execute()

        assert self._partitions.size == 1
        obj = self._partitions[0][0].get()
        if isinstance(obj, pd.DataFrame):
            raise NotImplementedError(
                "OmnisciOnNativeDataframe._set_index is not yet suported"
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
        """
        Set the default index for the frame.

        Parameters
        ----------
        drop : bool
            If True then drop current index columns, otherwise
            make them data columns.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
                if isinstance(self.columns, MultiIndex) and not isinstance(name, tuple):
                    name = (name, *([""] * (self.columns.nlevels - 1)))
                exprs[name] = self.ref(c)
            for c in self.columns:
                if c in exprs:
                    raise ValueError(f"cannot insert {c}, already exists")
                exprs[c] = self.ref(c)
            new_columns = Index.__new__(
                Index,
                data=exprs.keys(),
                dtype="O",
                name=self.columns.names
                if isinstance(self.columns, MultiIndex)
                else self.columns.name,
            )
            return self.__constructor__(
                columns=new_columns,
                dtypes=self._dtypes_for_exprs(exprs),
                op=TransformNode(self, exprs),
                index_cols=None,
                force_execution_mode=self._force_execution_mode,
            )

    def _reset_index_names(self):
        """
        Reset names for all index columns.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        if self.has_multiindex():
            return self.set_index_names([None] * len(self._index_cols))
        return self.set_index_name(None)

    def _set_columns(self, new_columns):
        """
        Rename columns.

        Parameters
        ----------
        new_columns : list-like of str
            New column names.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Return column labels of the frame.

        Returns
        -------
        pandas.Index
        """
        return super(OmnisciOnNativeDataframe, self)._get_columns()

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """
        Get a DataFrame exchange protocol object representing data of the Modin DataFrame.

        Parameters
        ----------
        nan_as_null : bool, default: False
            A keyword intended for the consumer to tell the producer
            to overwrite null values in the data with ``NaN`` (or ``NaT``).
            This currently has no effect; once support for nullable extension
            dtypes is added, this value should be propagated to columns.
        allow_copy : bool, default: True
            A keyword that defines whether or not the library is allowed
            to make a copy of the data. For example, copying data would be necessary
            if a library supports strided buffers, given that this protocol
            specifies contiguous buffers. Currently, if the flag is set to ``False``
            and a copy is needed, a ``RuntimeError`` will be raised.

        Returns
        -------
        ProtocolDataframe
            A dataframe object following the dataframe exchange protocol specification.
        """
        if self._has_unsupported_data:
            ErrorMessage.default_to_pandas(message="`__dataframe__`")
            pd_df = self.to_pandas()
            if hasattr(pd_df, "__dataframe__"):
                return pd_df.__dataframe__()
            raise NotImplementedError(
                "OmniSci execution does not support exchange protocol if the frame contains data types "
                + "that are unsupported by OmniSci."
            )

        from ..exchange.dataframe_protocol.dataframe import OmnisciProtocolDataframe

        return OmnisciProtocolDataframe(
            self, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df: "ProtocolDataframe") -> "OmnisciOnNativeDataframe":
        """
        Convert a DataFrame implementing the dataframe exchange protocol to a Core Modin Dataframe.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        df : ProtocolDataframe
            The DataFrame object supporting the dataframe exchange protocol.

        Returns
        -------
        OmnisciOnNativeDataframe
            A new Core Modin Dataframe object.
        """
        if isinstance(df, cls):
            return df

        if not hasattr(df, "__dataframe__"):
            raise ValueError(
                "`df` does not support DataFrame exchange protocol, i.e. `__dataframe__` method"
            )

        from modin.core.dataframe.pandas.exchange.dataframe_protocol.from_dataframe import (
            from_dataframe_to_pandas,
        )

        # TODO: build a PyArrow table instead of a pandas DataFrame from the protocol object
        # as it's possible to do zero-copy with `cls.from_arrow`
        ErrorMessage.default_to_pandas(message="`from_dataframe`")
        pd_df = from_dataframe_to_pandas(df)
        return cls.from_pandas(pd_df)

    columns = property(_get_columns)
    index = property(_get_index)

    @property
    def dtypes(self):
        """
        Return column data types.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """
        if self._index_cols is not None:
            return self._dtypes[len(self._index_cols) :]
        return self._dtypes

    def has_multiindex(self):
        """
        Check for multi-index usage.

        Return True if the frame has a multi-index (index with
        multiple columns) and False otherwise.

        Returns
        -------
        bool
        """
        if self._index_cache is not None:
            return isinstance(self._index_cache, MultiIndex)
        return self._index_cols is not None and len(self._index_cols) > 1

    def get_index_name(self):
        """
        Get the name of the index column.

        Returns None for default index and multi-index.

        Returns
        -------
        str or None
        """
        if self._index_cols is None:
            return None
        if len(self._index_cols) > 1:
            return None
        return self._index_cols[0]

    def set_index_name(self, name):
        """
        Set new name for the index column.

        Shouldn't be called for frames with multi-index.

        Parameters
        ----------
        name : str or None
            New index name.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Get index column names.

        Returns
        -------
        list of str
        """
        if self.has_multiindex():
            return self._index_cols.copy()
        return [self.get_index_name()]

    def set_index_names(self, names):
        """
        Set index labels for frames with multi-index.

        Parameters
        ----------
        names : list of str
            New index labels.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        """
        Transform the frame to pandas format.

        Returns
        -------
        pandas.DataFrame
        """
        self._execute()

        if self._force_execution_mode == "lazy":
            raise RuntimeError("unexpected to_pandas triggered on lazy frame")

        df = self._partition_mgr_cls.to_pandas(self._partitions)

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
        """
        Demangle index column names to index labels.

        Parameters
        ----------
        cols : list of str
            Index column names.

        Returns
        -------
        list of str
            Demangled index names.
        """
        if len(cols) == 1:
            return self._index_name(cols[0])
        return [self._index_name(n) for n in cols]

    def _index_name(self, col):
        """
        Demangle index column name into index label.

        Parameters
        ----------
        col : str
            Index column name.

        Returns
        -------
        str
            Demangled index name.
        """
        if col == "__index__":
            return None

        match = re.search("__index__\\d+_(.*)", col)
        if match:
            name = match.group(1)
            if name in ("__None__", "__reduced__"):
                return None
            return name

        return col

    def _find_index_or_col(self, col):
        """
        Find a column name corresponding to a column or index label.

        Parameters
        ----------
        col : str
            A column or index label.

        Returns
        -------
        str
            A column name corresponding to a label.
        """
        if col in self.columns:
            return col

        if self._index_cols is not None:
            for idx_col in self._index_cols:
                if col == idx_col or re.match(f"__index__\\d+_{col}", idx_col):
                    return idx_col

        raise ValueError(f"Unknown column '{col}'")

    @classmethod
    def from_pandas(cls, df):
        """
        Build a frame from a `pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
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
        ) = cls._partition_mgr_cls.from_pandas(df, True)

        if len(unsupported_cols) > 0:
            ErrorMessage.single_warning(
                f"Frame contain columns with unsupported data-types: {unsupported_cols}. "
                + "All operations with this frame will be default to pandas!"
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
        """
        Return mangled index names for index labels.

        Mangled names are used for index columns because index
        labels cannot always be used as OmniSci table column
        names. E.e. label can be a non-string value or an
        unallowed string (empty strings, etc.) for a table column
        name.

        Parameters
        ----------
        names : list of str
            Index labels.

        Returns
        -------
        list of str
            Mangled names.
        """
        return [
            f"__index__{i}_{'__None__' if n is None else n}"
            for i, n in enumerate(names)
        ]

    @classmethod
    def from_arrow(cls, at, index_cols=None, index=None):
        """
        Build a frame from an Arrow table.

        Parameters
        ----------
        at : pyarrow.Table
            Source table.
        index_cols : list of str, optional
            List of index columns in the source table which
            are ignored in transformation.
        index : pandas.Index, optional
            An index to be used by the new frame. Should present
            if `index_cols` is not None.

        Returns
        -------
        OmnisciOnNativeDataframe
            The new frame.
        """
        (
            new_frame,
            new_lengths,
            new_widths,
            unsupported_cols,
        ) = cls._partition_mgr_cls.from_arrow(at, return_dims=True)

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
                + "All operations with this frame will be default to pandas!"
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
        """
        Check if an index is a trivial index, i.e. a sequence [0..n].

        Parameters
        ----------
        index : pandas.Index
            An index to check.

        Returns
        -------
        bool
        """
        if len(index) == 0:
            return True
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
