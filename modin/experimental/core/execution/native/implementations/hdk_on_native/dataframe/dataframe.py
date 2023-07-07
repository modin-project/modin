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

"""Module provides ``HdkOnNativeDataframe`` class implementing lazy frame."""

import re
import numpy as np
from collections import OrderedDict

from typing import List, Hashable, Optional, Tuple, Union, Iterable

import pyarrow
from pyarrow.types import is_dictionary

import pandas as pd
from pandas._libs.lib import no_default
from pandas.core.indexes.api import Index, MultiIndex, RangeIndex
from pandas.core.dtypes.common import (
    get_dtype,
    is_list_like,
    is_bool_dtype,
    is_string_dtype,
    is_any_int_dtype,
    is_datetime64_dtype,
    is_categorical_dtype,
)

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.base.dataframe.utils import Axis, JoinType
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
)
from modin.experimental.core.storage_formats.hdk.query_compiler import (
    DFAlgQueryCompiler,
)
from .utils import (
    ColNameCodec,
    maybe_range,
    arrow_to_pandas,
    check_join_supported,
    check_cols_to_join,
    get_data_for_join_by_index,
    build_categorical_from_at,
)
from ..db_worker import DbTable
from ..hdk_worker import ImportedTable
from ..partitioning.partition_manager import HdkOnNativeDataframePartitionManager
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from modin.core.dataframe.pandas.utils import concatenate
from modin.core.dataframe.base.dataframe.utils import join_columns
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
from modin.pandas.utils import check_both_not_none
from modin.pandas.indexing import is_range_like

IDX_COL_NAME = ColNameCodec.IDX_COL_NAME
ROWID_COL_NAME = ColNameCodec.ROWID_COL_NAME
UNNAMED_IDX_COL_NAME = ColNameCodec.UNNAMED_IDX_COL_NAME
encode_col_name = ColNameCodec.encode
decode_col_name = ColNameCodec.decode
mangle_index_names = ColNameCodec.mangle_index_names
demangle_index_names = ColNameCodec.demangle_index_names


class HdkOnNativeDataframe(PandasDataframe):
    """
    Lazy dataframe based on Arrow table representation and embedded HDK storage format.

    Currently, materialized dataframe always has a single partition. This partition
    can hold either Arrow table or pandas dataframe.

    Operations on a dataframe are not instantly executed and build an operations
    tree instead. When frame's data is accessed this tree is transformed into
    a query which is executed in HDK storage format. In case of simple transformations
    Arrow API can be used instead of HDK storage format.

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
        True for frames holding data not supported by Arrow or HDK storage format.

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
        names to handle labels which cannot be directly used as an HDK table
        column name (e.g. non-string labels, SQL keywords etc.).
    _table_cols : list of str
        A list of all frame's columns. It includes index columns if any. Index
        columns are always in the head of the list.
    _index_cache : pandas.Index, callable or None
        Materialized index of the frame or None when index is not materialized.
        If ``callable() -> (pandas.Index, list of row lengths or None)`` type,
        then the calculation will be done in `__init__`.
    _has_unsupported_data : bool
        True for frames holding data not supported by Arrow or HDK storage format.
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
        The values "arrow" and "hdk" are used to force the corresponding
        execution mode.
    """

    _query_compiler_cls = DFAlgQueryCompiler
    _partition_mgr_cls = HdkOnNativeDataframePartitionManager

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
        assert partitions is None or (
            partitions.size == 1 and partitions[0][0] is not None
        )

        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1

        self._op = op
        self._index_cols = index_cols
        self._partitions = partitions
        self.set_index_cache(index)
        self.set_columns_cache(columns)
        # The following code assumes that the type of `columns` is pandas.Index.
        # The initial type of `columns` might be callable.
        columns = self._columns_cache.get()
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        self._has_unsupported_data = has_unsupported_data
        if self._op is None:
            self._op = FrameNode(self)

        if self._index_cols is not None:
            self._table_cols = self._index_cols + columns.tolist()
        else:
            self._table_cols = columns.tolist()

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
                    self.set_dtypes_cache(pd.Series(dtypes, index=dtype_index))
                else:
                    self.set_dtypes_cache(pd.Series(dtypes, index=self._table_cols))
            else:
                self.set_dtypes_cache(pd.Series(dtypes, index=columns))
        else:
            self.set_dtypes_cache(dtypes)

        self._uses_rowid = uses_rowid
        self._force_execution_mode = force_execution_mode

    def copy(
        self,
        partitions=no_default,
        index=no_default,
        columns=no_default,
        dtypes=no_default,
        op=no_default,
        index_cols=no_default,
        uses_rowid=no_default,
    ):
        """
        Copy this DataFrame.

        Parameters
        ----------
        partitions : np.ndarray, optional
            Partitions of the frame.
        index : pandas.Index or list, optional
            Index of the frame to be used as an index cache. If None then will be
            computed on demand.
        columns : pandas.Index or list, optional
            Columns of the frame.
        dtypes : pandas.Index or list, optional
            Column data types.
        op : DFAlgNode, optional
            A tree describing how frame is computed. For materialized frames it
            is always ``FrameNode``.
        index_cols : list of str, optional
            A list of columns included into the frame's index. None value means
            a default index (row id is used as an index).
        uses_rowid : bool, optional
            True for frames which require access to the virtual 'rowid' column
            for its execution.

        Returns
        -------
        HdkOnNativeDataframe
            A copy of this DataFrame.
        """
        if partitions is no_default:
            partitions = self._partitions
        if index is no_default:
            index = self.copy_index_cache()
        if columns is no_default:
            columns = self.copy_columns_cache()
        if op is no_default:
            op = self._op
        if dtypes is no_default:
            dtypes = self.copy_dtypes_cache()
        if index_cols is no_default:
            index_cols = self._index_cols
        if uses_rowid is no_default:
            uses_rowid = self._uses_rowid
        return self.__constructor__(
            partitions=partitions,
            index=index,
            columns=columns,
            row_lengths=self._row_lengths_cache,
            column_widths=self._column_widths_cache,
            dtypes=dtypes,
            op=op,
            index_cols=index_cols,
            uses_rowid=uses_rowid,
            force_execution_mode=self._force_execution_mode,
            has_unsupported_data=self._has_unsupported_data,
        )

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
        if col == ROWID_COL_NAME:
            return InputRefExpr(self, col, get_dtype(int))
        return InputRefExpr(self, col, self.get_dtype(col))

    def take_2d_labels_or_positional(
        self,
        row_labels: Optional[List[Hashable]] = None,
        row_positions: Optional[List[int]] = None,
        col_labels: Optional[List[Hashable]] = None,
        col_positions: Optional[List[int]] = None,
    ) -> "HdkOnNativeDataframe":
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
        HdkOnNativeDataframe
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
                expr = base.ref(col)
                if exprs.setdefault(col, expr) is not expr:
                    raise NotImplementedError(
                        "duplicate column names are not supported"
                    )
            dtypes = self._dtypes_for_exprs(exprs)
            base = self.__constructor__(
                columns=new_columns,
                dtypes=dtypes,
                op=TransformNode(base, exprs),
                index_cols=self._index_cols,
                force_execution_mode=self._force_execution_mode,
            )

        if row_labels is not None:
            raise NotImplementedError("Row labels masking is not yet supported")

        if row_positions is None:
            return base

        row_positions = maybe_range(row_positions)
        base = base._maybe_materialize_rowid()
        op = MaskNode(base, row_labels=row_labels, row_positions=row_positions)
        base = self.__constructor__(
            columns=base.columns,
            dtypes=base.copy_dtypes_cache(),
            op=op,
            index_cols=base._index_cols,
            force_execution_mode=base._force_execution_mode,
        )

        # Reverse the frame rows, if performing a reverse order selection via HDK.
        if (
            is_range_like(row_positions) and row_positions.step < 0
        ) and not base._can_execute_arrow():
            cols = base.columns
            table_cols = base._table_cols
            # Add the rowid column
            rowid_col = "__tmp_rowid__"
            while rowid_col in table_cols:
                rowid_col += "1"
            exprs = base._index_exprs()
            exprs[rowid_col] = base.ref(ROWID_COL_NAME)
            for col in cols:
                exprs[col] = base.ref(col)
            base = base.copy(
                columns=[rowid_col] + base.columns.tolist(),
                dtypes=base._dtypes_for_exprs(exprs),
                op=TransformNode(base, exprs),
                uses_rowid=True,
            )
            # Sort by the rowid column
            base = base.copy(op=SortNode(base, [rowid_col], [False], "last"))
            # Remove the rowid column
            exprs = OrderedDict()
            for col in table_cols:
                exprs[col] = base.ref(col)
            base = base.copy(
                columns=cols,
                dtypes=base._dtypes_for_exprs(exprs),
                op=TransformNode(base, exprs),
            )

        return base

    def _has_arrow_table(self):
        """
        Return True for materialized frame with Arrow table.

        Returns
        -------
        bool
        """
        return self._partitions is not None and isinstance(
            self._partitions[0][0].get(), pyarrow.Table
        )

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

    @_inherit_docstrings(PandasDataframe._maybe_update_proxies)
    def _maybe_update_proxies(self, dtypes, new_parent=None):
        if new_parent is not None:
            super()._maybe_update_proxies(dtypes, new_parent)
        if self._partitions is None:
            return
        table = self._partitions[0][0].get()
        if isinstance(table, ImportedTable):
            table = table.to_arrow()
        if isinstance(table, pyarrow.Table):
            super()._maybe_update_proxies(dtypes, new_parent=table)

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
        HdkOnNativeDataframe
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
                by_frame = self.take_2d_labels_or_positional(col_labels=by_cols)
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
                "HDK doesn't allow complex expression to be a group key. "
                + f"The only allowed frame nodes are: {tuple(o.__name__ for o in allowed_nodes)}, "
                + f"met '{type(by_frame._op).__name__}'."
            )

        if agg in ("head", "tail"):
            n = kwargs["agg_kwargs"]["n"]
            return self._groupby_head_tail(agg, n, groupby_cols)

        col_to_delete_template = "__delete_me_{name}"

        def generate_by_name(by):
            """Generate unuqie name for `by` column in the resulted frame."""
            if as_index:
                return f"{IDX_COL_NAME}0_{by}"
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
            op=TransformNode(base, exprs),
            index_cols=None,
            force_execution_mode=self._force_execution_mode,
        )

        new_columns = []
        index_cols = None

        # TODO: check performance changes after enabling 'dropna' and decide
        # is it worth it or not.
        if groupby_args["dropna"]:
            ErrorMessage.single_warning(
                "'dropna' is temporary disabled due to https://github.com/modin-project/modin/issues/2896"
            )
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
                new_frame = new_frame.take_2d_labels_or_positional(
                    col_labels=filtered_columns
                )
        return new_frame

    def _groupby_head_tail(
        self, agg: str, n: int, cols: Iterable[str]
    ) -> "HdkOnNativeDataframe":
        """
        Return first/last n rows of each group.

        Parameters
        ----------
        agg : {"head", "tail"}
        n : int
            If positive: number of entries to include from start/end of each group.
            If negative: number of entries to exclude from start/end of each group.
        cols : Iterable[str]
            Group by column names.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
        if isinstance(self._op, SortNode):
            base = self._op.input[0]
            order_keys = self._op.columns
            ascending = self._op.ascending
            na_pos = self._op.na_position.upper()
            fold = True  # Fold TransformNodes
        else:
            base = self._maybe_materialize_rowid()
            order_keys = base._index_cols[0:1]
            ascending = [True]
            na_pos = "FIRST"
            fold = base is self  # Do not fold if rowid is added
        if (n < 0) == (agg == "head"):  # Invert sorting
            ascending = [not a for a in ascending]
            na_pos = "FIRST" if na_pos == "LAST" else "LAST"
        partition_keys = [base.ref(col) for col in cols]
        order_keys = [base.ref(col) for col in order_keys]

        row_num_name = "__HDK_ROW_NUMBER__"
        row_num_op = OpExpr("ROW_NUMBER", [], get_dtype(int))
        row_num_op.set_window_opts(partition_keys, order_keys, ascending, na_pos)
        exprs = base._index_exprs()
        exprs.update((col, base.ref(col)) for col in base.columns)
        exprs[row_num_name] = row_num_op
        transform = base.copy(
            columns=list(base.columns) + [row_num_name],
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(base, exprs, fold),
        )

        if n < 0:
            cond = transform.ref(row_num_name).ge(-n + 1)
        else:
            cond = transform.ref(row_num_name).le(n)

        filter = transform.copy(op=FilterNode(transform, cond))
        exprs = filter._index_exprs()
        exprs.update((col, filter.ref(col)) for col in base.columns)
        return base.copy(op=TransformNode(filter, exprs))

    def agg(self, agg):
        """
        Perform specified aggregation along columns.

        Parameters
        ----------
        agg : str
            Name of the aggregation function to perform.

        Returns
        -------
        HdkOnNativeDataframe
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
        HdkOnNativeDataframe
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
                "Heterogenous data is not supported in HDK storage format"
            )

        new_op = TransformNode(self, exprs)
        dtypes = self._dtypes_for_exprs(exprs)
        new_frame = self.__constructor__(
            columns=self.columns,
            dtypes=dtypes,
            op=new_op,
            index=self._index_cache,
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
        HdkOnNativeDataframe
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
            dtypes=base.copy_dtypes_cache(),
            op=FilterNode(base, condition),
            index_cols=base._index_cols,
            force_execution_mode=base._force_execution_mode,
        )
        return result

    def isna(self, invert):
        """
        Detect missing values.

        Parameters
        ----------
        invert : bool

        Returns
        -------
        HdkOnNativeDataframe
        """
        expr = "is_not_null" if invert else "is_null"
        exprs = self._index_exprs()
        for col in self.columns:
            exprs[col] = getattr(self.ref(col), expr)()
        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index=self._index_cache,
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def invert(self):
        """
        Apply bitwise inverse to each column.

        Returns
        -------
        HdkOnNativeDataframe
        """
        exprs = self._index_exprs()
        for col in self.columns:
            exprs[col] = self.ref(col).invert()
        return self.copy(op=TransformNode(self, exprs))

    def dt_extract(self, obj):
        """
        Extract a date or a time unit from a datetime value.

        Parameters
        ----------
        obj : str
            Datetime unit to extract.

        Returns
        -------
        HdkOnNativeDataframe
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
            index=self._index_cache,
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
        HdkOnNativeDataframe
            The new frame.
        """
        columns = col_dtypes.keys()
        new_dtypes = self.copy_dtypes_cache()
        for column in columns:
            try:
                old_dtype = np.dtype(self._dtypes[column])
                new_dtype = np.dtype(col_dtypes[column])
            except TypeError:
                raise NotImplementedError(
                    f"Type conversion {self._dtypes[column]} -> {col_dtypes[column]}"
                )
            if old_dtype != new_dtype:
                # NotImplementedError is raised if the type cast is not supported.
                _get_common_dtype(new_dtype, self._dtypes[column])
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
            index=self._index_cache,
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def join(
        self,
        other: "HdkOnNativeDataframe",
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
        other : HdkOnNativeDataframe
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
        HdkOnNativeDataframe
            The new frame.
        """
        check_join_supported(how)
        assert (
            left_on is not None and right_on is not None
        ), "Merge with unspecified 'left_on' or 'right_on' parameter is not supported in the engine"
        assert len(left_on) == len(
            right_on
        ), "'left_on' and 'right_on' lengths don't match"

        if other is self:
            # To avoid the self-join failure - #5891
            if isinstance(self._op, FrameNode):
                other = self.copy()
            else:
                exprs = OrderedDict((c, self.ref(c)) for c in self._table_cols)
                other = self.__constructor__(
                    columns=self.columns,
                    dtypes=self._dtypes_for_exprs(exprs),
                    op=TransformNode(self, exprs),
                    index_cols=self._index_cols,
                    force_execution_mode=self._force_execution_mode,
                )

        orig_left_on = left_on
        orig_right_on = right_on
        left, left_on = check_cols_to_join("left_on", self, left_on)
        right, right_on = check_cols_to_join("right_on", other, right_on)
        for left_col, right_col in zip(left_on, right_on):
            left_dt = self._dtypes[left_col]
            right_dt = other._dtypes[right_col]
            if not (
                (is_any_int_dtype(left_dt) and is_any_int_dtype(right_dt))
                or (is_string_dtype(left_dt) and is_string_dtype(right_dt))
                or (is_datetime64_dtype(left_dt) and is_datetime64_dtype(right_dt))
                or (is_categorical_dtype(left_dt) and is_categorical_dtype(right_dt))
            ):
                raise NotImplementedError(
                    f"Join on columns of '{left_dt}' and '{right_dt}' dtypes"
                )

        # If either left_on or right_on has been changed, it means that there
        # are index columns in the list. Joining by index in this case.
        if (left_on is not orig_left_on) or (right_on is not orig_right_on):
            index_cols, exprs, new_dtypes, new_columns = get_data_for_join_by_index(
                self, other, how, orig_left_on, orig_right_on, sort, suffixes
            )
            ignore_index = False
        else:
            ignore_index = True
            index_cols = None
            exprs = OrderedDict()
            new_dtypes = []

            new_columns, left_renamer, right_renamer = join_columns(
                left.columns, right.columns, left_on, right_on, suffixes
            )
            for old_c, new_c in left_renamer.items():
                new_dtypes.append(left._dtypes[old_c])
                exprs[new_c] = left.ref(old_c)

            for old_c, new_c in right_renamer.items():
                new_dtypes.append(right._dtypes[old_c])
                exprs[new_c] = right.ref(old_c)

        condition = left._build_equi_join_condition(right, left_on, right_on)

        op = JoinNode(
            left,
            right,
            how=how,
            exprs=exprs,
            condition=condition,
        )

        res = left.__constructor__(
            dtypes=new_dtypes,
            columns=new_columns,
            index_cols=index_cols,
            op=op,
            force_execution_mode=self._force_execution_mode,
        )

        if sort:
            res = res.sort_rows(
                left_on, ascending=True, ignore_index=ignore_index, na_position="last"
            )

        return res

    def _build_equi_join_condition(self, rhs, lhs_cols, rhs_cols):
        """
        Build condition for equi-join.

        Parameters
        ----------
        rhs : HdkOnNativeDataframe
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
        other_modin_frames : list of HdkOnNativeDataframe
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
        HdkOnNativeDataframe
            The new frame.
        """
        index_cols = None
        col_name_to_dtype = OrderedDict()
        for col in self.columns:
            col_name_to_dtype[col] = self._dtypes[col]

        if join == "inner":
            for frame in other_modin_frames:
                for col in list(col_name_to_dtype):
                    if col not in frame.columns:
                        del col_name_to_dtype[col]
        elif join == "outer":
            for frame in other_modin_frames:
                for col in frame.columns:
                    if col not in col_name_to_dtype:
                        col_name_to_dtype[col] = frame._dtypes[col]
        else:
            raise NotImplementedError(f"Unsupported join type {join=}")

        frames = []
        for frame in [self] + other_modin_frames:
            # Empty frames are filtered out only in case of the outer join.
            if (
                join == "inner"
                or len(frame.columns) != 0
                or (frame.has_index_cache and len(frame.index) != 0)
                or (not frame.has_index_cache and frame.index_cols)
            ):
                if isinstance(frame._op, UnionNode):
                    frames.extend(frame._op.input)
                else:
                    frames.append(frame)

        if len(col_name_to_dtype) == 0:
            if len(frames) == 0:
                dtypes = pd.Series()
            elif ignore_index:
                index_cols = [UNNAMED_IDX_COL_NAME]
                dtypes = pd.Series([get_dtype(int)], index=index_cols)
            else:
                index_names = ColNameCodec.concat_index_names(frames)
                index_cols = list(index_names)
                dtypes = pd.Series(index_names.values(), index=index_cols)
        else:
            # Find common dtypes
            for frame in other_modin_frames:
                frame_dtypes = frame._dtypes
                for col in col_name_to_dtype:
                    if col in frame_dtypes:
                        col_name_to_dtype[col] = pd.core.dtypes.cast.find_common_type(
                            [col_name_to_dtype[col], frame_dtypes[col]]
                        )

            if sort:
                col_name_to_dtype = OrderedDict(
                    (col, col_name_to_dtype[col]) for col in sorted(col_name_to_dtype)
                )

            if ignore_index:
                table_col_name_to_dtype = col_name_to_dtype
            else:
                table_col_name_to_dtype = ColNameCodec.concat_index_names(frames)
                index_cols = list(table_col_name_to_dtype)
                table_col_name_to_dtype.update(col_name_to_dtype)

            dtypes = pd.Series(
                table_col_name_to_dtype.values(), index=table_col_name_to_dtype.keys()
            )
            for i, frame in enumerate(frames):
                frame_dtypes = frame._dtypes.get()
                if (
                    len(frame_dtypes) != len(dtypes)
                    or any(frame_dtypes.index != dtypes.index)
                    or any(frame_dtypes.values != dtypes.values)
                ):
                    exprs = OrderedDict()
                    uses_rowid = False
                    for col in table_col_name_to_dtype:
                        if col in frame_dtypes:
                            expr = frame.ref(col)
                        elif col == UNNAMED_IDX_COL_NAME:
                            if frame._index_cols is not None:
                                assert len(frame._index_cols) == 1
                                expr = frame.ref(frame._index_cols[0])
                            else:
                                uses_rowid = True
                                expr = frame.ref(ROWID_COL_NAME)
                        else:
                            expr = LiteralExpr(None, table_col_name_to_dtype[col])
                        if expr._dtype != table_col_name_to_dtype[col]:
                            expr = expr.cast(table_col_name_to_dtype[col])
                        exprs[col] = expr
                    frames[i] = frame.__constructor__(
                        columns=dtypes.index,
                        dtypes=dtypes,
                        uses_rowid=uses_rowid,
                        op=TransformNode(frame, exprs),
                        force_execution_mode=frame._force_execution_mode,
                    )

        return self.__constructor__(
            index_cols=index_cols,
            columns=col_name_to_dtype.keys(),
            dtypes=dtypes,
            op=UnionNode(frames, col_name_to_dtype, ignore_index),
            force_execution_mode=self._force_execution_mode,
        )

    def _join_by_index(self, other_modin_frames, how, sort, ignore_index):
        """
        Perform equi-join operation for multiple frames by index columns.

        Parameters
        ----------
        other_modin_frames : list of HdkOnNativeDataframe
            Frames to join with.
        how : str
            A type of join.
        sort : bool
            Sort the result by join keys.
        ignore_index : bool
            If True then reset column index for the resulting frame.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
        try:
            check_join_supported(how)
        except NotImplementedError as err:
            # The outer join is not supported by HDK, however, if all the frames
            # have a trivial index, we can simply concatenate the columns with arrow.
            if (frame := self._join_arrow_columns(other_modin_frames)) is not None:
                return frame
            raise err

        lhs = self._maybe_materialize_rowid()
        reset_index_names = False
        new_columns_dtype = self.columns.dtype
        for rhs in other_modin_frames:
            rhs = rhs._maybe_materialize_rowid()
            if len(lhs._index_cols) != len(rhs._index_cols):
                raise NotImplementedError(
                    "join by indexes with different sizes is not supported"
                )
            if new_columns_dtype != rhs.columns.dtype:
                new_columns_dtype = None

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
                Index, data=new_columns, dtype=new_columns_dtype
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

    def _join_arrow_columns(self, other_modin_frames):
        """
        Join arrow table columns.

        If all the frames have a trivial index and an arrow
        table in partitions, concatenate the table columns.

        Parameters
        ----------
        other_modin_frames : list of HdkOnNativeDataframe
            Frames to join with.

        Returns
        -------
        HdkOnNativeDataframe or None
        """
        frames = [self] + other_modin_frames
        if all(
            f._index_cols is None
            # Make sure all the frames have an arrow table in partitions.
            and isinstance(f._execute(), (DbTable, pyarrow.Table))
            for f in frames
        ):
            tables = [
                t
                if isinstance(t := f._partitions[0][0].get(), pyarrow.Table)
                else t.to_arrow()
                for f in frames
            ]
            column_names = [c for t in tables for c in t.column_names]
            if len(column_names) != len(set(column_names)):
                raise NotImplementedError("Duplicate column names")
            max_len = max(len(t) for t in tables)
            columns = [c for t in tables for c in t.columns]
            # Make all columns of the same length, if required.
            for i, col in enumerate(columns):
                if len(col) < max_len:
                    columns[i] = pyarrow.chunked_array(
                        col.chunks + [pyarrow.nulls(max_len - len(col), col.type)]
                    )
            return self.from_arrow(
                at=pyarrow.table(columns, column_names),
                columns=[c for f in frames for c in f.columns],
                encode_col_names=False,
            )
        return None

    def concat(
        self,
        axis: Union[int, Axis],
        other_modin_frames: List["HdkOnNativeDataframe"],
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
        other_modin_frames : list of HdkOnNativeDataframe
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
        HdkOnNativeDataframe
            The new frame.
        """
        axis = Axis(axis)
        if axis == Axis.ROW_WISE:
            return self._union_all(
                axis.value, other_modin_frames, join, sort, ignore_index
            )

        if not other_modin_frames:
            return self

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
        other : scalar, list-like, or HdkOnNativeDataframe
            The second operand.
        op_name : str
            An operation to perform.
        **kwargs : dict
            Keyword args.

        Returns
        -------
        HdkOnNativeDataframe
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
                index=self._index_cache,
                index_cols=self._index_cols,
                force_execution_mode=self._force_execution_mode,
            )
        elif isinstance(other, list):
            if kwargs.get("axis", 1) == 0:
                raise NotImplementedError(f"{op_name} on rows")
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
                index=self._index_cache,
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
                index=self._index_cache,
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
        HdkOnNativeDataframe
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
            index=self._index_cache,
            index_cols=self._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def cat_codes(self):
        """
        Extract codes for a category column.

        The frame should have a single data column.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
        assert len(self.columns) == 1
        assert is_categorical_dtype(self._dtypes[-1])

        exprs = self._index_exprs()
        col_expr = self.ref(self.columns[-1])
        code_expr = OpExpr("KEY_FOR_STRING", [col_expr], get_dtype("int32"))
        null_val = LiteralExpr(np.int32(-1))
        col_name = MODIN_UNNAMED_SERIES_LABEL
        exprs[col_name] = build_if_then_else(
            col_expr.is_null(), null_val, code_expr, get_dtype("int32")
        )
        dtypes = [expr._dtype for expr in exprs.values()]

        return self.__constructor__(
            columns=Index([col_name]),
            dtypes=pd.Series(dtypes, index=Index(exprs.keys())),
            op=TransformNode(self, exprs),
            index=self._index_cache,
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
        HdkOnNativeDataframe
            The new frame.
        """
        if na_position != "first" and na_position != "last":
            raise ValueError(f"Unsupported na_position value '{na_position}'")

        base = self

        # If index is preserved and we have no index columns then we
        # need to create one using __rowid__ virtual column.
        if not ignore_index and base._index_cols is None:
            base = base._materialize_rowid()

        if not isinstance(columns, list):
            columns = [columns]
        columns = [base._find_index_or_col(col) for col in columns]

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
            if base._index_cols is not None:
                drop_index_cols_before = [
                    col for col in base._index_cols if col not in columns
                ]
                drop_index_cols_after = [
                    col for col in base._index_cols if col in columns
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
                    base = base.__constructor__(
                        columns=base.columns,
                        dtypes=base._dtypes_for_exprs(exprs),
                        op=TransformNode(base, exprs),
                        index_cols=index_cols,
                        force_execution_mode=base._force_execution_mode,
                    )

                base = base.__constructor__(
                    columns=base.columns,
                    dtypes=base.copy_dtypes_cache(),
                    op=SortNode(base, columns, ascending, na_position),
                    index_cols=base._index_cols,
                    force_execution_mode=base._force_execution_mode,
                )

                if drop_index_cols_after:
                    exprs = OrderedDict()
                    for col in base.columns:
                        exprs[col] = base.ref(col)
                    base = base.__constructor__(
                        columns=base.columns,
                        dtypes=base._dtypes_for_exprs(exprs),
                        op=TransformNode(base, exprs),
                        index_cols=None,
                        force_execution_mode=base._force_execution_mode,
                    )

                return base
            else:
                return base.__constructor__(
                    columns=base.columns,
                    dtypes=base.copy_dtypes_cache(),
                    op=SortNode(base, columns, ascending, na_position),
                    index_cols=None,
                    force_execution_mode=base._force_execution_mode,
                )
        else:
            return base.__constructor__(
                columns=base.columns,
                dtypes=base.copy_dtypes_cache(),
                op=SortNode(base, columns, ascending, na_position),
                index_cols=base._index_cols,
                force_execution_mode=base._force_execution_mode,
            )

    def filter(self, key):
        """
        Filter rows by a boolean key column.

        Parameters
        ----------
        key : HdkOnNativeDataframe
            A frame with a single bool data column used as a filter.

        Returns
        -------
        HdkOnNativeDataframe
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
            dtypes=filter_base.copy_dtypes_cache(),
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
                idx_name = mangle_index_names([None])[0]
                exprs[idx_name] = filtered_base.ref(idx_name)
                exprs.move_to_end(idx_name, last=False)

        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(filtered_base, exprs),
            index_cols=filtered_base._index_cols,
            force_execution_mode=self._force_execution_mode,
        )

    def force_import(self) -> DbTable:
        """
        Force table import.

        Returns
        -------
        DbTable
            The imported table.
        """
        if self._has_unsupported_data:
            raise NotImplementedError("Unable to import a frame with unsupported data")
        self._execute()
        return self._partition_mgr_cls.import_table(self)

    def _maybe_materialize_rowid(self):
        """
        Materialize virtual 'rowid' column if frame uses it as an index.

        Returns
        -------
        HdkOnNativeDataframe
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
        HdkOnNativeDataframe
            The new frame.
        """
        name = self._index_cache.get().name if self.has_index_cache else None
        name = mangle_index_names([name])[0]
        exprs = OrderedDict()
        exprs[name] = self.ref(ROWID_COL_NAME)
        for col in self._table_cols:
            exprs[col] = self.ref(col)
        return self.__constructor__(
            columns=self.columns,
            dtypes=self._dtypes_for_exprs(exprs),
            op=TransformNode(self, exprs),
            index_cols=[name],
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
        rhs : HdkOnNativeDataframe
            The second frame.

        Returns
        -------
        HdkOnNativeDataframe
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

        Returns
        -------
        DbTable or pyarrow.Table or pandas.Dataframe
        """
        if isinstance(self._op, FrameNode):
            return self._op.execute_arrow()

        result = None
        stack = [self._materialize, self]
        while stack:
            frame = stack.pop()
            if callable(frame):
                result = frame()
                continue
            if isinstance(frame._op, FrameNode):
                result = frame._op.execute_arrow()
                continue
            if not frame._op.can_execute_hdk() and stack[-1] != frame._materialize:
                stack.append(frame._materialize)
            if frame._uses_rowid or frame._op.require_executed_base():
                for i in reversed(frame._op.input):
                    if not isinstance(i._op, FrameNode):
                        stack.append(i._materialize)
                        stack.append(i)
            else:
                stack.extend(reversed(frame._op.input))
        return result

    def _materialize(self):
        """
        Materialize this frame.

        Returns
        -------
        DbTable or pyarrow.Table
        """
        mode = self._force_execution_mode
        assert mode != "lazy", "Unexpected execution triggered on lazy frame!"

        if isinstance(self._op, FrameNode):
            return self._op.execute_arrow()

        if (
            mode == "arrow"
            or not self._op.can_execute_hdk()
            or (self._can_execute_arrow() and mode != "hdk")
        ):
            new_table = self._execute_arrow()
            partitions = self._partition_mgr_cls.from_arrow(
                new_table, unsupported_cols=[], encode_col_names=False
            )[0]
        else:
            partitions = self._partition_mgr_cls.run_exec_plan(self._op)

        self._partitions = partitions
        self._op = FrameNode(self)
        return partitions[0][0].get()

    def _can_execute_arrow(self):
        """
        Check for possibility of Arrow execution.

        Check if operation's tree for the frame can be executed using
        Arrow API instead of HDK query.

        Returns
        -------
        bool
        """
        if self._force_execution_mode == "hdk":
            return False

        stack = [self]
        while stack:
            op = stack.pop()._op
            if not op.can_execute_arrow():
                return False
            if input := getattr(op, "input", None):
                stack.extend(input)
        return True

    def _execute_arrow(self):
        """
        Compute the frame data using Arrow API.

        Returns
        -------
        pyarrow.Table
            The resulting table.
        """
        result = None
        stack = [self]

        while stack:
            frame = stack.pop()

            if callable(frame):
                if isinstance(result := frame(result), DbTable):
                    result = result.to_arrow()
            elif input := getattr(frame._op, "input", None):
                if len(input) == 1:
                    stack.append(frame._op.execute_arrow)
                    stack.append(input[0])
                else:

                    def to_arrow(result, op=frame._op, tables=[], frames=iter(input)):
                        """
                        Convert the input list to a list of arrow tables.

                        This function is created for each input list. When the function
                        is created, the frames iterator is saved in the `frames` argument.
                        Then, the function is added to the stack followed by the first
                        frame from the `frames` iterator. When the frame is processed, the
                        arrow table is added to the `tables` list. This procedure is
                        repeated until the iterator is not empty. When all the frames are
                        processed, the arrow tables are passed to `execute_arrow` and the
                        result is returned.
                        """
                        if (f := next(frames, None)) is None:
                            return op.execute_arrow(tables)
                        else:
                            # When this function is called, the `frame` attribute contains
                            # a reference to this function.
                            stack.append(frame if callable(frame) else to_arrow)
                            stack.append(tables.append)
                            stack.append(f)
                            return result

                    to_arrow(result)
            elif isinstance(result := frame._op.execute_arrow(result), DbTable):
                result = result.to_arrow()

        return result

    def _build_index_cache(self):
        """Materialize index and store it in the cache."""
        obj = self._execute()

        if self._index_cols is None:
            self.set_index_cache(Index.__new__(RangeIndex, data=range(len(obj))))
            return
        if isinstance(obj, DbTable):
            # TODO: Get the index columns only
            obj = obj.to_arrow()
        if isinstance(obj, pyarrow.Table):
            # The index columns must be in the beginning of the list
            col_names = obj.column_names[len(self._index_cols) :]
            index_at = obj.drop(col_names)
            index_df = index_at.to_pandas()
            index_df.set_index(self._index_cols, inplace=True)
            idx = index_df.index
            idx.rename(demangle_index_names(self._index_cols), inplace=True)
            if (
                isinstance(idx, (pd.DatetimeIndex, pd.TimedeltaIndex))
                and len(idx) >= 3  # infer_freq() requires at least 3 values
            ):
                idx.freq = pd.infer_freq(idx)
            self.set_index_cache(idx)
        else:
            self.set_index_cache(obj.index)

    def _get_index(self):
        """
        Get the index of the frame in pandas format.

        Materializes the frame if required.

        Returns
        -------
        pandas.Index
        """
        if not self.has_index_cache:
            self._build_index_cache()
        return self._index_cache.get()

    def _set_index(self, new_index):
        """
        Set new index for the frame.

        Parameters
        ----------
        new_index : pandas.Index
            New index.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
        if not isinstance(new_index, (Index, MultiIndex)):
            raise NotImplementedError(
                "HdkOnNativeDataframe._set_index is not yet suported"
            )

        obj = self._execute()
        if isinstance(obj, pd.DataFrame):
            raise NotImplementedError(
                "HdkOnNativeDataframe._set_index is not yet suported"
            )
        else:
            at = obj if isinstance(obj, pyarrow.Table) else obj.to_arrow()
            if self._index_cols:
                at = at.drop(self._index_cols)

            new_index = new_index.copy()
            index_names = mangle_index_names(new_index.names)
            new_index.names = index_names
            index_df = pd.DataFrame(data={}, index=new_index)
            index_df = index_df.reset_index()
            index_at = pyarrow.Table.from_pandas(index_df)

            for i, field in enumerate(at.schema):
                index_at = index_at.append_column(field, at.column(i))

            return self.from_arrow(index_at, index_names, new_index, self.columns)

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
        HdkOnNativeDataframe
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
                name = ColNameCodec.demangle_index_name(c)
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
        HdkOnNativeDataframe
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
        HdkOnNativeDataframe
            The new frame.
        """
        if (
            self.columns.identical(new_columns)
            if isinstance(new_columns, Index)
            else all(self.columns == new_columns)
        ):
            return self
        exprs = self._index_exprs()
        for old, new in zip(self.columns, new_columns):
            expr = self.ref(old)
            if exprs.setdefault(new, expr) is not expr:
                raise NotImplementedError("duplicate column names are not supported")
        return self.__constructor__(
            columns=new_columns,
            dtypes=self._dtypes.tolist(),
            op=TransformNode(self, exprs),
            index=self._index_cache,
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
        return super(HdkOnNativeDataframe, self)._get_columns()

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
                "HDK execution does not support exchange protocol if the frame contains data types "
                + "that are unsupported by HDK."
            )

        from ..interchange.dataframe_protocol.dataframe import HdkProtocolDataframe

        return HdkProtocolDataframe(
            self, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df: ProtocolDataframe) -> "HdkOnNativeDataframe":
        """
        Convert a DataFrame implementing the dataframe exchange protocol to a Core Modin Dataframe.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        df : ProtocolDataframe
            The DataFrame object supporting the dataframe exchange protocol.

        Returns
        -------
        HdkOnNativeDataframe
            A new Core Modin Dataframe object.
        """
        if isinstance(df, cls):
            return df

        if not hasattr(df, "__dataframe__"):
            raise ValueError(
                "`df` does not support DataFrame exchange protocol, i.e. `__dataframe__` method"
            )

        from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
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
            # [] operator will return pandas.Series
            return self._dtypes[len(self._index_cols) :]
        return self._dtypes.get()

    def has_multiindex(self):
        """
        Check for multi-index usage.

        Return True if the frame has a multi-index (index with
        multiple columns) and False otherwise.

        Returns
        -------
        bool
        """
        if self.has_materialized_index:
            return isinstance(self.index, MultiIndex)
        return self._index_cols is not None and len(self._index_cols) > 1

    def get_index_name(self):
        """
        Get the name of the index column.

        Returns None for default index and multi-index.

        Returns
        -------
        str or None
        """
        if self.has_index_cache:
            return self._index_cache.get().name
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
        HdkOnNativeDataframe
            The new frame.
        """
        if self.has_multiindex():
            ErrorMessage.single_warning("Scalar name for MultiIndex is not supported!")
            return self

        if self._index_cols is None and name is None:
            return self

        names = mangle_index_names([name])
        exprs = OrderedDict()
        if self._index_cols is None:
            exprs[names[0]] = self.ref(ROWID_COL_NAME)
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
        if self.has_index_cache:
            return self._index_cache.get().names
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
        HdkOnNativeDataframe
            The new frame.
        """
        if not self.has_multiindex():
            raise ValueError("Can set names for MultiIndex only")

        if len(names) != len(self._index_cols):
            raise ValueError(
                f"Unexpected names count: expected {len(self._index_cols)} got {len(names)}"
            )

        names = mangle_index_names(names)
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
        if self._force_execution_mode == "lazy":
            raise RuntimeError("unexpected to_pandas triggered on lazy frame")

        obj = self._execute()

        if isinstance(obj, DbTable):
            obj = obj.to_arrow()
        if isinstance(obj, pyarrow.Table):
            # If the table is exported from HDK, the string columns are converted
            # to dictionary. On conversion to pandas, these columns will be of type
            # Categorical, that is not correct. To make the valid conversion, these
            # fields are cast to string.
            schema = obj.schema
            cast = {
                idx: arrow_type.name
                for idx, (arrow_type, pandas_type) in enumerate(
                    zip(schema, self._dtypes)
                )
                if is_dictionary(arrow_type.type)
                and not is_categorical_dtype(pandas_type)
            }
            if cast:
                for idx, new_type in cast.items():
                    schema = schema.set(idx, pyarrow.field(new_type, pyarrow.string()))
                obj = obj.cast(schema)
            # concatenate() is called by _partition_mgr_cls.to_pandas
            # to preserve the categorical dtypes
            df = concatenate([arrow_to_pandas(obj)])
        else:
            df = obj.copy()

        # If we make dataframe from Arrow table then we might need to set
        # index columns.
        if len(df.columns) != len(self.columns):
            assert self._index_cols
            if self.has_materialized_index:
                df.drop(columns=self._index_cols, inplace=True)
                df.index = self._index_cache.get().copy()
            else:
                df.set_index(self._index_cols, inplace=True)
                df.index.rename(demangle_index_names(self._index_cols), inplace=True)
            assert len(df.columns) == len(self.columns)
        else:
            assert self._index_cols is None
            assert df.index.name is None or isinstance(
                self._partitions[0][0].get(), pd.DataFrame
            ), f"index name '{df.index.name}' is not None"
            if self.has_materialized_index:
                df.index = self._index_cache.get().copy()

        # Restore original column labels encoded in HDK to meet its
        # restrictions on column names.
        df.columns = self.columns

        return df

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
            if col in self._index_cols:
                return col

            pattern = re.compile(f"{IDX_COL_NAME}\\d+_{encode_col_name(col)}")
            for idx_col in self._index_cols:
                if pattern.match(idx_col):
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
        HdkOnNativeDataframe
            The new frame.
        """
        new_index = df.index
        new_columns = df.columns

        if isinstance(new_columns, MultiIndex):
            # MultiIndex columns are not supported by the HDK backend.
            # We just print this warning here and fall back to pandas.
            index_cols = None
            ErrorMessage.single_warning(
                "MultiIndex columns are not currently supported by the HDK backend."
            )
        # If there is non-trivial index, we put it into columns.
        # If the index is trivial, but there are no columns, we put
        # it into columns either because, otherwise, we don't know
        # the number of rows and, thus, unable to restore the index.
        # That's what we usually have for arrow tables and execution
        # result. Unnamed index is renamed to {IDX_COL_PREF}. Also all
        # columns get encoded to handle names unsupported in HDK.
        elif (
            len(new_index) == 0
            and not isinstance(new_index, MultiIndex)
            and new_index.name is None
        ) or (len(new_columns) != 0 and cls._is_trivial_index(new_index)):
            index_cols = None
        else:
            orig_index_names = new_index.names
            orig_df = df
            index_cols = mangle_index_names(new_index.names)
            df.index.names = index_cols
            df = df.reset_index()
            orig_df.index.names = orig_index_names

        new_dtypes = df.dtypes

        def encoder(n):
            return (
                n
                if n == MODIN_UNNAMED_SERIES_LABEL
                else encode_col_name(n, ignore_reserved=False)
            )

        if index_cols is not None:
            cols = index_cols.copy()
            cols.extend([encoder(n) for n in df.columns[len(index_cols) :]])
            df.columns = cols
        else:
            df = df.rename(columns=encoder)

        (
            new_parts,
            new_lengths,
            new_widths,
            unsupported_cols,
        ) = cls._partition_mgr_cls.from_pandas(
            df, return_dims=True, encode_col_names=False
        )

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
    def from_arrow(
        cls, at, index_cols=None, index=None, columns=None, encode_col_names=True
    ):
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
        columns : Index or array-like, optional
            Column labels to use for resulting frame.
        encode_col_names : bool, default: True
            Encode column names.

        Returns
        -------
        HdkOnNativeDataframe
            The new frame.
        """
        (
            new_frame,
            new_lengths,
            new_widths,
            unsupported_cols,
        ) = cls._partition_mgr_cls.from_arrow(
            at, return_dims=True, encode_col_names=encode_col_names
        )

        if columns is not None:
            new_columns = columns
            new_index = index
        elif index_cols:
            data_cols = [col for col in at.column_names if col not in index_cols]
            new_columns = pd.Index(data=data_cols, dtype="O")
            new_index = index
        else:
            assert index is None
            new_columns = pd.Index(data=at.column_names, dtype="O")
            new_index = None

        dtype_index = [] if index_cols is None else list(index_cols)
        dtype_index.extend(new_columns)
        new_dtypes = []

        for col in at.columns:
            if pyarrow.types.is_dictionary(col.type):
                new_dtypes.append(
                    LazyProxyCategoricalDtype._build_proxy(
                        parent=at,
                        column_name=col._name,
                        materializer=build_categorical_from_at,
                    )
                )
            else:
                new_dtypes.append(cls._arrow_type_to_dtype(col.type))

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
            dtypes=pd.Series(data=new_dtypes, index=dtype_index),
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
        if not (isinstance(index, pd.Index) and index.dtype == np.int64):
            return False
        return (
            index.is_monotonic_increasing
            and index.is_unique
            and index.min() == 0
            and index.max() == len(index) - 1
        )
