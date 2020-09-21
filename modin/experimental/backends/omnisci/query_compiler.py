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

from modin.backends.base.query_compiler import (
    BaseQueryCompiler,
    _set_axis as default_axis_setter,
)
from modin.backends.pandas.query_compiler import PandasQueryCompiler

import pandas

from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_list_like


class DFAlgQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
    with a lazy DataFrame Algebra based backend."""

    lazy_execution = True

    def __init__(self, frame, shape_hint=None):
        assert frame is not None
        self._modin_frame = frame
        self._shape_hint = shape_hint

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    default_to_pandas = PandasQueryCompiler.default_to_pandas

    def copy(self):
        return self.__constructor__(self._modin_frame, self._shape_hint)

    def getitem_column_array(self, key, numeric=False):
        shape_hint = "column" if len(key) == 1 else None
        if numeric:
            new_modin_frame = self._modin_frame.mask(col_numeric_idx=key)
        else:
            new_modin_frame = self._modin_frame.mask(col_indices=key)
        return self.__constructor__(new_modin_frame, shape_hint)

    def getitem_array(self, key):
        if isinstance(key, type(self)):
            try:
                new_modin_frame = self._modin_frame.filter(key._modin_frame)
                return self.__constructor__(new_modin_frame, self._shape_hint)
            except NotImplementedError:
                key = key.to_pandas()

        if is_bool_indexer(key):
            return self.default_to_pandas(lambda df: df[key])

        if any(k not in self.columns for k in key):
            raise KeyError(
                "{} not index".format(
                    str([k for k in key if k not in self.columns]).replace(",", "")
                )
            )
        return self.getitem_column_array(key)

    # Merge

    def merge(self, right, **kwargs):
        on = kwargs.get("on", None)
        left_index = kwargs.get("left_index", False)
        right_index = kwargs.get("right_index", False)
        """Only non-index joins with explicit 'on' are supported"""
        if left_index is False and right_index is False and on is not None:
            how = kwargs.get("how", "inner")
            sort = kwargs.get("sort", False)
            suffixes = kwargs.get("suffixes", None)
            if not isinstance(on, list):
                assert isinstance(on, str), f"unsupported 'on' value {on}"
                on = [on]
            return self.__constructor__(
                self._modin_frame.join(
                    right._modin_frame,
                    how=how,
                    on=on,
                    sort=sort,
                    suffixes=suffixes,
                )
            )
        else:
            return self.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    def view(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.mask(row_numeric_idx=index, col_numeric_idx=columns)
        )

    def groupby_size(
        query_compiler,
        by,
        axis,
        groupby_args,
        map_args,
        **kwargs,
    ):
        """Perform a groupby size.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        new_frame = query_compiler._modin_frame.groupby_agg(
            by,
            axis,
            {query_compiler._modin_frame.columns[0]: "size"},
            groupby_args,
            **kwargs,
        )
        if groupby_args["as_index"]:
            shape_hint = "column"
            new_frame = new_frame._set_columns(["__reduced__"])
        else:
            shape_hint = None
            new_frame = new_frame._set_columns(list(new_frame.columns)[:-1] + ["size"])
        new_qc = query_compiler.__constructor__(new_frame, shape_hint=shape_hint)
        if groupby_args["squeeze"]:
            new_qc = new_qc.squeeze()
        return new_qc

    def groupby_sum(query_compiler, by, axis, groupby_args, map_args, **kwargs):
        """Groupby with sum aggregation.

        Parameters
        ----------
        by
            The column value to group by. This can come in the form of a query compiler
        axis : (0 or 1)
            The axis the group by
        groupby_args : dict of {"str": value}
            The arguments for groupby. These can include 'level', 'sort', 'as_index',
            'group_keys', and 'squeeze'.
        kwargs
            The keyword arguments for the sum operation

        Returns
        -------
        QueryCompiler
            A new QueryCompiler
        """
        new_frame = query_compiler._modin_frame.groupby_agg(
            by, axis, "sum", groupby_args, **kwargs
        )
        new_qc = query_compiler.__constructor__(new_frame)
        if groupby_args["squeeze"]:
            new_qc = new_qc.squeeze()
        return new_qc

    def groupby_count(query_compiler, by, axis, groupby_args, map_args, **kwargs):
        """Perform a groupby count.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        QueryCompiler
        """
        new_frame = query_compiler._modin_frame.groupby_agg(
            by, axis, "count", groupby_args, **kwargs
        )
        new_qc = query_compiler.__constructor__(new_frame)
        if groupby_args["squeeze"]:
            new_qc = new_qc.squeeze()
        return new_qc

    def groupby_dict_agg(self, by, func_dict, groupby_args, agg_args, drop=False):
        """Apply aggregation functions to a grouped dataframe per-column.

        Parameters
        ----------
        by : DFAlgQueryCompiler
            The column to group by
        func_dict : dict of str, callable/string
            The dictionary mapping of column to function
        groupby_args : dict
            The dictionary of keyword arguments for the group by.
        agg_args : dict
            The dictionary of keyword arguments for the aggregation functions
        drop : bool
            Whether or not to drop the column from the data.

        Returns
        -------
        DFAlgQueryCompiler
            The result of the per-column aggregations on the grouped dataframe.
        """
        # TODO: handle drop arg
        new_frame = self._modin_frame.groupby_agg(
            by, 0, func_dict, groupby_args, **agg_args
        )
        new_qc = self.__constructor__(new_frame)
        if groupby_args["squeeze"]:
            new_qc = new_qc.squeeze()
        return new_qc

    def _get_index(self):
        return self._modin_frame.index

    def _set_index(self, index):
        default_axis_setter(0)(self, index)
        # NotImplementedError: OmnisciOnRayFrame._set_index is not yet suported
        # self._modin_frame.index = index

    def _get_columns(self):
        return self._modin_frame.columns

    def _set_columns(self, columns):
        self._modin_frame = self._modin_frame._set_columns(columns)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        assert not inplace, "inplace=True should be handled on upper level"
        new_frame = self._modin_frame.fillna(
            value=value,
            method=method,
            axis=axis,
            limit=limit,
            downcast=downcast,
        )
        return self.__constructor__(new_frame, self._shape_hint)

    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(o, type(self)) for o in other
        ), "Different Manager objects are being used. This is not allowed"
        sort = kwargs.get("sort", None)
        if sort is None:
            sort = False
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        other_modin_frames = [o._modin_frame for o in other]

        new_modin_frame = self._modin_frame._concat(
            axis, other_modin_frames, join=join, sort=sort, ignore_index=ignore_index
        )
        return self.__constructor__(new_modin_frame)

    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
        assert index is None, "Only column drop is supported"
        return self.__constructor__(
            self._modin_frame.mask(
                row_indices=index, col_indices=self.columns.drop(columns)
            )
        )

    def dt_year(self):
        """Extract year from Datetime info

        Returns:
            A new QueryCompiler.
        """
        return self.__constructor__(
            self._modin_frame.dt_extract("year"), self._shape_hint
        )

    def dt_month(self):
        """Extract month from Datetime info

        Returns:
            A new QueryCompiler.
        """
        return self.__constructor__(
            self._modin_frame.dt_extract("month"), self._shape_hint
        )

    def _bin_op(self, other, op_name, **kwargs):
        level = kwargs.get("level", None)
        if level is not None:
            return getattr(super(), op_name)(other=other, op_name=op_name, **kwargs)

        if isinstance(other, DFAlgQueryCompiler):
            shape_hint = (
                self._shape_hint if self._shape_hint == other._shape_hint else None
            )
            other = other._modin_frame
        else:
            shape_hint = self._shape_hint

        new_modin_frame = self._modin_frame.bin_op(other, op_name, **kwargs)
        return self.__constructor__(new_modin_frame, shape_hint)

    def add(self, other, **kwargs):
        return self._bin_op(other, "add", **kwargs)

    def sub(self, other, **kwargs):
        return self._bin_op(other, "sub", **kwargs)

    def mul(self, other, **kwargs):
        return self._bin_op(other, "mul", **kwargs)

    def floordiv(self, other, **kwargs):
        return self._bin_op(other, "floordiv", **kwargs)

    def truediv(self, other, **kwargs):
        return self._bin_op(other, "truediv", **kwargs)

    def eq(self, other, **kwargs):
        return self._bin_op(other, "eq", **kwargs)

    def ge(self, other, **kwargs):
        return self._bin_op(other, "ge", **kwargs)

    def gt(self, other, **kwargs):
        return self._bin_op(other, "gt", **kwargs)

    def le(self, other, **kwargs):
        return self._bin_op(other, "le", **kwargs)

    def lt(self, other, **kwargs):
        return self._bin_op(other, "lt", **kwargs)

    def ne(self, other, **kwargs):
        return self._bin_op(other, "ne", **kwargs)

    def reset_index(self, **kwargs):
        level = kwargs.get("level", None)
        if level is not None:
            return super().reset_index(**kwargs)

        drop = kwargs.get("drop", False)
        shape_hint = self._shape_hint if drop else None

        return self.__constructor__(
            self._modin_frame.reset_index(drop), shape_hint=shape_hint
        )

    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return self.__constructor__(
            self._modin_frame.astype(col_dtypes), self._shape_hint
        )

    def setitem(self, axis, key, value):
        """Set the column defined by `key` to the `value` provided.

        Args:
            key: The column name to set.
            value: The value to set the column to.

        Returns:
             A new QueryCompiler
        """
        if axis == 1 or not isinstance(value, type(self)):
            return super().setitem(axis=axis, key=key, value=value)

        return self._setitem(axis, key, value)

    _setitem = PandasQueryCompiler.setitem

    def insert(self, loc, column, value):
        """Insert new column data.

        Args:
            loc: Insertion index.
            column: Column labels to insert.
            value: Dtype object values to insert.

        Returns:
            A new DFAlgQueryCompiler with new data inserted.
        """
        if is_list_like(value):
            return super().insert(loc=loc, column=column, value=value)

        return self.__constructor__(self._modin_frame.insert(loc, column, value))

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        """Reorder the rows based on the lexicographic order of the given columns.

        Parameters
        ----------
        columns : scalar or list of scalar
            The column or columns to sort by
        ascending : bool
            Sort in ascending order (True) or descending order (False)

        Returns
        -------
        DFAlgQueryCompiler
            A new query compiler that contains result of the sort
        """
        ignore_index = kwargs.get("ignore_index", False)
        na_position = kwargs.get("na_position", "last")
        return self.__constructor__(
            self._modin_frame.sort_rows(columns, ascending, ignore_index, na_position),
            self._shape_hint,
        )

    def columnarize(self):
        """
        Transposes this QueryCompiler if it has a single row but multiple columns.

        This method should be called for QueryCompilers representing a Series object,
        i.e. self.is_series_like() should be True.

        Returns
        -------
        BaseQueryCompiler
            Transposed new QueryCompiler or self.
        """
        if self._shape_hint == "column":
            assert len(self.columns) == 1, "wrong shape hint"
            return self

        if self._shape_hint == "row":
            # It is OK to trigger execution here because we cannot
            # transpose in OmniSci anyway.
            assert len(self.index) == 1, "wrong shape hint"
            return self.transpose()

        if len(self.columns) != 1 or (
            len(self.index) == 1 and self.index[0] == "__reduced__"
        ):
            res = self.transpose()
            res._shape_hint = "column"
            return res

        self._shape_hint = "column"
        return self

    def is_series_like(self):
        """Return True if QueryCompiler has a single column or row"""
        if self._shape_hint is not None:
            return True
        return len(self.columns) == 1 or len(self.index) == 1

    def cat_codes(self):
        return self.__constructor__(self._modin_frame.cat_codes(), self._shape_hint)

    def has_multiindex(self, axis=0):
        """
        Check if specified axis is indexed by MultiIndex.

        Parameters
        ----------
        axis : 0 or 1, default 0
            The axis to check (0 - index, 1 - columns).

        Returns
        -------
        bool
            True if index at specified axis is MultiIndex and False otherwise.
        """
        if axis == 0:
            return self._modin_frame.has_multiindex()
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    def free(self):
        return

    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    @property
    def dtypes(self):
        return self._modin_frame.dtypes
