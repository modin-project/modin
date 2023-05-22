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

"""
Module contains ``PandasQueryCompiler`` class.

``PandasQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for the ``PandasDataframe``.
"""

import re
import numpy as np
import pandas
import functools
from pandas.api.types import is_scalar
from pandas.core.common import is_bool_indexer
from pandas.core.indexing import check_bool_indexer
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)
from pandas.core.dtypes.cast import find_common_type
from pandas.errors import DataError, MergeError
from pandas._libs.lib import no_default
from collections.abc import Iterable
from typing import List, Hashable
import warnings
import hashlib
from pandas.core.groupby.base import transformation_kernels

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.config import Engine, ExperimentalGroupbyImpl
from modin.error_message import ErrorMessage
from modin.utils import (
    try_cast_to_pandas,
    wrap_udf_function,
    hashable,
    _inherit_docstrings,
    MODIN_UNNAMED_SERIES_LABEL,
)
from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.algebra import (
    Fold,
    Map,
    TreeReduce,
    Reduce,
    Binary,
    GroupByReduce,
)
from modin.core.dataframe.algebra.default2pandas.groupby import (
    GroupBy,
    GroupByDefault,
    SeriesGroupByDefault,
)
from .utils import get_group_names, merge_partitioning
from .groupby import GroupbyReduceImpl


def _get_axis(axis):
    """
    Build index labels getter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to get labels from. 0 is for index and 1 is for column.

    Returns
    -------
    callable(PandasQueryCompiler) -> pandas.Index
    """
    if axis == 0:
        return lambda self: self._modin_frame.index
    else:
        return lambda self: self._modin_frame.columns


def _set_axis(axis):
    """
    Build index labels setter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to set labels on. 0 is for index and 1 is for column.

    Returns
    -------
    callable(PandasQueryCompiler)
    """
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


def _str_map(func_name):
    """
    Build function that calls specified string function on frames ``str`` accessor.

    Parameters
    ----------
    func_name : str
        String function name to execute on ``str`` accessor.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
    """

    def str_op_builder(df, *args, **kwargs):
        """Apply specified function against `str` accessor of the passed frame."""
        str_s = df.squeeze(axis=1).str
        res = getattr(pandas.Series.str, func_name)(str_s, *args, **kwargs)
        if hasattr(res, "to_frame"):
            res = res.to_frame()
        return res

    return str_op_builder


def _dt_prop_map(property_name):
    """
    Build function that access specified property of the ``dt`` property of the passed frame.

    Parameters
    ----------
    property_name : str
        Date-time property name to access.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied in the partitions.

    Notes
    -----
    This applies non-callable properties of ``Series.dt``.
    """

    def dt_op_builder(df, *args, **kwargs):
        """Access specified date-time property of the passed frame."""
        squeezed_df = df.squeeze(axis=1)
        if isinstance(squeezed_df, pandas.DataFrame) and len(squeezed_df.columns) == 0:
            return squeezed_df
        assert isinstance(squeezed_df, pandas.Series)
        prop_val = getattr(squeezed_df.dt, property_name)
        if isinstance(prop_val, pandas.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, pandas.DataFrame):
            return prop_val
        else:
            return pandas.DataFrame([prop_val])

    return dt_op_builder


def _dt_func_map(func_name):
    """
    Build function that apply specified method against ``dt`` property of the passed frame.

    Parameters
    ----------
    func_name : str
        Date-time function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied in the partitions.

    Notes
    -----
    This applies callable methods of ``Series.dt``.
    """

    def dt_op_builder(df, *args, **kwargs):
        """Apply specified function against ``dt`` accessor of the passed frame."""
        dt_s = df.squeeze(axis=1).dt
        dt_func_result = getattr(pandas.Series.dt, func_name)(dt_s, *args, **kwargs)
        # If we don't specify the dtype for the frame, the frame might get the
        # wrong dtype, e.g. for to_pydatetime in https://github.com/modin-project/modin/issues/4436
        return pandas.DataFrame(dt_func_result, dtype=dt_func_result.dtype)

    return dt_op_builder


def copy_df_for_func(func, display_name: str = None):
    """
    Build function that execute specified `func` against passed frame inplace.

    Built function copies passed frame, applies `func` to the copy and returns
    the modified frame.

    Parameters
    ----------
    func : callable(pandas.DataFrame)
        The function, usually updates a dataframe inplace.
    display_name : str, optional
        The function's name, which is displayed by progress bar.

    Returns
    -------
    callable(pandas.DataFrame)
        A callable function to be applied in the partitions.
    """

    def caller(df, *args, **kwargs):
        """Apply specified function the passed frame inplace."""
        df = df.copy()
        func(df, *args, **kwargs)
        return df

    if display_name is not None:
        caller.__name__ = display_name
    return caller


@_inherit_docstrings(BaseQueryCompiler)
class PandasQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for the pandas storage format.

    This class translates common query compiler API into the DataFrame Algebra
    queries, that is supposed to be executed by :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`.

    Parameters
    ----------
    modin_frame : PandasDataframe
        Modin Frame to query with the compiled queries.
    shape_hint : {"row", "column", None}, default: None
        Shape hint for frames known to be a column or a row, otherwise None.
    """

    def __init__(self, modin_frame, shape_hint=None):
        self._modin_frame = modin_frame
        self._shape_hint = shape_hint

    @property
    def lazy_execution(self):
        """
        Whether underlying Modin frame should be executed in a lazy mode.

        If True, such QueryCompiler will be handled differently at the front-end in order
        to reduce triggering the computation as much as possible.

        Returns
        -------
        bool
        """
        frame = self._modin_frame
        return not frame.has_materialized_index or not frame.has_materialized_columns

    def finalize(self):
        self._modin_frame.finalize()

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    # Dataframe exchange protocol

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        return self._modin_frame.__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df, data_cls):
        return cls(data_cls.from_dataframe(df))

    # END Dataframe exchange protocol

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._modin_frame.dtypes

    # END Index, columns, and dtypes objects

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._modin_frame.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._modin_frame.add_suffix(suffix, axis))

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(self._modin_frame.copy())

    # END Copy

    # Append/Concat/Join (Not Merge)
    # The append/concat/join operations should ideally never trigger remote
    # compute. These operations should only ever be manipulations of the
    # metadata of the resulting object. It should just be a simple matter of
    # appending the other object's blocks and adding np.nan columns for the new
    # columns, if needed. If new columns are added, some compute may be
    # required, though it can be delayed.
    #
    # Currently this computation is not delayed, and it may make a copy of the
    # DataFrame in memory. This can be problematic and should be fixed in the
    # future. TODO (devin-petersohn): Delay reindexing

    def concat(self, axis, other, **kwargs):
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
        other_modin_frame = [o._modin_frame for o in other]
        new_modin_frame = self._modin_frame.concat(axis, other_modin_frame, join, sort)
        result = self.__constructor__(new_modin_frame)
        if ignore_index:
            if axis == 0:
                return result.reset_index(drop=True)
            else:
                result.columns = pandas.RangeIndex(len(result.columns))
                return result
        return result

    # END Append/Concat/Join

    # Data Management Methods
    def free(self):
        # TODO create a way to clean up this object.
        return

    # END Data Management Methods

    # To NumPy
    def to_numpy(self, **kwargs):
        arr = self._modin_frame.to_numpy(**kwargs)
        ErrorMessage.catch_bugs_and_request_email(
            len(arr) != len(self.index) or len(arr[0]) != len(self.columns)
        )
        return arr

    # END To NumPy

    # Binary operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    add = Binary.register(pandas.DataFrame.add, infer_dtypes="common_cast")
    combine = Binary.register(pandas.DataFrame.combine, infer_dtypes="common_cast")
    combine_first = Binary.register(pandas.DataFrame.combine_first, infer_dtypes="bool")
    eq = Binary.register(pandas.DataFrame.eq, infer_dtypes="bool")
    floordiv = Binary.register(pandas.DataFrame.floordiv, infer_dtypes="common_cast")
    ge = Binary.register(pandas.DataFrame.ge, infer_dtypes="bool")
    gt = Binary.register(pandas.DataFrame.gt, infer_dtypes="bool")
    le = Binary.register(pandas.DataFrame.le, infer_dtypes="bool")
    lt = Binary.register(pandas.DataFrame.lt, infer_dtypes="bool")
    mod = Binary.register(pandas.DataFrame.mod, infer_dtypes="common_cast")
    mul = Binary.register(pandas.DataFrame.mul, infer_dtypes="common_cast")
    rmul = Binary.register(pandas.DataFrame.rmul, infer_dtypes="common_cast")
    ne = Binary.register(pandas.DataFrame.ne, infer_dtypes="bool")
    pow = Binary.register(pandas.DataFrame.pow, infer_dtypes="common_cast")
    radd = Binary.register(pandas.DataFrame.radd, infer_dtypes="common_cast")
    rfloordiv = Binary.register(pandas.DataFrame.rfloordiv, infer_dtypes="common_cast")
    rmod = Binary.register(pandas.DataFrame.rmod, infer_dtypes="common_cast")
    rpow = Binary.register(pandas.DataFrame.rpow, infer_dtypes="common_cast")
    rsub = Binary.register(pandas.DataFrame.rsub, infer_dtypes="common_cast")
    rtruediv = Binary.register(pandas.DataFrame.rtruediv, infer_dtypes="float")
    sub = Binary.register(pandas.DataFrame.sub, infer_dtypes="common_cast")
    truediv = Binary.register(pandas.DataFrame.truediv, infer_dtypes="float")
    __and__ = Binary.register(pandas.DataFrame.__and__, infer_dtypes="bool")
    __or__ = Binary.register(pandas.DataFrame.__or__, infer_dtypes="bool")
    __rand__ = Binary.register(pandas.DataFrame.__rand__, infer_dtypes="bool")
    __ror__ = Binary.register(pandas.DataFrame.__ror__, infer_dtypes="bool")
    __rxor__ = Binary.register(pandas.DataFrame.__rxor__, infer_dtypes="bool")
    __xor__ = Binary.register(pandas.DataFrame.__xor__, infer_dtypes="bool")
    df_update = Binary.register(
        copy_df_for_func(pandas.DataFrame.update, display_name="update"),
        join_type="left",
    )
    series_update = Binary.register(
        copy_df_for_func(
            lambda x, y: pandas.Series.update(x.squeeze(axis=1), y.squeeze(axis=1)),
            display_name="update",
        ),
        join_type="left",
    )

    # Needed for numpy API
    _logical_and = Binary.register(
        lambda df, other, *args, **kwargs: pandas.DataFrame(
            np.logical_and(df, other, *args, **kwargs)
        ),
        infer_dtypes="bool",
    )
    _logical_or = Binary.register(
        lambda df, other, *args, **kwargs: pandas.DataFrame(
            np.logical_or(df, other, *args, **kwargs)
        ),
        infer_dtypes="bool",
    )
    _logical_xor = Binary.register(
        lambda df, other, *args, **kwargs: pandas.DataFrame(
            np.logical_xor(df, other, *args, **kwargs)
        ),
        infer_dtypes="bool",
    )

    def where(self, cond, other, **kwargs):
        assert isinstance(
            cond, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"
        # it's doesn't work if `other` is Series._query_compiler because
        # `n_ary_op` performs columns copartition both for `cond` and `other`.
        if isinstance(other, type(self)) and other._shape_hint is not None:
            other = other.to_pandas()
        if isinstance(other, type(self)):
            # Make sure to set join_type=None so the `where` result always has
            # the same row and column labels as `self`.
            new_modin_frame = self._modin_frame.n_ary_op(
                lambda df, cond, other: df.where(cond, other, **kwargs),
                [
                    cond._modin_frame,
                    other._modin_frame,
                ],
                join_type=None,
            )
        # This will be a Series of scalars to be applied based on the condition
        # dataframe.
        else:

            def where_builder_series(df, cond):
                return df.where(cond, other, **kwargs)

            new_modin_frame = self._modin_frame.n_ary_op(
                where_builder_series, [cond._modin_frame], join_type="left"
            )
        return self.__constructor__(new_modin_frame)

    def merge(self, right, **kwargs):
        how = kwargs.get("how", "inner")
        on = kwargs.get("on", None)
        left_on = kwargs.get("left_on", None)
        right_on = kwargs.get("right_on", None)
        left_index = kwargs.get("left_index", False)
        right_index = kwargs.get("right_index", False)
        sort = kwargs.get("sort", False)

        if how in ["left", "inner"] and left_index is False and right_index is False:
            right_pandas = right.to_pandas()

            kwargs["sort"] = False

            def map_func(left, right=right_pandas, kwargs=kwargs):  # pragma: no cover
                return pandas.merge(left, right_pandas, **kwargs)

            # Want to ensure that these are python lists
            if left_on is not None and right_on is not None:
                left_on = list(left_on) if is_list_like(left_on) else [left_on]
                right_on = list(right_on) if is_list_like(right_on) else [right_on]
            elif on is not None:
                on = list(on) if is_list_like(on) else [on]

            new_columns = None
            new_dtypes = None
            if self._modin_frame.has_materialized_columns:
                if left_on is None and right_on is None:
                    if on is None:
                        on = [c for c in self.columns if c in right_pandas.columns]
                    _left_on, _right_on = on, on
                else:
                    if left_on is None or right_on is None:
                        raise MergeError(
                            "Must either pass only 'on' or 'left_on' and 'right_on', not combination of them."
                        )
                    _left_on, _right_on = left_on, right_on

                try:
                    new_columns, left_renamer, right_renamer = join_columns(
                        self.columns,
                        right_pandas.columns,
                        _left_on,
                        _right_on,
                        kwargs.get("suffixes", ("_x", "_y")),
                    )
                except NotImplementedError:
                    # This happens when one of the keys to join is an index level. Pandas behaviour
                    # is really complicated in this case, so we're not computing resulted columns for now.
                    pass
                else:
                    if self._modin_frame.has_materialized_dtypes:
                        new_dtypes = []
                        for old_col in left_renamer.keys():
                            new_dtypes.append(self.dtypes[old_col])
                        for old_col in right_renamer.keys():
                            new_dtypes.append(right_pandas.dtypes[old_col])
                        new_dtypes = pandas.Series(new_dtypes, index=new_columns)

            new_self = self.__constructor__(
                self._modin_frame.apply_full_axis(
                    axis=1,
                    func=map_func,
                    # We're going to explicitly change the shape across the 1-axis,
                    # so we want for partitioning to adapt as well
                    keep_partitioning=False,
                    num_splits=merge_partitioning(
                        self._modin_frame, right._modin_frame, axis=1
                    ),
                    new_columns=new_columns,
                    dtypes=new_dtypes,
                    sync_labels=False,
                )
            )

            # Here we want to understand whether we're joining on a column or on an index level.
            # It's cool if indexes are already materialized so we can easily check that, if not
            # it's fine too, we can also decide that by columns, which tend to be already
            # materialized quite often compared to the indexes.
            keep_index = False
            if self._modin_frame.has_materialized_index:
                if left_on is not None and right_on is not None:
                    keep_index = any(
                        o in self.index.names
                        and o in right_on
                        and o in right_pandas.index.names
                        for o in left_on
                    )
                elif on is not None:
                    keep_index = any(
                        o in self.index.names and o in right_pandas.index.names
                        for o in on
                    )
            else:
                # Have to trigger columns materialization. Hope they're already available at this point.
                if left_on is not None and right_on is not None:
                    keep_index = any(
                        o not in right_pandas.columns
                        and o in left_on
                        and o not in self.columns
                        for o in right_on
                    )
                elif on is not None:
                    keep_index = any(
                        o not in right_pandas.columns and o not in self.columns
                        for o in on
                    )

            if sort:
                if left_on is not None and right_on is not None:
                    new_self = (
                        new_self.sort_index(axis=0, level=left_on + right_on)
                        if keep_index
                        else new_self.sort_rows_by_column_values(left_on + right_on)
                    )
                elif on is not None:
                    new_self = (
                        new_self.sort_index(axis=0, level=on)
                        if keep_index
                        else new_self.sort_rows_by_column_values(on)
                    )

            return new_self if keep_index else new_self.reset_index(drop=True)
        else:
            return self.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    def join(self, right, **kwargs):
        on = kwargs.get("on", None)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)

        if how in ["left", "inner"]:
            right_pandas = right.to_pandas()

            def map_func(left, right=right_pandas, kwargs=kwargs):  # pragma: no cover
                return pandas.DataFrame.join(left, right, **kwargs)

            new_self = self.__constructor__(
                self._modin_frame.apply_full_axis(
                    axis=1,
                    func=map_func,
                    # We're going to explicitly change the shape across the 1-axis,
                    # so we want for partitioning to adapt as well
                    keep_partitioning=False,
                    num_splits=merge_partitioning(
                        self._modin_frame, right._modin_frame, axis=1
                    ),
                )
            )
            return new_self.sort_rows_by_column_values(on) if sort else new_self
        else:
            return self.default_to_pandas(pandas.DataFrame.join, right, **kwargs)

    # END Inter-Data operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        new_index, _ = (self.index, None) if axis else self.index.reindex(labels)
        new_columns, _ = self.columns.reindex(labels) if axis else (self.columns, None)
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: df.reindex(labels=labels, axis=axis, **kwargs),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def reset_index(self, **kwargs):
        if self.lazy_execution:

            def _reset(df, *axis_lengths, partition_idx):  # pragma: no cover
                df = df.reset_index(**kwargs)

                if isinstance(df.index, pandas.RangeIndex):
                    # If the resulting index is a pure RangeIndex that means that
                    # `.reset_index` actually dropped all of the levels of the
                    # original index and so we have to recompute it manually for each partition
                    start = sum(axis_lengths[:partition_idx])
                    stop = sum(axis_lengths[: partition_idx + 1])

                    df.index = pandas.RangeIndex(start, stop)
                return df

            if self._modin_frame.has_columns_cache and kwargs["drop"]:
                new_columns = self._modin_frame.copy_columns_cache()
            else:
                new_columns = None

            return self.__constructor__(
                self._modin_frame.apply_full_axis(
                    axis=1,
                    func=_reset,
                    enumerate_partitions=True,
                    new_columns=new_columns,
                    dtypes=(
                        self._modin_frame._dtypes if kwargs.get("drop", False) else None
                    ),
                    sync_labels=False,
                    pass_axis_lengths_to_partitions=True,
                )
            )

        allow_duplicates = kwargs.pop("allow_duplicates", no_default)
        names = kwargs.pop("names", None)
        if allow_duplicates not in (no_default, False) or names is not None:
            return self.default_to_pandas(
                pandas.DataFrame.reset_index,
                allow_duplicates=allow_duplicates,
                names=names,
                **kwargs,
            )

        drop = kwargs.get("drop", False)
        level = kwargs.get("level", None)
        new_index = None
        if level is not None:
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [self.index._get_level_number(lev) for lev in level]
            uniq_sorted_level = sorted(set(level))
            if len(uniq_sorted_level) < self.index.nlevels:
                # We handle this by separately computing the index. We could just
                # put the labels into the data and pull them back out, but that is
                # expensive.
                new_index = (
                    self.index.droplevel(uniq_sorted_level)
                    if len(level) < self.index.nlevels
                    else pandas.RangeIndex(len(self.index))
                )
        elif not drop:
            uniq_sorted_level = list(range(self.index.nlevels))

        if not drop:
            if len(uniq_sorted_level) < self.index.nlevels:
                # These are the index levels that will remain after the reset_index
                keep_levels = [
                    i for i in range(self.index.nlevels) if i not in uniq_sorted_level
                ]
                new_copy = self.copy()
                # Change the index to have only the levels that will be inserted
                # into the data. We will replace the old levels later.
                new_copy.index = self.index.droplevel(keep_levels)
                new_copy.index.names = [
                    "level_{}".format(level_value)
                    if new_copy.index.names[level_index] is None
                    else new_copy.index.names[level_index]
                    for level_index, level_value in enumerate(uniq_sorted_level)
                ]
                new_modin_frame = new_copy._modin_frame.from_labels()
                # Replace the levels that will remain as a part of the index.
                new_modin_frame.index = new_index
            else:
                new_modin_frame = self._modin_frame.from_labels()
            if isinstance(new_modin_frame.columns, pandas.MultiIndex):
                # Fix col_level and col_fill in generated column names because from_labels works with assumption
                # that col_level and col_fill are not specified but it expands tuples in level names.
                col_level = kwargs.get("col_level", 0)
                col_fill = kwargs.get("col_fill", "")
                if col_level != 0 or col_fill != "":
                    # Modify generated column names if col_level and col_fil have values different from default.
                    levels_names_list = [
                        f"level_{level_index}" if level_name is None else level_name
                        for level_index, level_name in enumerate(self.index.names)
                    ]
                    if col_fill is None:
                        # Initialize col_fill if it is None.
                        # This is some weird undocumented Pandas behavior to take first
                        # element of the last column name.
                        last_col_name = levels_names_list[uniq_sorted_level[-1]]
                        last_col_name = (
                            list(last_col_name)
                            if isinstance(last_col_name, tuple)
                            else [last_col_name]
                        )
                        if len(last_col_name) not in (1, self.columns.nlevels):
                            raise ValueError(
                                "col_fill=None is incompatible "
                                + f"with incomplete column name {last_col_name}"
                            )
                        col_fill = last_col_name[0]
                    columns_list = new_modin_frame.columns.tolist()
                    for level_index, level_value in enumerate(uniq_sorted_level):
                        level_name = levels_names_list[level_value]
                        # Expand tuples into separate items and fill the rest with col_fill
                        top_level = [col_fill] * col_level
                        middle_level = (
                            list(level_name)
                            if isinstance(level_name, tuple)
                            else [level_name]
                        )
                        bottom_level = [col_fill] * (
                            self.columns.nlevels - (col_level + len(middle_level))
                        )
                        item = tuple(top_level + middle_level + bottom_level)
                        if len(item) > self.columns.nlevels:
                            raise ValueError(
                                "Item must have length equal to number of levels."
                            )
                        columns_list[level_index] = item
                    new_modin_frame.columns = pandas.MultiIndex.from_tuples(
                        columns_list, names=self.columns.names
                    )
            new_self = self.__constructor__(new_modin_frame)
        else:
            new_self = self.copy()
            new_self.index = (
                # Cheaper to compute row lengths than index
                pandas.RangeIndex(sum(new_self._modin_frame.row_lengths))
                if new_index is None
                else new_index
            )
        return new_self

    def set_index_from_columns(
        self, keys: List[Hashable], drop: bool = True, append: bool = False
    ):
        new_modin_frame = self._modin_frame.to_labels(keys)
        if append:
            arrays = []
            # Appending keeps the original order of the index levels, then appends the
            # new index objects.
            names = list(self.index.names)
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)

            # Add the names in the correct order.
            names.extend(new_modin_frame.index.names)
            if isinstance(new_modin_frame.index, pandas.MultiIndex):
                for i in range(new_modin_frame.index.nlevels):
                    arrays.append(new_modin_frame.index._get_level_values(i))
            else:
                arrays.append(new_modin_frame.index)
            new_modin_frame.index = ensure_index_from_sequences(arrays, names)
        if not drop:
            # The algebraic operator for this operation always drops the column, but we
            # can copy the data in this object and just use the index from the result of
            # the query compiler call.
            result = self._modin_frame.copy()
            result.index = new_modin_frame.index
        else:
            result = new_modin_frame
        return self.__constructor__(result)

    # END Reindex/reset_index

    # Transpose
    # For transpose, we aren't going to immediately copy everything. Since the
    # actual transpose operation is very fast, we will just do it before any
    # operation that gets called on the transposed data. See _prepare_method
    # for how the transpose is applied.
    #
    # Our invariants assume that the blocks are transposed, but not the
    # data inside. Sometimes we have to reverse this transposition of blocks
    # for simplicity of implementation.

    def transpose(self, *args, **kwargs):
        # Switch the index and columns and transpose the data within the blocks.
        return self.__constructor__(self._modin_frame.transpose())

    def is_series_like(self):
        return len(self.columns) == 1 or len(self.index) == 1

    # END Transpose

    # TreeReduce operations
    count = TreeReduce.register(pandas.DataFrame.count, pandas.DataFrame.sum)
    sum = TreeReduce.register(pandas.DataFrame.sum)
    prod = TreeReduce.register(pandas.DataFrame.prod)
    any = TreeReduce.register(pandas.DataFrame.any, pandas.DataFrame.any)
    all = TreeReduce.register(pandas.DataFrame.all, pandas.DataFrame.all)
    # memory_usage adds an extra column for index usage, but we don't want to distribute
    # the index memory usage calculation.
    _memory_usage_without_index = TreeReduce.register(
        pandas.DataFrame.memory_usage,
        lambda x, *args, **kwargs: pandas.DataFrame.sum(x),
        axis=0,
    )

    def memory_usage(self, **kwargs):
        index = kwargs.get("index", True)
        deep = kwargs.get("deep", False)
        usage_without_index = self._memory_usage_without_index(index=False, deep=deep)
        return (
            self.from_pandas(
                pandas.DataFrame(
                    [self.index.memory_usage()],
                    columns=["Index"],
                    index=[MODIN_UNNAMED_SERIES_LABEL],
                ),
                data_cls=type(self._modin_frame),
            ).concat(axis=1, other=[usage_without_index])
            if index
            else usage_without_index
        )

    def max(self, axis, **kwargs):
        def map_func(df, **kwargs):
            return pandas.DataFrame.max(df, **kwargs)

        def reduce_func(df, **kwargs):
            if kwargs.get("numeric_only", False):
                kwargs = kwargs.copy()
                kwargs["numeric_only"] = False
            return pandas.DataFrame.max(df, **kwargs)

        return TreeReduce.register(map_func, reduce_func)(self, axis=axis, **kwargs)

    def min(self, axis, **kwargs):
        def map_func(df, **kwargs):
            return pandas.DataFrame.min(df, **kwargs)

        def reduce_func(df, **kwargs):
            if kwargs.get("numeric_only", False):
                kwargs = kwargs.copy()
                kwargs["numeric_only"] = False
            return pandas.DataFrame.min(df, **kwargs)

        return TreeReduce.register(map_func, reduce_func)(self, axis=axis, **kwargs)

    def mean(self, axis, **kwargs):
        if kwargs.get("level") is not None:
            return self.default_to_pandas(pandas.DataFrame.mean, axis=axis, **kwargs)

        skipna = kwargs.get("skipna", True)

        # TODO-FIX: this function may work incorrectly with user-defined "numeric" values.
        # Since `count(numeric_only=True)` discards all unknown "numeric" types, we can get incorrect
        # divisor inside the reduce function.
        def map_fn(df, **kwargs):
            """
            Perform Map phase of the `mean`.

            Compute sum and number of elements in a given partition.
            """
            result = pandas.DataFrame(
                {
                    "sum": df.sum(axis=axis, skipna=skipna),
                    "count": df.count(axis=axis, numeric_only=True),
                }
            )
            return result if axis else result.T

        def reduce_fn(df, **kwargs):
            """
            Perform Reduce phase of the `mean`.

            Compute sum for all the the partitions and divide it to
            the total number of elements.
            """
            sum_cols = df["sum"] if axis else df.loc["sum"]
            count_cols = df["count"] if axis else df.loc["count"]

            if not isinstance(sum_cols, pandas.Series):
                # If we got `NaN` as the result of the sum in any axis partition,
                # then we must consider the whole sum as `NaN`, so setting `skipna=False`
                sum_cols = sum_cols.sum(axis=axis, skipna=False)
                count_cols = count_cols.sum(axis=axis, skipna=False)
            return sum_cols / count_cols

        return TreeReduce.register(
            map_fn,
            reduce_fn,
        )(self, axis=axis, **kwargs)

    # END TreeReduce operations

    # Reduce operations
    idxmax = Reduce.register(pandas.DataFrame.idxmax)
    idxmin = Reduce.register(pandas.DataFrame.idxmin)
    median = Reduce.register(pandas.DataFrame.median)
    nunique = Reduce.register(pandas.DataFrame.nunique)
    skew = Reduce.register(pandas.DataFrame.skew)
    kurt = Reduce.register(pandas.DataFrame.kurt)
    sem = Reduce.register(pandas.DataFrame.sem)
    std = Reduce.register(pandas.DataFrame.std)
    var = Reduce.register(pandas.DataFrame.var)
    sum_min_count = Reduce.register(pandas.DataFrame.sum)
    prod_min_count = Reduce.register(pandas.DataFrame.prod)
    quantile_for_single_value = Reduce.register(pandas.DataFrame.quantile)
    mad = Reduce.register(pandas.DataFrame.mad)

    def to_datetime(self, *args, **kwargs):
        if len(self.columns) == 1:
            return Map.register(
                # to_datetime has inplace side effects, see GH#3063
                lambda df, *args, **kwargs: pandas.to_datetime(
                    df.squeeze(axis=1), *args, **kwargs
                ).to_frame()
            )(self, *args, **kwargs)
        else:
            return Reduce.register(pandas.to_datetime, axis=1)(self, *args, **kwargs)

    # END Reduce operations

    def _resample_func(
        self, resample_kwargs, func_name, new_columns=None, df_op=None, *args, **kwargs
    ):
        """
        Resample underlying time-series data and apply aggregation on it.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters in the format of ``modin.pandas.DataFrame.resample`` signature.
        func_name : str
            Aggregation function name to apply on resampler object.
        new_columns : list of labels, optional
            Actual column labels of the resulted frame, supposed to be a hint for the
            Modin frame. If not specified will be computed automaticly.
        df_op : callable(pandas.DataFrame) -> [pandas.DataFrame, pandas.Series], optional
            Preprocessor function to apply to the passed frame before resampling.
        *args : args
            Arguments to pass to the aggregation function.
        **kwargs : kwargs
            Arguments to pass to the aggregation function.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the result of resample aggregation.
        """

        def map_func(df, resample_kwargs=resample_kwargs):  # pragma: no cover
            """Resample time-series data of the passed frame and apply aggregation function on it."""
            if df_op is not None:
                df = df_op(df)
            resampled_val = df.resample(**resample_kwargs)
            op = getattr(pandas.core.resample.Resampler, func_name)
            if callable(op):
                try:
                    # This will happen with Arrow buffer read-only errors. We don't want to copy
                    # all the time, so this will try to fast-path the code first.
                    val = op(resampled_val, *args, **kwargs)
                except ValueError:
                    resampled_val = df.copy().resample(**resample_kwargs)
                    val = op(resampled_val, *args, **kwargs)
            else:
                val = getattr(resampled_val, func_name)

            if isinstance(val, pandas.Series):
                return val.to_frame()
            else:
                return val

        new_modin_frame = self._modin_frame.apply_full_axis(
            axis=0, func=map_func, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def resample_get_group(self, resample_kwargs, name, obj):
        return self._resample_func(resample_kwargs, "get_group", name=name, obj=obj)

    def resample_app_ser(self, resample_kwargs, func, *args, **kwargs):
        return self._resample_func(
            resample_kwargs,
            "apply",
            df_op=lambda df: df.squeeze(axis=1),
            func=func,
            *args,
            **kwargs,
        )

    def resample_app_df(self, resample_kwargs, func, *args, **kwargs):
        return self._resample_func(resample_kwargs, "apply", func=func, *args, **kwargs)

    def resample_agg_ser(self, resample_kwargs, func, *args, **kwargs):
        return self._resample_func(
            resample_kwargs,
            "aggregate",
            df_op=lambda df: df.squeeze(axis=1),
            func=func,
            *args,
            **kwargs,
        )

    def resample_agg_df(self, resample_kwargs, func, *args, **kwargs):
        return self._resample_func(
            resample_kwargs, "aggregate", func=func, *args, **kwargs
        )

    def resample_transform(self, resample_kwargs, arg, *args, **kwargs):
        return self._resample_func(
            resample_kwargs, "transform", arg=arg, *args, **kwargs
        )

    def resample_pipe(self, resample_kwargs, func, *args, **kwargs):
        return self._resample_func(resample_kwargs, "pipe", func=func, *args, **kwargs)

    def resample_ffill(self, resample_kwargs, limit):
        return self._resample_func(resample_kwargs, "ffill", limit=limit)

    def resample_backfill(self, resample_kwargs, limit):
        return self._resample_func(resample_kwargs, "backfill", limit=limit)

    def resample_bfill(self, resample_kwargs, limit):
        return self._resample_func(resample_kwargs, "bfill", limit=limit)

    def resample_pad(self, resample_kwargs, limit):
        return self._resample_func(resample_kwargs, "pad", limit=limit)

    def resample_nearest(self, resample_kwargs, limit):
        return self._resample_func(resample_kwargs, "nearest", limit=limit)

    def resample_fillna(self, resample_kwargs, method, limit):
        return self._resample_func(
            resample_kwargs, "fillna", method=method, limit=limit
        )

    def resample_asfreq(self, resample_kwargs, fill_value):
        return self._resample_func(resample_kwargs, "asfreq", fill_value=fill_value)

    def resample_interpolate(
        self,
        resample_kwargs,
        method,
        axis,
        limit,
        inplace,
        limit_direction,
        limit_area,
        downcast,
        **kwargs,
    ):
        return self._resample_func(
            resample_kwargs,
            "interpolate",
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    def resample_count(self, resample_kwargs):
        return self._resample_func(resample_kwargs, "count")

    def resample_nunique(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "nunique", *args, **kwargs)

    def resample_first(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "first", *args, **kwargs)

    def resample_last(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "last", *args, **kwargs)

    def resample_max(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "max", *args, **kwargs)

    def resample_mean(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "median", *args, **kwargs)

    def resample_median(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "median", *args, **kwargs)

    def resample_min(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "min", *args, **kwargs)

    def resample_ohlc_ser(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(
            resample_kwargs,
            "ohlc",
            df_op=lambda df: df.squeeze(axis=1),
            *args,
            **kwargs,
        )

    def resample_ohlc_df(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "ohlc", *args, **kwargs)

    def resample_prod(self, resample_kwargs, min_count, *args, **kwargs):
        return self._resample_func(
            resample_kwargs,
            "prod",
            min_count=min_count,
            *args,
            **kwargs,
        )

    def resample_size(self, resample_kwargs):
        return self._resample_func(
            resample_kwargs, "size", new_columns=[MODIN_UNNAMED_SERIES_LABEL]
        )

    def resample_sem(self, resample_kwargs, *args, **kwargs):
        return self._resample_func(resample_kwargs, "sem", *args, **kwargs)

    def resample_std(self, resample_kwargs, ddof, *args, **kwargs):
        return self._resample_func(resample_kwargs, "std", ddof=ddof, *args, **kwargs)

    def resample_sum(self, resample_kwargs, min_count, *args, **kwargs):
        return self._resample_func(
            resample_kwargs,
            "sum",
            min_count=min_count,
            *args,
            **kwargs,
        )

    def resample_var(self, resample_kwargs, ddof, *args, **kwargs):
        return self._resample_func(resample_kwargs, "var", ddof=ddof, *args, **kwargs)

    def resample_quantile(self, resample_kwargs, q, **kwargs):
        return self._resample_func(resample_kwargs, "quantile", q=q, **kwargs)

    def expanding_aggregate(self, axis, expanding_args, func, *args, **kwargs):
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(
                df.expanding(*expanding_args).aggregate(func=func, *args, **kwargs)
            ),
            new_index=self.index,
        )
        return self.__constructor__(new_modin_frame)

    expanding_sum = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).sum(*args, **kwargs)
        )
    )

    expanding_min = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).min(*args, **kwargs)
        )
    )

    expanding_max = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).max(*args, **kwargs)
        )
    )

    expanding_mean = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).mean(*args, **kwargs)
        )
    )

    expanding_median = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).median(*args, **kwargs)
        )
    )

    expanding_var = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).var(*args, **kwargs)
        )
    )

    expanding_std = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).std(*args, **kwargs)
        )
    )

    expanding_count = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).count(*args, **kwargs)
        )
    )

    def expanding_cov(
        self,
        fold_axis,
        expanding_args,
        squeeze_self,
        squeeze_other,
        other=None,
        pairwise=None,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        other_for_pandas = (
            other
            if other is None
            else other.to_pandas().squeeze(axis=1)
            if squeeze_other
            else other.to_pandas()
        )
        if len(self.columns) > 1:
            # computing covariance for each column requires having the other columns,
            # so we can't parallelize this as a full-column operation
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.expanding(df, *expanding_args).cov(
                    other=other_for_pandas,
                    pairwise=pairwise,
                    ddof=ddof,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return Fold.register(
            lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
                (df.squeeze(axis=1) if squeeze_self else df)
                .expanding(*expanding_args)
                .cov(*args, **kwargs)
            )
        )(
            self,
            fold_axis,
            expanding_args,
            other=other_for_pandas,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def expanding_corr(
        self,
        fold_axis,
        expanding_args,
        squeeze_self,
        squeeze_other,
        other=None,
        pairwise=None,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        other_for_pandas = (
            other
            if other is None
            else other.to_pandas().squeeze(axis=1)
            if squeeze_other
            else other.to_pandas()
        )
        if len(self.columns) > 1:
            # computing correlation for each column requires having the other columns,
            # so we can't parallelize this as a full-column operation
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.expanding(df, *expanding_args).corr(
                    other=other_for_pandas,
                    pairwise=pairwise,
                    ddof=ddof,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return Fold.register(
            lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
                (df.squeeze(axis=1) if squeeze_self else df)
                .expanding(*expanding_args)
                .corr(*args, **kwargs)
            )
        )(
            self,
            fold_axis,
            expanding_args,
            other=other_for_pandas,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    expanding_quantile = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).quantile(*args, **kwargs)
        )
    )

    expanding_sem = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).sem(*args, **kwargs)
        )
    )

    expanding_kurt = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).kurt(*args, **kwargs)
        )
    )

    expanding_skew = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).skew(*args, **kwargs)
        )
    )

    expanding_rank = Fold.register(
        lambda df, expanding_args, *args, **kwargs: pandas.DataFrame(
            df.expanding(*expanding_args).rank(*args, **kwargs)
        )
    )

    window_mean = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).mean(*args, **kwargs)
        )
    )
    window_sum = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).sum(*args, **kwargs)
        )
    )
    window_var = Fold.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).var(ddof=ddof, *args, **kwargs)
        )
    )
    window_std = Fold.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).std(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_count = Fold.register(
        lambda df, rolling_args: pandas.DataFrame(df.rolling(*rolling_args).count())
    )
    rolling_sum = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).sum(*args, **kwargs)
        )
    )
    rolling_sem = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).sem(*args, **kwargs)
        )
    )
    rolling_mean = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).mean(*args, **kwargs)
        )
    )
    rolling_median = Fold.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).median(**kwargs)
        )
    )
    rolling_var = Fold.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).var(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_std = Fold.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).std(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_min = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).min(*args, **kwargs)
        )
    )
    rolling_max = Fold.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).max(*args, **kwargs)
        )
    )
    rolling_skew = Fold.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).skew(**kwargs)
        )
    )
    rolling_kurt = Fold.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).kurt(**kwargs)
        )
    )
    rolling_apply = Fold.register(
        lambda df, rolling_args, func, raw, engine, engine_kwargs, args, kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).apply(
                func=func,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs,
                args=args,
                kwargs=kwargs,
            )
        )
    )
    rolling_quantile = Fold.register(
        lambda df, rolling_args, quantile, interpolation, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).quantile(
                quantile=quantile, interpolation=interpolation, **kwargs
            )
        )
    )
    rolling_rank = Fold.register(
        lambda df, rolling_args, method, ascending, pct, numeric_only, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).rank(
                method=method,
                ascending=ascending,
                pct=pct,
                numeric_only=numeric_only,
                **kwargs,
            )
        )
    )

    def rolling_corr(self, axis, rolling_args, other, pairwise, *args, **kwargs):
        if len(self.columns) > 1:
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.rolling(df, *rolling_args).corr(
                    other=other, pairwise=pairwise, *args, **kwargs
                )
            )
        else:
            return Fold.register(
                lambda df: pandas.DataFrame(
                    df.rolling(*rolling_args).corr(
                        other=other, pairwise=pairwise, *args, **kwargs
                    )
                )
            )(self, axis)

    def rolling_cov(self, axis, rolling_args, other, pairwise, ddof, **kwargs):
        if len(self.columns) > 1:
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.rolling(df, *rolling_args).cov(
                    other=other, pairwise=pairwise, ddof=ddof, **kwargs
                )
            )
        else:
            return Fold.register(
                lambda df: pandas.DataFrame(
                    df.rolling(*rolling_args).cov(
                        other=other, pairwise=pairwise, ddof=ddof, **kwargs
                    )
                )
            )(self, axis)

    def rolling_aggregate(self, axis, rolling_args, func, *args, **kwargs):
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(
                df.rolling(*rolling_args).aggregate(func=func, *args, **kwargs)
            ),
            new_index=self.index,
        )
        return self.__constructor__(new_modin_frame)

    def unstack(self, level, fill_value):
        if not isinstance(self.index, pandas.MultiIndex) or (
            isinstance(self.index, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.index.nlevels
        ):
            axis = 1
            new_columns = [MODIN_UNNAMED_SERIES_LABEL]
            need_reindex = True
        else:
            axis = 0
            new_columns = None
            need_reindex = False

        def map_func(df):  # pragma: no cover
            return pandas.DataFrame(df.unstack(level=level, fill_value=fill_value))

        def is_tree_like_or_1d(calc_index, valid_index):
            """
            Check whether specified index is a single dimensional or built in a tree manner.

            Parameters
            ----------
            calc_index : pandas.Index
                Frame index to check.
            valid_index : pandas.Index
                Frame index on the opposite from `calc_index` axis.

            Returns
            -------
            bool
                True if `calc_index` is not MultiIndex or MultiIndex and built in a tree manner.
                False otherwise.
            """
            if not isinstance(calc_index, pandas.MultiIndex):
                return True
            actual_len = 1
            for lvl in calc_index.levels:
                actual_len *= len(lvl)
            return len(self.index) * len(self.columns) == actual_len * len(valid_index)

        is_tree_like_or_1d_index = is_tree_like_or_1d(self.index, self.columns)
        is_tree_like_or_1d_cols = is_tree_like_or_1d(self.columns, self.index)

        is_all_multi_list = False
        if (
            isinstance(self.index, pandas.MultiIndex)
            and isinstance(self.columns, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.index.nlevels
            and is_tree_like_or_1d_index
            and is_tree_like_or_1d_cols
        ):
            is_all_multi_list = True
            real_cols_bkp = self.columns
            obj = self.copy()
            obj.columns = np.arange(len(obj.columns))
        else:
            obj = self

        new_modin_frame = obj._modin_frame.apply_full_axis(
            axis, map_func, new_columns=new_columns
        )
        result = self.__constructor__(new_modin_frame)

        def compute_index(index, columns, consider_index=True, consider_columns=True):
            """
            Compute new index for the unstacked frame.

            Parameters
            ----------
            index : pandas.Index
                Index of the original frame.
            columns : pandas.Index
                Columns of the original frame.
            consider_index : bool, default: True
                Whether original index contains duplicated values.
                If True all duplicates will be droped.
            consider_columns : bool, default: True
                Whether original columns contains duplicated values.
                If True all duplicates will be droped.

            Returns
            -------
            pandas.Index
                New index to use in the unstacked frame.
            """

            def get_unique_level_values(index):
                return [
                    index.get_level_values(lvl).unique()
                    for lvl in np.arange(index.nlevels)
                ]

            new_index = (
                get_unique_level_values(index)
                if consider_index
                else index
                if isinstance(index, list)
                else [index]
            )

            new_columns = (
                get_unique_level_values(columns) if consider_columns else [columns]
            )
            return pandas.MultiIndex.from_product([*new_columns, *new_index])

        if is_all_multi_list and is_tree_like_or_1d_index and is_tree_like_or_1d_cols:
            result = result.sort_index()
            index_level_values = [lvl for lvl in obj.index.levels]

            result.index = compute_index(
                index_level_values, real_cols_bkp, consider_index=False
            )
            return result

        if need_reindex:
            if is_tree_like_or_1d_index and is_tree_like_or_1d_cols:
                is_recompute_index = isinstance(self.index, pandas.MultiIndex)
                is_recompute_columns = not is_recompute_index and isinstance(
                    self.columns, pandas.MultiIndex
                )
                new_index = compute_index(
                    self.index, self.columns, is_recompute_index, is_recompute_columns
                )
            elif is_tree_like_or_1d_index != is_tree_like_or_1d_cols:
                if isinstance(self.columns, pandas.MultiIndex) or not isinstance(
                    self.index, pandas.MultiIndex
                ):
                    return result
                else:
                    index = (
                        self.index.sortlevel()[0]
                        if is_tree_like_or_1d_index
                        and not is_tree_like_or_1d_cols
                        and isinstance(self.index, pandas.MultiIndex)
                        else self.index
                    )
                    index = pandas.MultiIndex.from_tuples(
                        list(index) * len(self.columns)
                    )
                    columns = self.columns.repeat(len(self.index))
                    index_levels = [
                        index.get_level_values(i) for i in range(index.nlevels)
                    ]
                    new_index = pandas.MultiIndex.from_arrays(
                        [columns] + index_levels,
                        names=self.columns.names + self.index.names,
                    )
            else:
                return result
            result = result.reindex(0, new_index)
        return result

    def stack(self, level, dropna):
        if not isinstance(self.columns, pandas.MultiIndex) or (
            isinstance(self.columns, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.columns.nlevels
        ):
            new_columns = [MODIN_UNNAMED_SERIES_LABEL]
        else:
            new_columns = None

        new_modin_frame = self._modin_frame.apply_full_axis(
            1,
            lambda df: pandas.DataFrame(df.stack(level=level, dropna=dropna)),
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    def isin(self, values, ignore_indices=False, shape_hint=None):
        if isinstance(values, type(self)):
            # HACK: if we don't cast to pandas, then the execution engine will try to
            # propagate the distributed Series to workers and most likely would have
            # some performance problems.
            # TODO: A better way of doing so could be passing this `values` as a query compiler
            # and broadcast accordingly.
            values = values.to_pandas()
            if ignore_indices:
                # Pandas logic is that it ignores indexing if 'values' is a 1D object
                values = values.squeeze(axis=1)

        def isin_func(df, values):
            if shape_hint == "column":
                df = df.squeeze(axis=1)
            res = df.isin(values)
            if isinstance(res, pandas.Series):
                res = res.to_frame(
                    MODIN_UNNAMED_SERIES_LABEL if res.name is None else res.name
                )
            return res

        return Map.register(isin_func, shape_hint=shape_hint, dtypes=np.bool_)(
            self, values
        )

    abs = Map.register(pandas.DataFrame.abs, dtypes="copy")
    applymap = Map.register(pandas.DataFrame.applymap)
    conj = Map.register(lambda df, *args, **kwargs: pandas.DataFrame(np.conj(df)))
    convert_dtypes = Fold.register(pandas.DataFrame.convert_dtypes)
    invert = Map.register(pandas.DataFrame.__invert__, dtypes="copy")
    isna = Map.register(pandas.DataFrame.isna, dtypes=np.bool_)
    _isfinite = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.isfinite(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _isinf = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.isinf(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _isnat = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.isnat(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _isneginf = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.isneginf(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _isposinf = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.isposinf(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _iscomplex = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.iscomplex(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _isreal = Map.register(  # Needed for numpy API
        lambda df, *args, **kwargs: pandas.DataFrame(np.isreal(df, *args, **kwargs)),
        dtypes=np.bool_,
    )
    _logical_not = Map.register(np.logical_not, dtypes=np.bool_)  # Needed for numpy API
    _tanh = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.tanh(df, *args, **kwargs))
    )  # Needed for numpy API
    _sqrt = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.sqrt(df, *args, **kwargs))
    )  # Needed for numpy API
    _exp = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.exp(df, *args, **kwargs))
    )  # Needed for numpy API
    negative = Map.register(pandas.DataFrame.__neg__)
    notna = Map.register(pandas.DataFrame.notna, dtypes=np.bool_)
    round = Map.register(pandas.DataFrame.round)
    replace = Map.register(pandas.DataFrame.replace)
    series_view = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            df.squeeze(axis=1).view(*args, **kwargs)
        )
    )
    to_numeric = Map.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            pandas.to_numeric(df.squeeze(axis=1), *args, **kwargs)
        )
    )
    to_timedelta = Map.register(
        lambda s, *args, **kwargs: pandas.to_timedelta(
            s.squeeze(axis=1), *args, **kwargs
        ).to_frame(),
        dtypes="timedelta64[ns]",
    )

    # END Map partitions operations

    # String map partitions operations

    str_capitalize = Map.register(_str_map("capitalize"), dtypes="copy")
    str_center = Map.register(_str_map("center"), dtypes="copy")
    str_contains = Map.register(_str_map("contains"), dtypes=np.bool_)
    str_count = Map.register(_str_map("count"), dtypes=int)
    str_endswith = Map.register(_str_map("endswith"), dtypes=np.bool_)
    str_find = Map.register(_str_map("find"), dtypes="copy")
    str_findall = Map.register(_str_map("findall"), dtypes="copy")
    str_get = Map.register(_str_map("get"), dtypes="copy")
    str_index = Map.register(_str_map("index"), dtypes="copy")
    str_isalnum = Map.register(_str_map("isalnum"), dtypes=np.bool_)
    str_isalpha = Map.register(_str_map("isalpha"), dtypes=np.bool_)
    str_isdecimal = Map.register(_str_map("isdecimal"), dtypes=np.bool_)
    str_isdigit = Map.register(_str_map("isdigit"), dtypes=np.bool_)
    str_islower = Map.register(_str_map("islower"), dtypes=np.bool_)
    str_isnumeric = Map.register(_str_map("isnumeric"), dtypes=np.bool_)
    str_isspace = Map.register(_str_map("isspace"), dtypes=np.bool_)
    str_istitle = Map.register(_str_map("istitle"), dtypes=np.bool_)
    str_isupper = Map.register(_str_map("isupper"), dtypes=np.bool_)
    str_join = Map.register(_str_map("join"), dtypes="copy")
    str_len = Map.register(_str_map("len"), dtypes=int)
    str_ljust = Map.register(_str_map("ljust"), dtypes="copy")
    str_lower = Map.register(_str_map("lower"), dtypes="copy")
    str_lstrip = Map.register(_str_map("lstrip"), dtypes="copy")
    str_match = Map.register(_str_map("match"), dtypes="copy")
    str_normalize = Map.register(_str_map("normalize"), dtypes="copy")
    str_pad = Map.register(_str_map("pad"), dtypes="copy")
    _str_partition = Map.register(_str_map("partition"), dtypes="copy")

    def str_partition(self, sep=" ", expand=True):
        # For `expand`, need an operator that can create more columns than before
        if expand:
            return super().str_partition(sep=sep, expand=expand)
        return self._str_partition(sep=sep, expand=False)

    str_repeat = Map.register(_str_map("repeat"), dtypes="copy")
    _str_extract = Map.register(_str_map("extract"), dtypes="copy")

    def str_extract(self, pat, flags, expand):
        regex = re.compile(pat, flags=flags)
        # need an operator that can create more columns than before
        if expand and regex.groups == 1:
            qc = self._str_extract(pat, flags=flags, expand=expand)
            qc.columns = get_group_names(regex)
        else:
            qc = super().str_extract(pat, flags=flags, expand=expand)
        return qc

    str_replace = Map.register(_str_map("replace"), dtypes="copy", shape_hint="column")
    str_rfind = Map.register(_str_map("rfind"), dtypes="copy", shape_hint="column")
    str_rindex = Map.register(_str_map("rindex"), dtypes="copy", shape_hint="column")
    str_rjust = Map.register(_str_map("rjust"), dtypes="copy", shape_hint="column")
    _str_rpartition = Map.register(
        _str_map("rpartition"), dtypes="copy", shape_hint="column"
    )

    def str_rpartition(self, sep=" ", expand=True):
        if expand:
            # For `expand`, need an operator that can create more columns than before
            return super().str_rpartition(sep=sep, expand=expand)
        return self._str_rpartition(sep=sep, expand=False)

    _str_rsplit = Map.register(_str_map("rsplit"), dtypes="copy", shape_hint="column")

    def str_rsplit(self, pat=None, n=-1, expand=False):
        if expand:
            # For `expand`, need an operator that can create more columns than before
            return super().str_rsplit(pat=pat, n=n, expand=expand)
        return self._str_rsplit(pat=pat, n=n, expand=False)

    str_rstrip = Map.register(_str_map("rstrip"), dtypes="copy", shape_hint="column")
    str_slice = Map.register(_str_map("slice"), dtypes="copy", shape_hint="column")
    str_slice_replace = Map.register(
        _str_map("slice_replace"), dtypes="copy", shape_hint="column"
    )
    _str_split = Map.register(_str_map("split"), dtypes="copy", shape_hint="column")

    def str_split(self, pat=None, n=-1, expand=False, regex=None):
        if expand:
            # For `expand`, need an operator that can create more columns than before
            return super().str_split(pat=pat, n=n, expand=expand, regex=regex)
        return self._str_split(pat=pat, n=n, expand=False, regex=regex)

    str_startswith = Map.register(
        _str_map("startswith"), dtypes=np.bool_, shape_hint="column"
    )
    str_strip = Map.register(_str_map("strip"), dtypes="copy", shape_hint="column")
    str_swapcase = Map.register(
        _str_map("swapcase"), dtypes="copy", shape_hint="column"
    )
    str_title = Map.register(_str_map("title"), dtypes="copy", shape_hint="column")
    str_translate = Map.register(
        _str_map("translate"), dtypes="copy", shape_hint="column"
    )
    str_upper = Map.register(_str_map("upper"), dtypes="copy", shape_hint="column")
    str_wrap = Map.register(_str_map("wrap"), dtypes="copy", shape_hint="column")
    str_zfill = Map.register(_str_map("zfill"), dtypes="copy", shape_hint="column")
    str___getitem__ = Map.register(
        _str_map("__getitem__"), dtypes="copy", shape_hint="column"
    )

    # END String map partitions operations

    def unique(self):
        new_modin_frame = self._modin_frame.apply_full_axis(
            0,
            lambda x: x.squeeze(axis=1).unique(),
            new_columns=self.columns,
        )
        return self.__constructor__(new_modin_frame)

    def searchsorted(self, **kwargs):
        def searchsorted(df):
            """Apply `searchsorted` function to a single partition."""
            result = df.squeeze(axis=1).searchsorted(**kwargs)
            if not is_list_like(result):
                result = [result]
            return pandas.DataFrame(result)

        return self.default_to_pandas(searchsorted)

    # Dt map partitions operations

    dt_date = Map.register(_dt_prop_map("date"), dtypes=np.object_)
    dt_time = Map.register(_dt_prop_map("time"), dtypes=np.object_)
    dt_timetz = Map.register(_dt_prop_map("timetz"), dtypes=np.object_)
    dt_year = Map.register(_dt_prop_map("year"), dtypes=np.int64)
    dt_month = Map.register(_dt_prop_map("month"), dtypes=np.int64)
    dt_day = Map.register(_dt_prop_map("day"), dtypes=np.int64)
    dt_hour = Map.register(_dt_prop_map("hour"), dtypes=np.int64)
    dt_minute = Map.register(_dt_prop_map("minute"), dtypes=np.int64)
    dt_second = Map.register(_dt_prop_map("second"), dtypes=np.int64)
    dt_microsecond = Map.register(_dt_prop_map("microsecond"), dtypes=np.int64)
    dt_nanosecond = Map.register(_dt_prop_map("nanosecond"), dtypes=np.int64)
    dt_week = Map.register(_dt_prop_map("week"), dtypes=np.int64)
    dt_weekofyear = Map.register(_dt_prop_map("weekofyear"), dtypes=np.int64)
    dt_dayofweek = Map.register(_dt_prop_map("dayofweek"), dtypes=np.int64)
    dt_weekday = Map.register(_dt_prop_map("weekday"), dtypes=np.int64)
    dt_dayofyear = Map.register(_dt_prop_map("dayofyear"), dtypes=np.int64)
    dt_quarter = Map.register(_dt_prop_map("quarter"), dtypes=np.int64)
    dt_is_month_start = Map.register(_dt_prop_map("is_month_start"), dtypes=np.bool_)
    dt_is_month_end = Map.register(_dt_prop_map("is_month_end"), dtypes=np.bool_)
    dt_is_quarter_start = Map.register(
        _dt_prop_map("is_quarter_start"), dtypes=np.bool_
    )
    dt_is_quarter_end = Map.register(_dt_prop_map("is_quarter_end"), dtypes=np.bool_)
    dt_is_year_start = Map.register(_dt_prop_map("is_year_start"), dtypes=np.bool_)
    dt_is_year_end = Map.register(_dt_prop_map("is_year_end"), dtypes=np.bool_)
    dt_is_leap_year = Map.register(_dt_prop_map("is_leap_year"), dtypes=np.bool_)
    dt_daysinmonth = Map.register(_dt_prop_map("daysinmonth"), dtypes=np.int64)
    dt_days_in_month = Map.register(_dt_prop_map("days_in_month"), dtypes=np.int64)
    dt_asfreq = Map.register(_dt_func_map("asfreq"))
    dt_to_period = Map.register(_dt_func_map("to_period"))
    dt_to_pydatetime = Map.register(_dt_func_map("to_pydatetime"), dtypes=np.object_)
    dt_tz_localize = Map.register(_dt_func_map("tz_localize"))
    dt_tz_convert = Map.register(_dt_func_map("tz_convert"))
    dt_normalize = Map.register(_dt_func_map("normalize"))
    dt_strftime = Map.register(_dt_func_map("strftime"), dtypes=np.object_)
    dt_round = Map.register(_dt_func_map("round"))
    dt_floor = Map.register(_dt_func_map("floor"))
    dt_ceil = Map.register(_dt_func_map("ceil"))
    dt_month_name = Map.register(_dt_func_map("month_name"), dtypes=np.object_)
    dt_day_name = Map.register(_dt_func_map("day_name"), dtypes=np.object_)
    dt_to_pytimedelta = Map.register(_dt_func_map("to_pytimedelta"), dtypes=np.object_)
    dt_total_seconds = Map.register(_dt_func_map("total_seconds"), dtypes=np.float64)
    dt_seconds = Map.register(_dt_prop_map("seconds"), dtypes=np.int64)
    dt_days = Map.register(_dt_prop_map("days"), dtypes=np.int64)
    dt_microseconds = Map.register(_dt_prop_map("microseconds"), dtypes=np.int64)
    dt_nanoseconds = Map.register(_dt_prop_map("nanoseconds"), dtypes=np.int64)
    dt_qyear = Map.register(_dt_prop_map("qyear"), dtypes=np.int64)
    dt_start_time = Map.register(_dt_prop_map("start_time"))
    dt_end_time = Map.register(_dt_prop_map("end_time"))
    dt_to_timestamp = Map.register(_dt_func_map("to_timestamp"))

    # END Dt map partitions operations

    def astype(self, col_dtypes, errors: str = "raise"):
        # `errors` parameter needs to be part of the function signature because
        # other query compilers may not take care of error handling at the API
        # layer. This query compiler assumes there won't be any errors due to
        # invalid type keys.
        return self.__constructor__(self._modin_frame.astype(col_dtypes, errors=errors))

    def infer_objects(self):
        return self.__constructor__(self._modin_frame.infer_objects())

    # Column/Row partitions reduce operations

    def first_valid_index(self):
        def first_valid_index_builder(df):
            """Get the position of the first valid index in a single partition."""
            return df.set_axis(pandas.RangeIndex(len(df.index)), axis="index").apply(
                lambda df: df.first_valid_index()
            )

        # We get the minimum from each column, then take the min of that to get
        # first_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(self._modin_frame.reduce(0, first_valid_index_builder))
            .min(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    def last_valid_index(self):
        def last_valid_index_builder(df):
            """Get the position of the last valid index in a single partition."""
            return df.set_axis(pandas.RangeIndex(len(df.index)), axis="index").apply(
                lambda df: df.last_valid_index()
            )

        # We get the maximum from each column, then take the max of that to get
        # last_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(self._modin_frame.reduce(0, last_valid_index_builder))
            .max(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    # END Column/Row partitions reduce operations

    def describe(
        self,
        percentiles: np.ndarray,
        datetime_is_numeric: bool,
    ):
        # Use pandas to calculate the correct columns
        empty_df = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .describe(
                percentiles, datetime_is_numeric=datetime_is_numeric, include="all"
            )
        )
        new_index = empty_df.index

        # Note: `describe` convert timestamp type to object type
        # which results in the loss of two values in index: `first` and `last`
        # for empty DataFrame.
        if not any(map(is_numeric_dtype, empty_df.dtypes)) and not datetime_is_numeric:
            for col_name in empty_df.dtypes.index:
                # if previosly type of `col_name` was datetime or timedelta
                if is_datetime_or_timedelta_dtype(self.dtypes[col_name]):
                    new_index = pandas.Index(
                        empty_df.index.to_list() + ["first"] + ["last"]
                    )
                    break

        def describe_builder(df, internal_indices=[]):  # pragma: no cover
            """Apply `describe` function to the subset of columns in a single partition."""
            # The index of the resulting dataframe is the same amongst all partitions
            # when dealing with the same data type. However, if we work with columns
            # that contain strings, we can get extra values in our result index such as
            # 'unique', 'top', and 'freq'. Since we call describe() on each partition,
            # we can have cases where certain partitions do not contain any of the
            # object string data leading to an index mismatch between partitions.
            # Thus, we must reindex each partition with the global new_index.
            return (
                df.iloc[:, internal_indices]
                .describe(
                    percentiles=percentiles,
                    datetime_is_numeric=datetime_is_numeric,
                    include="all",
                )
                .reindex(new_index)
            )

        return self.__constructor__(
            self._modin_frame.apply_full_axis_select_indices(
                0,
                describe_builder,
                empty_df.columns,
                new_index=new_index,
                new_columns=empty_df.columns,
            )
        )

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    cummax = Fold.register(pandas.DataFrame.cummax)
    cummin = Fold.register(pandas.DataFrame.cummin)
    cumsum = Fold.register(pandas.DataFrame.cumsum)
    cumprod = Fold.register(pandas.DataFrame.cumprod)
    _diff = Fold.register(pandas.DataFrame.diff)

    def diff(self, axis, periods):
        return self._diff(fold_axis=axis, axis=axis, periods=periods)

    def clip(self, lower, upper, **kwargs):
        if isinstance(lower, BaseQueryCompiler):
            lower = lower.to_pandas().squeeze(1)
        if isinstance(upper, BaseQueryCompiler):
            upper = upper.to_pandas().squeeze(1)
        kwargs["upper"] = upper
        kwargs["lower"] = lower
        axis = kwargs.get("axis", 0)
        if is_list_like(lower) or is_list_like(upper):
            new_modin_frame = self._modin_frame.fold(axis, lambda df: df.clip(**kwargs))
        else:
            new_modin_frame = self._modin_frame.map(lambda df: df.clip(**kwargs))
        return self.__constructor__(new_modin_frame)

    def corr(self, method="pearson", min_periods=1):
        if method == "pearson":
            numeric_self = self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
            return numeric_self._nancorr(min_periods=min_periods)
        else:
            return super().corr(method=method, min_periods=min_periods)

    def cov(self, min_periods=None, ddof=1):
        return self._nancorr(min_periods=min_periods, cov=True, ddof=ddof)

    def _nancorr(self, min_periods=1, cov=False, ddof=1):
        """
        Compute either pairwise covariance or pairwise correlation of columns.

        This function considers NA/null values the same like pandas does.

        Parameters
        ----------
        min_periods : int, default: 1
            Minimum number of observations required per pair of columns
            to have a valid result.
        cov : boolean, default: False
            Either covariance or correlation should be computed.
        ddof : int, default: 1
            Means Delta Degrees of Freedom. The divisor used in calculations.

        Returns
        -------
        PandasQueryCompiler
            The covariance or correlation matrix.
        """
        other = self.to_numpy()
        other_mask = self._isfinite().to_numpy()
        n_cols = other.shape[1]

        if min_periods is None:
            min_periods = 1

        def map_func(df):  # pragma: no cover
            """Compute covariance or correlation matrix for the passed frame."""
            df = df.to_numpy()
            n_rows = df.shape[0]
            df_mask = np.isfinite(df)

            result = np.empty((n_rows, n_cols), dtype=np.float64)

            for i in range(n_rows):
                df_ith_row = df[i]
                df_ith_mask = df_mask[i]

                for j in range(n_cols):
                    other_jth_col = other[:, j]

                    valid = df_ith_mask & other_mask[:, j]

                    vx = df_ith_row[valid]
                    vy = other_jth_col[valid]

                    nobs = len(vx)

                    if nobs < min_periods:
                        result[i, j] = np.nan
                    else:
                        vx = vx - vx.mean()
                        vy = vy - vy.mean()
                        sumxy = (vx * vy).sum()
                        sumxx = (vx * vx).sum()
                        sumyy = (vy * vy).sum()

                        denom = (nobs - ddof) if cov else np.sqrt(sumxx * sumyy)
                        if denom != 0:
                            result[i, j] = sumxy / denom
                        else:
                            result[i, j] = np.nan

            return pandas.DataFrame(result)

        columns = self.columns
        index = columns.copy()
        transponed_self = self.transpose()
        new_modin_frame = transponed_self._modin_frame.apply_full_axis(
            1, map_func, new_index=index, new_columns=columns
        )
        return transponed_self.__constructor__(new_modin_frame)

    def dot(self, other, squeeze_self=None, squeeze_other=None):
        if isinstance(other, PandasQueryCompiler):
            other = (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )

        def map_func(df, other=other, squeeze_self=squeeze_self):  # pragma: no cover
            """Compute matrix multiplication of the passed frames."""
            result = df.squeeze(axis=1).dot(other) if squeeze_self else df.dot(other)
            if is_list_like(result):
                return pandas.DataFrame(result)
            else:
                return pandas.DataFrame([result])

        num_cols = other.shape[1] if len(other.shape) > 1 else 1
        if len(self.columns) == 1:
            new_index = (
                [MODIN_UNNAMED_SERIES_LABEL]
                if (len(self.index) == 1 or squeeze_self) and num_cols == 1
                else None
            )
            new_columns = (
                [MODIN_UNNAMED_SERIES_LABEL] if squeeze_self and num_cols == 1 else None
            )
            axis = 0
        else:
            new_index = self.index
            new_columns = [MODIN_UNNAMED_SERIES_LABEL] if num_cols == 1 else None
            axis = 1

        new_modin_frame = self._modin_frame.apply_full_axis(
            axis, map_func, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def _nsort(self, n, columns=None, keep="first", sort_type="nsmallest"):
        """
        Return first N rows of the data sorted in the specified order.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : list of labels, optional
            Column labels to sort data by.
        keep : {"first", "last", "all"}, default: "first"
            How to pick first rows in case of duplicated values:
            - "first": prioritize first occurrence.
            - "last": prioritize last occurrence.
            - "all": do not drop any duplicates, even if it means selecting more than `n` rows.
        sort_type : {"nsmallest", "nlargest"}, default: "nsmallest"
            "nsmallest" means sort in descending order, "nlargest" means
            sort in ascending order.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the first N rows of the data
            sorted in the given order.
        """

        def map_func(df, n=n, keep=keep, columns=columns):  # pragma: no cover
            """Return first `N` rows of the sorted data for a single partition."""
            if columns is None:
                return pandas.DataFrame(
                    getattr(pandas.Series, sort_type)(
                        df.squeeze(axis=1), n=n, keep=keep
                    )
                )
            return getattr(pandas.DataFrame, sort_type)(
                df, n=n, columns=columns, keep=keep
            )

        if columns is None:
            new_columns = [MODIN_UNNAMED_SERIES_LABEL]
        else:
            new_columns = self.columns

        new_modin_frame = self._modin_frame.apply_full_axis(
            axis=0, func=map_func, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def nsmallest(self, *args, **kwargs):
        return self._nsort(sort_type="nsmallest", *args, **kwargs)

    def nlargest(self, *args, **kwargs):
        return self._nsort(sort_type="nlargest", *args, **kwargs)

    def eval(self, expr, **kwargs):
        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        empty_eval = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .eval(expr, inplace=False, **kwargs)
        )
        if isinstance(empty_eval, pandas.Series):
            new_columns = (
                [empty_eval.name]
                if empty_eval.name is not None
                else [MODIN_UNNAMED_SERIES_LABEL]
            )
        else:
            new_columns = empty_eval.columns
        new_modin_frame = self._modin_frame.apply_full_axis(
            1,
            lambda df: pandas.DataFrame(df.eval(expr, inplace=False, **kwargs)),
            new_index=self.index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def mode(self, **kwargs):
        axis = kwargs.get("axis", 0)

        def mode_builder(df):  # pragma: no cover
            """Compute modes for a single partition."""
            result = pandas.DataFrame(df.mode(**kwargs))
            # We return a dataframe with the same shape as the input to ensure
            # that all the partitions will be the same shape
            if axis == 0 and len(df) != len(result):
                # Pad rows
                result = result.reindex(index=pandas.RangeIndex(len(df.index)))
            elif axis == 1 and len(df.columns) != len(result.columns):
                # Pad columns
                result = result.reindex(columns=pandas.RangeIndex(len(df.columns)))
            return pandas.DataFrame(result)

        if axis == 0:
            new_index = pandas.RangeIndex(len(self.index))
            new_columns = self.columns
        else:
            new_index = self.index
            new_columns = pandas.RangeIndex(len(self.columns))
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis, mode_builder, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame).dropna(axis=axis, how="all")

    def fillna(self, **kwargs):
        squeeze_self = kwargs.pop("squeeze_self", False)
        squeeze_value = kwargs.pop("squeeze_value", False)
        axis = kwargs.get("axis", 0)
        value = kwargs.pop("value")
        method = kwargs.get("method", None)
        limit = kwargs.get("limit", None)
        full_axis = method is not None or limit is not None
        if isinstance(value, BaseQueryCompiler):
            if squeeze_self:
                # Self is a Series type object
                if full_axis:
                    value = value.to_pandas().squeeze(axis=1)

                    def fillna_builder(series):  # pragma: no cover
                        # `limit` parameter works only on `Series` type, so we have to squeeze both objects to get
                        # correct behavior.
                        return series.squeeze(axis=1).fillna(value=value, **kwargs)

                    new_modin_frame = self._modin_frame.apply_full_axis(
                        0, fillna_builder
                    )
                else:

                    def fillna_builder(df, value_arg):
                        if isinstance(value_arg, pandas.DataFrame):
                            value_arg = value_arg.squeeze(axis=1)
                        res = df.squeeze(axis=1).fillna(value=value_arg, **kwargs)
                        return pandas.DataFrame(res)

                    new_modin_frame = self._modin_frame.n_ary_op(
                        fillna_builder,
                        [value._modin_frame],
                        join_type="left",
                        copartition_along_columns=False,
                    )

                return self.__constructor__(new_modin_frame)
            else:
                # Self is a DataFrame type object
                if squeeze_value:
                    # Value is Series type object
                    value = value.to_pandas().squeeze(axis=1)

                    def fillna(df):
                        return df.fillna(value=value, **kwargs)

                    # Continue to end of this function

                else:
                    # Value is a DataFrame type object
                    def fillna_builder(df, right):
                        return df.fillna(value=right, **kwargs)

                    new_modin_frame = self._modin_frame.broadcast_apply(
                        0, fillna_builder, value._modin_frame
                    )
                    return self.__constructor__(new_modin_frame)

        elif isinstance(value, dict):
            if squeeze_self:
                # For Series dict works along the index.
                def fillna(df):
                    return pandas.DataFrame(
                        df.squeeze(axis=1).fillna(value=value, **kwargs)
                    )

            else:
                # For DataFrames dict works along columns, all columns have to be present.
                def fillna(df):
                    func_dict = {
                        col: val for (col, val) in value.items() if col in df.columns
                    }
                    return df.fillna(value=func_dict, **kwargs)

        else:

            def fillna(df):
                return df.fillna(value=value, **kwargs)

        if full_axis:
            new_modin_frame = self._modin_frame.fold(axis, fillna)
        else:
            new_modin_frame = self._modin_frame.map(fillna)
        return self.__constructor__(new_modin_frame)

    def quantile_for_list_of_values(self, **kwargs):
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q")
        numeric_only = kwargs.get("numeric_only", True)
        assert isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list))

        if numeric_only:
            new_columns = self._modin_frame.numeric_columns()
        else:
            new_columns = [
                col
                for col, dtype in zip(self.columns, self.dtypes)
                if (is_numeric_dtype(dtype) or is_datetime_or_timedelta_dtype(dtype))
            ]
        if axis == 1:
            query_compiler = self.getitem_column_array(new_columns)
            new_columns = self.index
        else:
            query_compiler = self

        def quantile_builder(df, **kwargs):
            result = df.quantile(**kwargs)
            return result.T if kwargs.get("axis", 0) == 1 else result

        # This took a long time to debug, so here is the rundown of why this is needed.
        # Previously, we were operating on select indices, but that was broken. We were
        # not correctly setting the columns/index. Because of how we compute `to_pandas`
        # and because of the static nature of the index for `axis=1` it is easier to
        # just handle this as the transpose (see `quantile_builder` above for the
        # transpose within the partition) than it is to completely rework other
        # internal methods. Basically we are returning the transpose of the object for
        # correctness and cleanliness of the code.
        if axis == 1:
            q_index = new_columns
            new_columns = pandas.Float64Index(q)
        else:
            q_index = pandas.Float64Index(q)
        new_modin_frame = query_compiler._modin_frame.apply_full_axis(
            axis,
            lambda df: quantile_builder(df, **kwargs),
            new_index=q_index,
            new_columns=new_columns,
            dtypes=np.float64,
        )
        result = self.__constructor__(new_modin_frame)
        return result.transpose() if axis == 1 else result

    def query(self, expr, **kwargs):
        def query_builder(df, **modin_internal_kwargs):
            return df.query(expr, inplace=False, **kwargs, **modin_internal_kwargs)

        return self.__constructor__(self._modin_frame.filter(1, query_builder))

    def rank(self, **kwargs):
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: df.rank(**kwargs),
            new_index=self.index,
            new_columns=self.columns if not numeric_only else None,
            dtypes=np.float64,
        )
        return self.__constructor__(new_modin_frame)

    def sort_index(self, **kwargs):
        axis = kwargs.pop("axis", 0)
        level = kwargs.pop("level", None)
        sort_remaining = kwargs.pop("sort_remaining", True)
        kwargs["inplace"] = False

        if level is not None or self.has_multiindex(axis=axis):
            return self.default_to_pandas(
                pandas.DataFrame.sort_index,
                axis=axis,
                level=level,
                sort_remaining=sort_remaining,
                **kwargs,
            )

        # sort_index can have ascending be None and behaves as if it is False.
        # sort_values cannot have ascending be None. Thus, the following logic is to
        # convert the ascending argument to one that works with sort_values
        ascending = kwargs.pop("ascending", True)
        if ascending is None:
            ascending = False
        kwargs["ascending"] = ascending
        if axis:
            new_columns = self.columns.to_frame().sort_index(**kwargs).index
            new_index = self.index
        else:
            new_index = self.index.to_frame().sort_index(**kwargs).index
            new_columns = self.columns
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: df.sort_index(
                axis=axis, level=level, sort_remaining=sort_remaining, **kwargs
            ),
            new_index,
            new_columns,
            dtypes="copy" if axis == 0 else None,
        )
        return self.__constructor__(new_modin_frame)

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index=True,
    ):
        ErrorMessage.missmatch_with_pandas(
            operation="melt", message="Order of rows could be different from pandas"
        )

        if var_name is None:
            var_name = "variable"

        def _convert_to_list(x):
            """Convert passed object to a list."""
            if is_list_like(x):
                x = [*x]
            elif x is not None:
                x = [x]
            else:
                x = []
            return x

        id_vars, value_vars = map(_convert_to_list, [id_vars, value_vars])

        if len(value_vars) == 0:
            value_vars = self.columns.drop(id_vars)

        if len(id_vars) != 0:
            to_broadcast = self.getitem_column_array(id_vars)._modin_frame
        else:
            to_broadcast = None

        def applyier(df, internal_indices, other=[], internal_other_indices=[]):
            """
            Apply `melt` function to a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            internal_indices : list of ints
                Positional indices of columns in this particular partition which
                represents `value_vars` columns in the source frame.
            other : pandas.DataFrame
                Broadcasted partition which contains `id_vars` columns of the
                source frame.
            internal_other_indices : list of ints
                Positional indices of columns in `other` partition which
                represents `id_vars` columns in the source frame.

            Returns
            -------
            pandas.DataFrame
                The result of the `melt` function for this particular partition.
            """
            if len(other):
                other = pandas.concat(other, axis=1)
                columns_to_add = other.columns.difference(df.columns)
                df = pandas.concat([df, other[columns_to_add]], axis=1)
            return df.melt(
                id_vars=id_vars,
                value_vars=df.columns[internal_indices],
                var_name=var_name,
                value_name=value_name,
                col_level=col_level,
            )

        # we have no able to calculate correct indices here, so making it `dummy_index`
        inconsistent_frame = self._modin_frame.broadcast_apply_select_indices(
            axis=0,
            apply_indices=value_vars,
            func=applyier,
            other=to_broadcast,
            new_index=["dummy_index"] * len(id_vars),
            new_columns=["dummy_index"] * len(id_vars),
        )
        # after applying `melt` for selected indices we will get partitions like this:
        #     id_vars   vars   value |     id_vars   vars   value
        #  0      foo   col3       1 |  0      foo   col5       a    so stacking it into
        #  1      fiz   col3       2 |  1      fiz   col5       b    `new_parts` to get
        #  2      bar   col3       3 |  2      bar   col5       c    correct answer
        #  3      zoo   col3       4 |  3      zoo   col5       d
        new_parts = np.array(
            [np.array([x]) for x in np.concatenate(inconsistent_frame._partitions.T)]
        )
        new_index = pandas.RangeIndex(len(self.index) * len(value_vars))
        new_modin_frame = self._modin_frame.__constructor__(
            new_parts,
            index=new_index,
            columns=id_vars + [var_name, value_name],
        )
        result = self.__constructor__(new_modin_frame)
        # this assigment needs to propagate correct indices into partitions
        result.index = new_index
        return result

    # END Map across rows/columns

    # __getitem__ methods
    __getitem_bool = Binary.register(
        lambda df, r: df[[r]] if is_scalar(r) else df[r],
        join_type="left",
        labels="drop",
    )

    # __setitem__ methods
    def setitem_bool(self, row_loc, col_loc, item):
        def _set_item(df, row_loc):  # pragma: no cover
            df = df.copy()
            df.loc[row_loc.squeeze(axis=1), col_loc] = item
            return df

        if self._modin_frame.has_materialized_dtypes and is_scalar(item):
            new_dtypes = self.dtypes.copy()
            old_dtypes = new_dtypes[col_loc]

            if hasattr(item, "dtype"):
                # If we're dealing with a numpy scalar (np.int, np.datetime64, ...)
                # we would like to get its internal dtype
                item_type = item.dtype
            elif hasattr(item, "to_numpy"):
                # If we're dealing with a scalar that can be converted to numpy (for example pandas.Timestamp)
                # we would like to convert it and get its proper internal dtype
                item_type = item.to_numpy().dtype
            else:
                item_type = type(item)

            if isinstance(old_dtypes, pandas.Series):
                new_dtypes[col_loc] = [
                    find_common_type([dtype, item_type]) for dtype in old_dtypes.values
                ]
            else:
                new_dtypes[col_loc] = find_common_type([old_dtypes, item_type])
        else:
            new_dtypes = None

        new_modin_frame = self._modin_frame.broadcast_apply_full_axis(
            axis=1,
            func=_set_item,
            other=row_loc._modin_frame,
            new_index=self._modin_frame.copy_index_cache(),
            new_columns=self._modin_frame.copy_columns_cache(),
            keep_partitioning=False,
            dtypes=new_dtypes,
        )
        return self.__constructor__(new_modin_frame)

    # END __setitem__ methods

    def __validate_bool_indexer(self, indexer):
        if len(indexer) != len(self.index):
            raise ValueError(
                f"Item wrong length {len(indexer)} instead of {len(self.index)}."
            )
        if isinstance(indexer, pandas.Series) and not indexer.equals(self.index):
            warnings.warn(
                "Boolean Series key will be reindexed to match DataFrame index.",
                PendingDeprecationWarning,
                stacklevel=4,
            )

    def getitem_array(self, key):
        if isinstance(key, type(self)):
            # here we check for a subset of bool indexers only to simplify the code;
            # there could (potentially) be more of those, but we assume the most frequent
            # ones are just of bool dtype
            if len(key.dtypes) == 1 and is_bool_dtype(key.dtypes[0]):
                self.__validate_bool_indexer(key.index)
                return self.__getitem_bool(key, broadcast=True, dtypes="copy")

            key = key.to_pandas().squeeze(axis=1)

        if is_bool_indexer(key):
            self.__validate_bool_indexer(key)
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pandas.RangeIndex(len(self.index))[key]
            if len(key):
                return self.getitem_row_array(key)
            else:
                return self.from_pandas(
                    pandas.DataFrame(columns=self.columns), type(self._modin_frame)
                )
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return self.getitem_column_array(key)

    def getitem_column_array(self, key, numeric=False):
        shape_hint = "column" if len(key) == 1 else None
        if numeric:
            new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
                col_positions=key
            )
        else:
            new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
                col_labels=key
            )
        return self.__constructor__(new_modin_frame, shape_hint=shape_hint)

    def getitem_row_array(self, key):
        return self.__constructor__(
            self._modin_frame.take_2d_labels_or_positional(row_positions=key)
        )

    def setitem(self, axis, key, value):
        return self._setitem(axis=axis, key=key, value=value, how=None)

    def _setitem(self, axis, key, value, how="inner"):
        """
        Set the row/column defined by `key` to the `value` provided.

        In contrast with `setitem` with this function you can specify how
        to handle non-aligned `self` and `value`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to set `value` along. 0 means set row, 1 means set column.
        key : scalar
            Row/column label to set `value` in.
        value : PandasQueryCompiler (1xN), list-like or scalar
            Define new row/column value.
        how : {"inner", "outer", "left", "right", None}, default: "inner"
            Type of join to perform if specified axis of `self` and `value` are not
            equal. If `how` is `None`, reindex `value` with `self` labels without joining.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated `key` value.
        """

        def setitem_builder(df, internal_indices=[]):  # pragma: no cover
            """
            Set the row/column to the `value` in a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            internal_indices : list of ints
                Positional indices of rows/columns in this particular partition
                which represents `key` in the source frame.

            Returns
            -------
            pandas.DataFrame
                Partition data with updated values.
            """
            df = df.copy()
            if len(internal_indices) == 1:
                if axis == 0:
                    df[df.columns[internal_indices[0]]] = value
                else:
                    df.iloc[internal_indices[0]] = value
            else:
                if axis == 0:
                    df[df.columns[internal_indices]] = value
                else:
                    df.iloc[internal_indices] = value
            return df

        if isinstance(value, type(self)):
            value.columns = [key]
            if axis == 1:
                value = value.transpose()
            idx = self.get_axis(axis ^ 1).get_indexer_for([key])[0]
            return self.insert_item(axis ^ 1, idx, value, how, replace=True)

        # TODO: rework by passing list-like values to `apply_select_indices`
        # as an item to distribute
        if is_list_like(value):
            new_modin_frame = self._modin_frame.apply_full_axis_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        else:
            new_modin_frame = self._modin_frame.apply_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        return self.__constructor__(new_modin_frame)

    # END __getitem__ methods

    # Drop/Dropna
    # This will change the shape of the resulting data.
    def dropna(self, **kwargs):
        return self.__constructor__(
            self._modin_frame.filter(
                kwargs.get("axis", 0) ^ 1,
                lambda df: pandas.DataFrame.dropna(df, **kwargs),
            )
        )

    def drop(self, index=None, columns=None, errors: str = "raise"):
        # `errors` parameter needs to be part of the function signature because
        # other query compilers may not take care of error handling at the API
        # layer. This query compiler assumes there won't be any errors due to
        # invalid keys.
        if index is not None:
            index = np.sort(self.index.get_indexer_for(self.index.difference(index)))
        if columns is not None:
            columns = np.sort(
                self.columns.get_indexer_for(self.columns.difference(columns))
            )
        new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
            row_positions=index, col_positions=columns
        )
        return self.__constructor__(new_modin_frame)

    # END Drop/Dropna

    def duplicated(self, **kwargs):
        def _compute_hash(df):
            result = df.apply(
                lambda s: hashlib.new("md5", str(tuple(s)).encode()).hexdigest(), axis=1
            )
            if isinstance(result, pandas.Series):
                result = result.to_frame(
                    result.name
                    if result.name is not None
                    else MODIN_UNNAMED_SERIES_LABEL
                )
            return result

        def _compute_duplicated(df):  # pragma: no cover
            result = df.duplicated(**kwargs)
            if isinstance(result, pandas.Series):
                result = result.to_frame(
                    result.name
                    if result.name is not None
                    else MODIN_UNNAMED_SERIES_LABEL
                )
            return result

        if self._modin_frame._partitions.shape[1] > 1:
            # if the number of columns (or column partitions) we are checking for duplicates is larger than 1,
            # we must first hash them to generate a single value that can be compared across rows.
            hashed_modin_frame = self._modin_frame.reduce(
                axis=1,
                function=_compute_hash,
                dtypes=np.dtype("O"),
            )
        else:
            hashed_modin_frame = self._modin_frame
        new_modin_frame = hashed_modin_frame.apply_full_axis(
            axis=0,
            func=_compute_duplicated,
            new_index=self._modin_frame.copy_index_cache(),
            new_columns=[MODIN_UNNAMED_SERIES_LABEL],
            dtypes=np.bool_,
            keep_partitioning=False,
        )
        return self.__constructor__(new_modin_frame, shape_hint="column")

    # Insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    def insert(self, loc, column, value):
        if isinstance(value, type(self)):
            value.columns = [column]
            return self.insert_item(axis=1, loc=loc, value=value, how=None)

        def insert(df, internal_indices=[]):  # pragma: no cover
            """
            Insert new column to the partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            internal_indices : list of ints
                Positional index of the column in this particular partition
                to insert new column after.
            """
            internal_idx = int(internal_indices[0])
            df.insert(internal_idx, column, value)
            return df

        # TODO: rework by passing list-like values to `apply_select_indices`
        # as an item to distribute
        new_modin_frame = self._modin_frame.apply_full_axis_select_indices(
            0,
            insert,
            numeric_indices=[loc],
            keep_remaining=True,
            new_index=self.index,
            new_columns=self.columns.insert(loc, column),
        )
        return self.__constructor__(new_modin_frame)

    # END Insert

    def explode(self, column):
        return self.__constructor__(
            self._modin_frame.explode(1, lambda df: df.explode(column))
        )

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        # if any of args contain modin object, we should
        # convert it to pandas
        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)
        if isinstance(func, dict):
            return self._dict_func(func, axis, *args, **kwargs)
        elif is_list_like(func):
            return self._list_like_func(func, axis, *args, **kwargs)
        else:
            return self._callable_func(func, axis, *args, **kwargs)

    def apply_on_series(self, func, *args, **kwargs):
        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)

        assert self.is_series_like()

        # We use apply_full_axis here instead of map since the latter assumes that the
        # shape of the DataFrame does not change. However, it is possible for functions
        # applied to Series objects to end up creating DataFrames. It is possible that
        # using apply_full_axis is much less performant compared to using a variant of
        # map.
        return self.__constructor__(
            self._modin_frame.apply_full_axis(
                1, lambda df: df.squeeze(axis=1).apply(func, *args, **kwargs)
            )
        )

    def _dict_func(self, func, axis, *args, **kwargs):
        """
        Apply passed functions to the specified rows/columns.

        Parameters
        ----------
        func : dict(label) -> [callable, str]
            Dictionary that maps axis labels to the function to apply against them.
        axis : {0, 1}
            Target axis to apply functions along. 0 means apply to columns,
            1 means apply to rows.
        *args : args
            Arguments to pass to the specified functions.
        **kwargs : kwargs
            Arguments to pass to the specified functions.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the results of passed functions.
        """
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        func = {k: wrap_udf_function(v) if callable(v) else v for k, v in func.items()}

        def dict_apply_builder(df, internal_indices=[]):  # pragma: no cover
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            # It looks like it doesn't need to use `internal_indices` option internally
            # for the case since `apply` use labels from dictionary keys in `func` variable.
            return pandas.DataFrame(df.apply(func, *args, **kwargs))

        labels = list(func.keys())
        return self.__constructor__(
            self._modin_frame.apply_full_axis_select_indices(
                axis,
                dict_apply_builder,
                labels,
                new_index=labels if axis == 1 else None,
                new_columns=labels if axis == 0 else None,
                keep_remaining=False,
            )
        )

    def _list_like_func(self, func, axis, *args, **kwargs):
        """
        Apply passed functions to each row/column.

        Parameters
        ----------
        func : list of callable
            List of functions to apply against each row/column.
        axis : {0, 1}
            Target axis to apply functions along. 0 means apply to columns,
            1 means apply to rows.
        *args : args
            Arguments to pass to the specified functions.
        **kwargs : kwargs
            Arguments to pass to the specified functions.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the results of passed functions.
        """
        # When the function is list-like, the function names become the index/columns
        new_index = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 0
            else self.index
        )
        new_columns = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 1
            else self.columns
        )
        func = [wrap_udf_function(f) if callable(f) else f for f in func]
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def _callable_func(self, func, axis, *args, **kwargs):
        """
        Apply passed function to each row/column.

        Parameters
        ----------
        func : callable or str
            Function to apply.
        axis : {0, 1}
            Target axis to apply function along. 0 means apply to columns,
            1 means apply to rows.
        *args : args
            Arguments to pass to the specified function.
        **kwargs : kwargs
            Arguments to pass to the specified function.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the results of passed function
            for each row/column.
        """
        if callable(func):
            func = wrap_udf_function(func)

        new_modin_frame = self._modin_frame.apply_full_axis(
            axis, lambda df: df.apply(func, axis=axis, *args, **kwargs)
        )
        return self.__constructor__(new_modin_frame)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    def _groupby_internal_columns(self, by, drop):
        """
        Extract internal columns from by argument of groupby.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list
        drop : bool
            Indicates whether or not by data came from self frame.
            True, by data came from self. False, external by data.

        Returns
        -------
        by : list of BaseQueryCompiler, column or index label, or Grouper
        internal_by : list of str
            List of internal column name to be dropped during groupby.
        """
        if isinstance(by, type(self)):
            if drop:
                internal_by = by.columns
                by = [by]
            else:
                internal_by = []
                by = [by]
        else:
            if not isinstance(by, list):
                by = [by] if by is not None else []
            internal_by = []
            for o in by:
                if isinstance(o, pandas.Grouper) and o.key in self.columns:
                    internal_by.append(o.key)
                elif hashable(o) and o in self.columns:
                    internal_by.append(o)
            internal_qc = (
                [self.getitem_column_array(internal_by)] if len(internal_by) else []
            )
            by = internal_qc + by[len(internal_by) :]
        return by, internal_by

    groupby_all = GroupbyReduceImpl.build_qc_method("all")
    groupby_any = GroupbyReduceImpl.build_qc_method("any")
    groupby_count = GroupbyReduceImpl.build_qc_method("count")
    groupby_max = GroupbyReduceImpl.build_qc_method("max")
    groupby_min = GroupbyReduceImpl.build_qc_method("min")
    groupby_prod = GroupbyReduceImpl.build_qc_method("prod")
    groupby_sum = GroupbyReduceImpl.build_qc_method("sum")
    groupby_skew = GroupbyReduceImpl.build_qc_method("skew")

    def groupby_nth(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        result = super().groupby_nth(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        if not groupby_kwargs.get("as_index", True):
            # pandas keeps order of columns intact, follow suit
            return result.getitem_column_array(self.columns)
        return result

    def groupby_mean(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        if ExperimentalGroupbyImpl.get():
            try:
                return self._groupby_shuffle(
                    by=by,
                    agg_func="mean",
                    axis=axis,
                    groupby_kwargs=groupby_kwargs,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                )
            except NotImplementedError as e:
                ErrorMessage.warn(
                    f"Can't use experimental reshuffling groupby implementation because of: {e}"
                    + "\nFalling back to a TreeReduce implementation."
                )

        _, internal_by = self._groupby_internal_columns(by, drop)

        numeric_only = agg_kwargs.get("numeric_only", False)
        datetime_cols = (
            {
                col: dtype
                for col, dtype in zip(self.dtypes.index, self.dtypes)
                if is_datetime64_any_dtype(dtype) and col not in internal_by
            }
            if not numeric_only
            else dict()
        )

        if len(datetime_cols) > 0:
            datetime_qc = self.getitem_array(datetime_cols)
            if datetime_qc.isna().any().any(axis=1).to_pandas().squeeze():
                return super().groupby_mean(
                    by=by,
                    axis=axis,
                    groupby_kwargs=groupby_kwargs,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                )

        qc_with_converted_datetime_cols = (
            self.astype({col: "int64" for col in datetime_cols.keys()})
            if len(datetime_cols) > 0
            else self
        )

        result = GroupbyReduceImpl.build_qc_method("mean")(
            query_compiler=qc_with_converted_datetime_cols,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

        if len(datetime_cols) > 0:
            result = result.astype({col: dtype for col, dtype in datetime_cols.items()})
        return result

    def groupby_size(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        if ExperimentalGroupbyImpl.get():
            try:
                return self._groupby_shuffle(
                    by=by,
                    agg_func="size",
                    axis=axis,
                    groupby_kwargs=groupby_kwargs,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                )
            except NotImplementedError as e:
                ErrorMessage.warn(
                    f"Can't use experimental reshuffling groupby implementation because of: {e}"
                    + "\nFalling back to a TreeReduce implementation."
                )

        result = self._groupby_dict_reduce(
            by=by,
            axis=axis,
            agg_func={self.columns[0]: [("__size_col__", "size")]},
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            groupby_kwargs=groupby_kwargs,
            drop=drop,
            method="size",
            default_to_pandas_func=lambda grp: grp.size(),
        )
        if groupby_kwargs.get("as_index", True):
            result.columns = [MODIN_UNNAMED_SERIES_LABEL]
        elif isinstance(result.columns, pandas.MultiIndex):
            # Dropping one extra-level which was added because of renaming aggregation
            result.columns = (
                result.columns[:-1].droplevel(-1).append(pandas.Index(["size"]))
            )
        return result

    def _groupby_dict_reduce(
        self,
        by,
        agg_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        **kwargs,
    ):
        """
        Group underlying data and apply aggregation functions to each group of the specified column/row.

        This method is responsible of performing dictionary groupby aggregation for such functions,
        that can be implemented via TreeReduce approach.

        Parameters
        ----------
        by : PandasQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        agg_func : dict(label) -> str
            Dictionary that maps row/column labels to the function names.
            **Note:** specified functions have to be supported by ``modin.core.dataframe.algebra.GroupByReduce``.
            Supported functions are listed in the ``modin.core.dataframe.algebra.GroupByReduce.groupby_reduce_functions``
            dictionary.
        axis : {0, 1}
            Axis to group and apply aggregation function along.
            0 is for index, when 1 is for columns.
        groupby_kwargs : dict
            GroupBy parameters in the format of ``modin.pandas.DataFrame.groupby`` signature.
        agg_args : list-like
            Serves the compatibility purpose. Does not affect the result.
        agg_kwargs : dict
            Arguments to pass to the aggregation functions.
        drop : bool, default: False
            If `by` is a QueryCompiler indicates whether or not by-data came
            from the `self`.
        **kwargs : dict
            Additional parameters to pass to the ``modin.core.dataframe.algebra.GroupByReduce.register``.

        Returns
        -------
        PandasQueryCompiler
            New QueryCompiler containing the result of groupby dictionary aggregation.
        """
        map_dict = {}
        reduce_dict = {}
        kwargs.setdefault(
            "default_to_pandas_func",
            lambda grp, *args, **kwargs: grp.agg(agg_func, *args, **kwargs),
        )

        rename_columns = any(
            not isinstance(fn, str) and isinstance(fn, Iterable)
            for fn in agg_func.values()
        )
        for col, col_funcs in agg_func.items():
            if not rename_columns:
                map_dict[col], reduce_dict[col], _ = GroupbyReduceImpl.get_impl(
                    col_funcs
                )
                continue

            if isinstance(col_funcs, str):
                col_funcs = [col_funcs]

            map_fns = []
            for i, fn in enumerate(col_funcs):
                if not isinstance(fn, str) and isinstance(fn, Iterable):
                    new_col_name, func = fn
                elif isinstance(fn, str):
                    new_col_name, func = fn, fn
                else:
                    raise TypeError

                map_fn, reduce_fn, _ = GroupbyReduceImpl.get_impl(func)

                map_fns.append((new_col_name, map_fn))
                reduced_col_name = (
                    (*col, new_col_name)
                    if isinstance(col, tuple)
                    else (col, new_col_name)
                )
                reduce_dict[reduced_col_name] = reduce_fn
            map_dict[col] = map_fns
        return GroupByReduce.register(map_dict, reduce_dict, **kwargs)(
            query_compiler=self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    def groupby_dtypes(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.groupby_agg(
            by=by,
            axis=axis,
            agg_func=lambda df: df.dtypes,
            # passing 'group_wise' will make the function be applied to the 'by' columns as well,
            # this is exactly what we want when 'as_index=False'
            how="axis_wise" if groupby_kwargs.get("as_index", True) else "group_wise",
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            groupby_kwargs=groupby_kwargs,
            drop=drop,
        )

    @_inherit_docstrings(BaseQueryCompiler.groupby_agg)
    def _groupby_shuffle(
        self,
        by,
        agg_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        how="axis_wise",
    ):
        if Engine.get() == "Python":
            raise NotImplementedError(
                "Reshuffling groupby is not implemented for python engine, see: "
                + "https://github.com/modin-project/modin/issues/5916"
            )

        # Defaulting to pandas in case of an empty frame as we can't process it properly.
        # Higher API level won't pass empty data here unless the frame has delayed
        # computations. FIXME: We apparently lose some laziness here (due to index access)
        # because of the inability to process empty groupby natively.
        if len(self.columns) == 0 or len(self.index) == 0:
            return super().groupby_agg(
                by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how, drop
            )

        if isinstance(by, type(self)) and drop:
            by = by.columns.tolist()

        if not isinstance(by, list):
            by = [by]

        is_all_labels = all(isinstance(col, (str, tuple)) for col in by)
        is_all_column_names = (
            all(col in self.columns for col in by) if is_all_labels else False
        )

        if not is_all_column_names:
            raise NotImplementedError(
                "Reshuffling groupby is only supported when grouping on a column(s) of the same frame. "
                + "https://github.com/modin-project/modin/issues/5926"
            )

        # So this check works only if we have dtypes cache materialized, otherwise the exception will be thrown
        # inside the kernel and so it will be uncatchable. TODO: figure out a better way to handle this.
        if self._modin_frame._dtypes is not None and any(
            dtype == "category" for dtype in self.dtypes[by].values
        ):
            raise NotImplementedError(
                "Reshuffling groupby is not yet supported when grouping on a categorical column. "
                + "https://github.com/modin-project/modin/issues/5925"
            )

        is_transform = how == "transform" or GroupBy.is_transformation_kernel(agg_func)

        if is_transform:
            # https://github.com/modin-project/modin/issues/5924
            ErrorMessage.missmatch_with_pandas(
                operation="reshuffling groupby",
                message="the order of rows may be shuffled for the result",
            )

        if isinstance(agg_func, dict):
            assert (
                how == "axis_wise"
            ), f"Only 'axis_wise' aggregation is supported with dictionary functions, got: {how}"

            subset = by + list(agg_func.keys())
            # extracting unique values; no we can't use np.unique here as it would
            # convert a list of tuples to a 2D matrix and so mess up the result
            subset = list(dict.fromkeys(subset))
            obj = self.getitem_column_array(subset)
        else:
            obj = self

        agg_func = functools.partial(
            GroupByDefault.get_aggregation_method(how), func=agg_func
        )

        result = obj._modin_frame.groupby(
            axis=axis,
            by=by,
            operator=lambda grp: agg_func(grp, *agg_args, **agg_kwargs),
            **groupby_kwargs,
        )
        result_qc = self.__constructor__(result)

        if not is_transform and not groupby_kwargs.get("as_index", True):
            return result_qc.reset_index(drop=True)

        return result_qc

    def groupby_corr(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        ErrorMessage.default_to_pandas("`GroupBy.corr`")
        # TODO(https://github.com/modin-project/modin/issues/1323) implement this.
        # Right now, using this class's groupby_agg method, even with how="group_wise",
        # produces a result with the wrong index, so default to pandas by using the
        # super class's groupby_agg method.
        return super().groupby_agg(
            by=by,
            agg_func="corr",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    def groupby_cov(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        ErrorMessage.default_to_pandas("`GroupBy.cov`")
        # TODO(https://github.com/modin-project/modin/issues/1322) implement this.
        # Right now, using this class's groupby_agg method, even with how="group_wise",
        # produces a result with the wrong index, so default to pandas by using the
        # super class's groupby_agg method.
        return super().groupby_agg(
            by=by,
            agg_func="cov",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    def groupby_agg(
        self,
        by,
        agg_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        how="axis_wise",
        drop=False,
        series_groupby=False,
    ):
        # Defaulting to pandas in case of an empty frame as we can't process it properly.
        # Higher API level won't pass empty data here unless the frame has delayed
        # computations. So we apparently lose some laziness here (due to index access)
        # because of the inability to process empty groupby natively.
        if len(self.columns) == 0 or len(self.index) == 0:
            return super().groupby_agg(
                by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how, drop
            )

        if ExperimentalGroupbyImpl.get():
            try:
                return self._groupby_shuffle(
                    by=by,
                    agg_func=agg_func,
                    axis=axis,
                    groupby_kwargs=groupby_kwargs,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                    how=how,
                )
            except NotImplementedError as e:
                ErrorMessage.warn(
                    f"Can't use experimental reshuffling groupby implementation because of: {e}"
                    + "\nFalling back to a full-axis implementation."
                )

        if isinstance(agg_func, dict) and GroupbyReduceImpl.has_impl_for(agg_func):
            return self._groupby_dict_reduce(
                by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )

        is_transform_method = how == "transform" or (
            isinstance(agg_func, str) and agg_func in transformation_kernels
        )

        original_agg_func = agg_func

        if isinstance(agg_func, dict):
            assert (
                how == "axis_wise"
            ), f"Only 'axis_wise' aggregation is supported with dictionary functions, got: {how}"
        else:
            agg_func = functools.partial(
                (
                    SeriesGroupByDefault if series_groupby else GroupByDefault
                ).get_aggregation_method(how),
                func=agg_func,
            )

        # since we're going to modify `groupby_kwargs` dict in a `groupby_agg_builder`,
        # we want to copy it to not propagate these changes into source dict, in case
        # of unsuccessful end of function
        groupby_kwargs = groupby_kwargs.copy()

        as_index = groupby_kwargs.get("as_index", True)
        by, internal_by = self._groupby_internal_columns(by, drop)

        broadcastable_by = [o._modin_frame for o in by if isinstance(o, type(self))]
        not_broadcastable_by = [o for o in by if not isinstance(o, type(self))]

        def groupby_agg_builder(df, by=None, drop=False, partition_idx=None):
            """
            Compute groupby aggregation for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            by : pandas.DataFrame, optional
                Broadcasted partition which contains `by` columns.
            drop : bool, default: False
                Indicates whether `by` partition came from the `self` frame.
            partition_idx : int, optional
                Positional partition index along groupby axis.

            Returns
            -------
            pandas.DataFrame
                DataFrame containing the result of groupby aggregation
                for this particular partition.
            """
            # Set `as_index` to True to track the metadata of the grouping object
            # It is used to make sure that between phases we are constructing the
            # right index and placing columns in the correct order.
            groupby_kwargs["as_index"] = True

            # We have to filter func-dict BEFORE inserting broadcasted 'by' columns
            # to avoid multiple aggregation results for 'by' cols in case they're
            # present in the func-dict:
            partition_agg_func = GroupByReduce.get_callable(agg_func, df)

            internal_by_cols = pandas.Index([])
            missed_by_cols = pandas.Index([])

            if by is not None:
                internal_by_df = by[internal_by]

                if isinstance(internal_by_df, pandas.Series):
                    internal_by_df = internal_by_df.to_frame()

                missed_by_cols = internal_by_df.columns.difference(df.columns)
                if len(missed_by_cols) > 0:
                    df = pandas.concat(
                        [df, internal_by_df[missed_by_cols]],
                        axis=1,
                        copy=False,
                    )

                internal_by_cols = internal_by_df.columns

                external_by = by.columns.difference(internal_by).unique()
                external_by_df = by[external_by].squeeze(axis=1)

                if isinstance(external_by_df, pandas.DataFrame):
                    external_by_cols = [o for _, o in external_by_df.items()]
                else:
                    external_by_cols = [external_by_df]

                by = internal_by_cols.tolist() + external_by_cols

            else:
                by = []

            by += not_broadcastable_by
            level = groupby_kwargs.get("level", None)
            if level is not None and not by:
                by = None
                by_length = len(level) if is_list_like(level) else 1
            else:
                by_length = len(by)

            def compute_groupby(df, drop=False, partition_idx=0):
                """Compute groupby aggregation for a single partition."""
                target_df = df.squeeze(axis=1) if series_groupby else df
                grouped_df = target_df.groupby(by=by, axis=axis, **groupby_kwargs)
                try:
                    result = partition_agg_func(grouped_df, *agg_args, **agg_kwargs)
                except DataError:
                    # This happens when the partition is filled with non-numeric data and a
                    # numeric operation is done. We need to build the index here to avoid
                    # issues with extracting the index.
                    result = pandas.DataFrame(index=grouped_df.size().index)
                if isinstance(result, pandas.Series):
                    result = result.to_frame(
                        result.name
                        if result.name is not None
                        else MODIN_UNNAMED_SERIES_LABEL
                    )

                selection = agg_func.keys() if isinstance(agg_func, dict) else None
                if selection is None:
                    # Some pandas built-in aggregation functions aggregate 'by' columns
                    # (for example 'apply', 'dtypes', maybe more...). Since we make sure
                    # that all of the 'by' columns are presented in every partition by
                    # inserting the missed ones, we will end up with all of the 'by'
                    # columns being aggregated in every partition. To avoid duplications
                    # in the result we drop all of the 'by' columns that were inserted
                    # in this partition AFTER handling 'as_index' parameter. The order
                    # is important for proper naming-conflicts handling.
                    misaggregated_cols = missed_by_cols.intersection(result.columns)
                else:
                    misaggregated_cols = []

                if not as_index:
                    GroupBy.handle_as_index_for_dataframe(
                        result,
                        internal_by_cols,
                        by_cols_dtypes=df[internal_by_cols].dtypes.values,
                        by_length=by_length,
                        selection=selection,
                        partition_idx=partition_idx,
                        drop=drop,
                        inplace=True,
                        method="transform" if is_transform_method else None,
                    )
                else:
                    new_index_names = tuple(
                        None
                        if isinstance(name, str)
                        and name.startswith(MODIN_UNNAMED_SERIES_LABEL)
                        else name
                        for name in result.index.names
                    )
                    result.index.names = new_index_names

                if len(misaggregated_cols) > 0:
                    result.drop(columns=misaggregated_cols, inplace=True)

                return result

            try:
                return compute_groupby(df, drop, partition_idx)
            except (ValueError, KeyError):
                # This will happen with Arrow buffer read-only errors. We don't want to copy
                # all the time, so this will try to fast-path the code first.
                return compute_groupby(df.copy(), drop, partition_idx)

        if isinstance(original_agg_func, dict):
            apply_indices = list(agg_func.keys())
        elif isinstance(original_agg_func, list):
            apply_indices = self.columns.difference(internal_by).tolist()
        else:
            apply_indices = None

        new_modin_frame = self._modin_frame.broadcast_apply_full_axis(
            axis=axis,
            func=lambda df, by=None, partition_idx=None: groupby_agg_builder(
                df, by, drop, partition_idx
            ),
            other=broadcastable_by,
            apply_indices=apply_indices,
            enumerate_partitions=True,
        )
        result = self.__constructor__(new_modin_frame)

        # that means that exception in `compute_groupby` was raised
        # in every partition, so we also should raise it
        # TODO: we should be able to drop this logic with pandas 2.0 as it removes `numeric_only=None`
        # parameter for groupby thus making the behavior of processing of non-numeric columns more
        # predictable (we can decide whether to raise an exception before actually executing groupby)
        if len(result.columns) == 0 and len(self.columns) != 0:
            # determening type of raised exception by applying `aggfunc`
            # to empty DataFrame
            try:
                pandas.DataFrame(index=[1], columns=[1]).agg(agg_func) if isinstance(
                    agg_func, dict
                ) else agg_func(
                    pandas.DataFrame(index=[1], columns=[1]).groupby(level=0),
                    **agg_kwargs,
                )
            except Exception as err:
                raise type(err)("No numeric types to aggregate.")

        return result

    # END Manual Partitioning methods

    def pivot(self, index, columns, values):
        from pandas.core.reshape.pivot import _convert_by

        def __convert_by(by):
            """Convert passed value to a list."""
            if isinstance(by, pandas.Index):
                by = list(by)
            by = _convert_by(by)
            if (
                len(by) > 0
                and (not is_list_like(by[0]) or isinstance(by[0], tuple))
                and not all([key in self.columns for key in by])
            ):
                by = [by]
            return by

        index, columns, values = map(__convert_by, [index, columns, values])
        is_custom_index = (
            len(index) == 1
            and is_list_like(index[0])
            and not isinstance(index[0], tuple)
        )

        if is_custom_index or len(index) == 0:
            to_reindex = columns
        else:
            to_reindex = index + columns

        if len(values) != 0:
            obj = self.getitem_column_array(to_reindex + values)
        else:
            obj = self

        if is_custom_index:
            obj.index = index

        reindexed = self.__constructor__(
            obj._modin_frame.apply_full_axis(
                1,
                lambda df: df.set_index(to_reindex, append=(len(to_reindex) == 1)),
                new_columns=obj.columns.drop(to_reindex),
            )
        )

        unstacked = reindexed.unstack(level=columns, fill_value=None)
        if len(reindexed.columns) == 1 and unstacked.columns.nlevels > 1:
            unstacked.columns = unstacked.columns.droplevel(0)

        return unstacked

    def pivot_table(
        self,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    ):
        ErrorMessage.missmatch_with_pandas(
            operation="pivot_table",
            message="Order of columns could be different from pandas",
        )

        from pandas.core.reshape.pivot import _convert_by

        def __convert_by(by):
            """Convert passed value to a list."""
            if isinstance(by, pandas.Index):
                return list(by)
            return _convert_by(by)

        index, columns, values = map(__convert_by, [index, columns, values])

        unique_keys = np.unique(index + columns)
        unique_values = np.unique(values)

        if len(values):
            to_group = self.getitem_column_array(unique_values)
        else:
            to_group = self.drop(columns=unique_keys)

        keys_columns = self.getitem_column_array(unique_keys)
        len_values = len(values)
        if len_values == 0:
            len_values = len(self.columns.drop(unique_keys))

        def applyier(df, other):  # pragma: no cover
            """
            Build pivot table for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            other : pandas.DataFrame
                Broadcasted partition that contains `value` columns
                of the self frame.

            Returns
            -------
            pandas.DataFrame
                Pivot table for this particular partition.
            """
            concated = pandas.concat([df, other], axis=1, copy=False)
            result = pandas.pivot_table(
                concated,
                index=index,
                values=values if len(values) > 0 else None,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=fill_value,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
            )

            # if only one value is specified, removing level that maps
            # columns from `values` to the actual values
            if len(index) > 0 and len_values == 1 and result.columns.nlevels > 1:
                result.columns = result.columns.droplevel(int(margins))

            # in that case Pandas transposes the result of `pivot_table`,
            # transposing it back to be consistent with column axis values along
            # different partitions
            if len(index) == 0 and len(columns) > 0:
                result = result.T

            return result

        result = self.__constructor__(
            to_group._modin_frame.broadcast_apply_full_axis(
                axis=0, func=applyier, other=keys_columns._modin_frame
            )
        )

        # transposing the result again, to be consistent with Pandas result
        if len(index) == 0 and len(columns) > 0:
            result = result.transpose()

        return result

    # Get_dummies
    def get_dummies(self, columns, **kwargs):
        # `columns` as None does not mean all columns, by default it means only
        # non-numeric columns.
        if columns is None:
            columns = [c for c in self.columns if not is_numeric_dtype(self.dtypes[c])]
            # If we aren't computing any dummies, there is no need for any
            # remote compute.
            if len(columns) == 0:
                return self.copy()
        elif not is_list_like(columns):
            columns = [columns]

        def map_fn(df):  # pragma: no cover
            cols_to_encode = df.columns.intersection(columns)
            return pandas.get_dummies(df, columns=cols_to_encode, **kwargs)

        # In some cases, we are mapping across all of the data. It is more
        # efficient if we are mapping over all of the data to do it this way
        # than it would be to reuse the code for specific columns.
        if len(columns) == len(self.columns):
            new_modin_frame = self._modin_frame.apply_full_axis(
                0, map_fn, new_index=self.index
            )
            untouched_frame = None
        else:
            new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
                col_labels=columns
            ).apply_full_axis(0, map_fn, new_index=self.index)
            untouched_frame = self.drop(columns=columns)
        # If we mapped over all the data we are done. If not, we need to
        # prepend the `new_modin_frame` with the raw data from the columns that were
        # not selected.
        if len(columns) != len(self.columns):
            new_modin_frame = untouched_frame._modin_frame.concat(
                1, [new_modin_frame], how="left", sort=False
            )
        return self.__constructor__(new_modin_frame)

    # END Get_dummies

    # Indexing
    def take_2d_positional(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.take_2d_labels_or_positional(
                row_positions=index, col_positions=columns
            )
        )

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            """
            Write `value` in a specified location in a single partition.

            Parameters
            ----------
            partition : pandas.DataFrame
                Partition of the self frame.
            row_internal_indices : list of ints
                Positional indices of rows in this particular partition
                to write `item` to.
            col_internal_indices : list of ints
                Positional indices of columns in this particular partition
                to write `item` to.
            item : 2D-array
                Value to write.

            Returns
            -------
            pandas.DataFrame
                Partition data with updated values.
            """
            partition = partition.copy()
            partition.iloc[row_internal_indices, col_internal_indices] = item
            return partition

        new_modin_frame = self._modin_frame.apply_select_indices(
            axis=None,
            func=iloc_mut,
            row_labels=row_numeric_index,
            col_labels=col_numeric_index,
            new_index=self.index,
            new_columns=self.columns,
            keep_remaining=True,
            item_to_distribute=broadcasted_items,
        )
        return self.__constructor__(new_modin_frame)

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        # Our algebra sort is only implemented for Engines that support virtual partitioning.
        if Engine.get() in ["Ray", "Dask", "Unidist"]:
            new_modin_frame = self._modin_frame.sort_by(
                0, columns, ascending=ascending, **kwargs
            )
            return self.__constructor__(new_modin_frame)
        ignore_index = kwargs.get("ignore_index", False)
        kwargs["ignore_index"] = False
        if not is_list_like(columns):
            columns = [columns]
        # Currently, sort_values will just reindex based on the sorted values.
        # TODO create a more efficient way to sort
        ErrorMessage.default_to_pandas("sort_values")
        broadcast_value_dict = {
            col: self.getitem_column_array([col]).to_pandas().squeeze(axis=1)
            for col in columns
        }
        # Clear index level names because they also appear in broadcast_value_dict
        orig_index_level_names = self.index.names
        tmp_index = self.index.copy()
        tmp_index.names = [None] * tmp_index.nlevels
        # Index may contain duplicates
        broadcast_values1 = pandas.DataFrame(broadcast_value_dict, index=tmp_index)
        # Index without duplicates
        broadcast_values2 = pandas.DataFrame(broadcast_value_dict)
        broadcast_values2 = broadcast_values2.reset_index(drop=True)
        # Index may contain duplicates
        new_index1 = broadcast_values1.sort_values(
            by=columns, axis=0, ascending=ascending, **kwargs
        ).index
        # Index without duplicates
        new_index2 = broadcast_values2.sort_values(
            by=columns, axis=0, ascending=ascending, **kwargs
        ).index

        result = self.reset_index(drop=True).reindex(axis=0, labels=new_index2)
        if ignore_index:
            result = result.reset_index(drop=True)
        else:
            result.index = new_index1
            result.index.names = orig_index_level_names
        return result

    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        if not is_list_like(rows):
            rows = [rows]
        ErrorMessage.default_to_pandas("sort_values")
        broadcast_value_list = [
            self.getitem_row_array([row]).to_pandas() for row in rows
        ]
        index_builder = list(zip(broadcast_value_list, rows))
        broadcast_values = pandas.concat(
            [row for row, idx in index_builder], copy=False
        )
        broadcast_values.columns = self.columns
        new_columns = broadcast_values.sort_values(
            by=rows, axis=1, ascending=ascending, **kwargs
        ).columns
        return self.reindex(axis=1, labels=new_columns)

    # Cat operations
    def cat_codes(self):
        def func(df) -> np.ndarray:
            # `df` is supposed to be consisted of multiple partitions,
            # which should be concatenated before applying a function.
            # `pd.concat` doesn't preserve categorical dtype
            # if the dfs have categorical columns
            # so we intentionaly restore the right dtype.
            # TODO: revert the change when https://github.com/pandas-dev/pandas/issues/51362 is fixed.
            ser = df.iloc[:, 0]
            if ser.dtype != "category":
                ser = ser.astype("category", copy=False)
            return ser.cat.codes.to_frame(name=MODIN_UNNAMED_SERIES_LABEL)

        res = self._modin_frame.fold(
            axis=0, func=func, new_columns=[MODIN_UNNAMED_SERIES_LABEL]
        )
        return self.__constructor__(res, shape_hint="column")

    # END Cat operations

    def compare(self, other, **kwargs):
        return self.__constructor__(
            self._modin_frame.broadcast_apply_full_axis(
                0,
                lambda left, right: pandas.DataFrame.compare(
                    left, other=right, **kwargs
                ),
                other._modin_frame,
            )
        )
