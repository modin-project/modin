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
Module contains ``SmallQueryCompiler`` class.

``SmallQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for small data and empty ``PandasDataFrame``.
"""

import pandas
from pandas.core.indexes.api import ensure_index_from_sequences
from typing import List, Hashable

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
from modin.utils import (
    _inherit_docstrings,
    try_cast_to_pandas,
)


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


@_inherit_docstrings(BaseQueryCompiler)
class SmallQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for the pandas storage format.

    This class translates common query compiler API to default all methods
    to pandas.

    Parameters
    ----------
    modin_frame : pandas.DataFrame
        Modin Frame to query with the compiled queries.
    """

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame
        print("THERES SOMETHING HERE")

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        # type(self) might not work
        # SmallQueryCompiler and PandasQueryCompiler are not the same
        args = (a.to_pandas() if isinstance(a, type(self)) else a for a in args)
        kwargs = {
            k: v.to_pandas if isinstance(v, type(self)) else v
            for k, v in kwargs.items()
        }

        result = pandas_op(self._modin_frame, *args, **kwargs)
        if isinstance(result, pandas.Series):
            if result.name is None:
                result.name = "__reduced__"
            result = result.to_frame()

        return result
        # if isinstance(result, pandas.DataFrame):
        #     return self.from_pandas(result, type(self._modin_frame))
        # else:
        #     return result

    def _register_default_pandas(func):
        def caller(dataframe, *args, **kwargs):
            print(func.__name__)
            args = try_cast_to_pandas(args)
            kwargs = try_cast_to_pandas(kwargs)
            return func(dataframe, *args, **kwargs)

        return caller

    abs = _register_default_pandas(pandas.DataFrame.abs)
    add = _register_default_pandas(pandas.DataFrame.add)
    all = _register_default_pandas(pandas.DataFrame.all)
    any = _register_default_pandas(pandas.DataFrame.any)
    applymap = _register_default_pandas(pandas.DataFrame.applymap)
    astype = _register_default_pandas(pandas.DataFrame.astype)
    combine = _register_default_pandas(pandas.DataFrame.combine)

    def to_pandas(self):
        return self._modin_frame

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return

    def free(self):
        return

    def finalize(self):
        self._modin_frame.finalize()

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

    # def __getattribute__(self, item):
    #     print("Getting attribute:", item)
    #     return super()._modin_frame.__getattribute__(item)

    # END Index, columns, and dtypes objects

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._modin_frame.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._modin_frame.add_suffix(suffix, axis))

    # END Metadata modification methods

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: df.reindex(labels=labels, axis=axis, **kwargs),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def columnarize(self):
        if len(self._modin_frame.columns) != 1 or (
            len(self._modin_frame.index) == 1
            and self._modin_frame.index[0] == MODIN_UNNAMED_SERIES_LABEL
        ):
            return self._modin_frame.transpose()
        return self._modin_frame

    def reset_index(self, **kwargs):
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
        else:
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
                pandas.RangeIndex(len(new_self.index))
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
