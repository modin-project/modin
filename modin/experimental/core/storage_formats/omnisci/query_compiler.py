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
Module contains ``DFAlgQueryCompiler`` class.

``DFAlgQueryCompiler`` is used for lazy DataFrame Algebra based engine.
"""

from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
    _set_axis as default_axis_setter,
    _get_axis as default_axis_getter,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.utils import _inherit_docstrings
from modin.error_message import ErrorMessage
import pandas

from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_list_like
from functools import wraps


def is_inoperable(value):
    """
    Check if value cannot be processed by OmniSci engine.

    Parameters
    ----------
    value : any
        A value to check.

    Returns
    -------
    bool
    """
    if isinstance(value, (tuple, list)):
        result = False
        for val in value:
            result = result or is_inoperable(val)
        return result
    elif isinstance(value, dict):
        return is_inoperable(list(value.values()))
    else:
        value = getattr(value, "_query_compiler", value)
        if hasattr(value, "_modin_frame"):
            return value._modin_frame._has_unsupported_data
    return False


def build_method_wrapper(name, method):
    """
    Build method wrapper to handle inoperable data types.

    Wrapper calls the original method if all its arguments can be processed
    by OmniSci engine and fallback to parent's method otherwise.

    Parameters
    ----------
    name : str
        Parent's method name to fallback to.
    method : callable
        A method to wrap.

    Returns
    -------
    callable
    """

    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        # If the method wasn't found in the parent query compiler that means,
        # that we're calling one that is OmniSci-specific, if we intend
        # to fallback to pandas on 'NotImplementedError' then the call of this
        # private method is caused by some public QC method, so we catch
        # the exception here and do fallback properly
        default_method = getattr(super(type(self), self), name, None)
        if is_inoperable([self, args, kwargs]):
            if default_method is None:
                raise NotImplementedError("Frame contains data of unsupported types.")
            return default_method(*args, **kwargs)
        try:
            return method(self, *args, **kwargs)
        # Defaulting to pandas if `NotImplementedError` was arisen
        except NotImplementedError as e:
            if default_method is None:
                raise e
            ErrorMessage.default_to_pandas(message=str(e))
            return default_method(*args, **kwargs)

    return method_wrapper


def bind_wrappers(cls):
    """
    Wrap class methods.

    Decorator allows to fallback to the parent query compiler methods when unsupported
    data types are used in a frame.

    Returns
    -------
    class
    """
    exclude = set(
        [
            "__init__",
            "to_pandas",
            "from_pandas",
            "from_arrow",
            "default_to_pandas",
            "_get_index",
            "_set_index",
            "_get_columns",
            "_set_columns",
        ]
    )
    for name, method in cls.__dict__.items():
        if name in exclude:
            continue

        if callable(method):
            setattr(
                cls,
                name,
                build_method_wrapper(name, method),
            )

    return cls


@bind_wrappers
@_inherit_docstrings(BaseQueryCompiler)
class DFAlgQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for the OmniSci storage format.

    This class doesn't perform much processing and mostly forwards calls to
    :py:class:`~modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe.OmnisciOnNativeDataframe`
    for lazy execution trees build.

    Parameters
    ----------
    frame : OmnisciOnNativeDataframe
        Modin Frame to query with the compiled queries.
    shape_hint : {"row", "column", None}, default: None
        Shape hint for frames known to be a column or a row, otherwise None.

    Attributes
    ----------
    _modin_frame : OmnisciOnNativeDataframe
        Modin Frame to query with the compiled queries.
    _shape_hint : {"row", "column", None}
        Shape hint for frames known to be a column or a row, otherwise None.
    """

    lazy_execution = True

    def __init__(self, frame, shape_hint=None):
        assert frame is not None
        self._modin_frame = frame
        if shape_hint is None and len(self._modin_frame.columns) == 1:
            shape_hint = "column"
        self._shape_hint = shape_hint

    def finalize(self):
        # TODO: implement this for OmniSci storage format
        raise NotImplementedError()

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        if len(df.columns) == 1:
            shape_hint = "column"
        elif len(df) == 1:
            shape_hint = "row"
        else:
            shape_hint = None
        return cls(data_cls.from_pandas(df), shape_hint=shape_hint)

    @classmethod
    def from_arrow(cls, at, data_cls):
        if len(at.columns) == 1:
            shape_hint = "column"
        elif len(at) == 1:
            shape_hint = "row"
        else:
            shape_hint = None
        return cls(data_cls.from_arrow(at), shape_hint=shape_hint)

    # Dataframe exchange protocol

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        return self._modin_frame.__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df, data_cls):
        return cls(data_cls.from_dataframe(df))

    # END Dataframe exchange protocol

    default_to_pandas = PandasQueryCompiler.default_to_pandas

    def copy(self):
        return self.__constructor__(self._modin_frame, self._shape_hint)

    def getitem_column_array(self, key, numeric=False):
        shape_hint = "column" if len(key) == 1 else None
        if numeric:
            new_modin_frame = self._modin_frame.mask(col_positions=key)
        else:
            new_modin_frame = self._modin_frame.mask(col_labels=key)
        return self.__constructor__(new_modin_frame, shape_hint)

    def getitem_array(self, key):
        if isinstance(key, type(self)):
            new_modin_frame = self._modin_frame.filter(key._modin_frame)
            return self.__constructor__(new_modin_frame, self._shape_hint)

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
        left_on = kwargs.get("left_on", None)
        right_on = kwargs.get("right_on", None)
        left_index = kwargs.get("left_index", False)
        right_index = kwargs.get("right_index", False)
        """Only non-index joins with explicit 'on' are supported"""
        if left_index is False and right_index is False:
            if left_on is None and right_on is None:
                if on is None:
                    on = [c for c in self.columns if c in right.columns]
                left_on = on
                right_on = on

            if not isinstance(left_on, list):
                left_on = [left_on]
            if not isinstance(right_on, list):
                right_on = [right_on]

            how = kwargs.get("how", "inner")
            sort = kwargs.get("sort", False)
            suffixes = kwargs.get("suffixes", None)
            return self.__constructor__(
                self._modin_frame.join(
                    right._modin_frame,
                    how=how,
                    left_on=left_on,
                    right_on=right_on,
                    sort=sort,
                    suffixes=suffixes,
                )
            )
        else:
            return self.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    def view(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.mask(row_positions=index, col_positions=columns)
        )

    def groupby_size(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        # Grouping on empty frame or on index level.
        if len(self.columns) == 0:
            raise NotImplementedError(
                "Grouping on empty frame or on index level is not yet implemented."
            )

        groupby_kwargs = groupby_kwargs.copy()
        as_index = groupby_kwargs.get("as_index", True)
        # Setting 'as_index' to True to avoid 'by' and 'agg' columns naming conflict
        groupby_kwargs["as_index"] = True
        new_frame = self._modin_frame.groupby_agg(
            by,
            axis,
            {self._modin_frame.columns[0]: "size"},
            groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )
        if as_index:
            shape_hint = "column"
            new_frame = new_frame._set_columns(["__reduced__"])
        else:
            shape_hint = None
            new_frame = new_frame._set_columns(["size"]).reset_index(drop=False)
        return self.__constructor__(new_frame, shape_hint=shape_hint)

    def groupby_sum(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        new_frame = self._modin_frame.groupby_agg(
            by,
            axis,
            "sum",
            groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )
        return self.__constructor__(new_frame)

    def groupby_count(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        new_frame = self._modin_frame.groupby_agg(
            by,
            axis,
            "count",
            groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )
        return self.__constructor__(new_frame)

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
    ):
        # TODO: handle `drop` args
        if callable(agg_func):
            raise NotImplementedError(
                "Python callable is not a valid aggregation function for OmniSci storage format."
            )
        if how != "axis_wise":
            raise NotImplementedError(
                f"'{how}' type of groupby-aggregation functions is not supported for OmniSci storage format."
            )

        new_frame = self._modin_frame.groupby_agg(
            by,
            axis,
            agg_func,
            groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )
        return self.__constructor__(new_frame)

    def count(self, **kwargs):
        return self._agg("count", **kwargs)

    def max(self, **kwargs):
        return self._agg("max", **kwargs)

    def min(self, **kwargs):
        return self._agg("min", **kwargs)

    def sum(self, **kwargs):
        min_count = kwargs.pop("min_count")
        if min_count != 0:
            raise NotImplementedError(
                f"OmniSci's sum does not support such set of parameters: min_count={min_count}."
            )
        return self._agg("sum", **kwargs)

    def mean(self, **kwargs):
        return self._agg("mean", **kwargs)

    def nunique(self, axis=0, dropna=True):
        if axis != 0 or not dropna:
            raise NotImplementedError(
                f"OmniSci's nunique does not support such set of parameters: axis={axis}, dropna={dropna}."
            )
        return self._agg("nunique")

    def _agg(self, agg, axis=0, level=None, **kwargs):
        """
        Perform specified aggregation along rows/columns.

        Parameters
        ----------
        agg : str
            Name of the aggregation function to perform.
        axis : {0, 1}, default: 0
            Axis to perform aggregation along. 0 is to apply function against each column,
            all the columns will be reduced into a single scalar. 1 is to aggregate
            across rows.
            *Note:* OmniSci storage format supports aggregation for 0 axis only, aggregation
            along rows will be defaulted to pandas.
        level : None, default: None
            Serves the compatibility purpose, always have to be None.
        **kwargs : dict
            Additional parameters to pass to the aggregation function.

        Returns
        -------
        DFAlgQueryCompiler
            New single-column (``axis=1``) or single-row (``axis=0``) query compiler containing
            the result of aggregation.
        """
        if level is not None or axis != 0:
            raise NotImplementedError(
                "OmniSci's aggregation functions does not support 'level' and 'axis' parameters."
            )

        # TODO: Do filtering on numeric columns if `numeric_only=True`
        if not kwargs.get("skipna", True) or kwargs.get("numeric_only"):
            raise NotImplementedError(
                "OmniSci's aggregation functions does not support 'skipna' and 'numeric_only' parameters."
            )
        # Processed above, so can be omitted
        kwargs.pop("skipna", None)
        kwargs.pop("numeric_only", None)

        new_frame = self._modin_frame.agg(agg)
        new_frame = new_frame._set_index(
            pandas.Index.__new__(pandas.Index, data=["__reduced__"], dtype="O")
        )
        return self.__constructor__(new_frame, shape_hint="row")

    def _get_index(self):
        """
        Return frame's index.

        Returns
        -------
        pandas.Index
        """
        if self._modin_frame._has_unsupported_data:
            return default_axis_getter(0)(self)
        return self._modin_frame.index

    def _set_index(self, index):
        """
        Set new index.

        Parameters
        ----------
        index : pandas.Index
            A new index.
        """
        if self._modin_frame._has_unsupported_data:
            default_axis_setter(0)(self, index)
        else:
            default_axis_setter(0)(self, index)
            # NotImplementedError: OmnisciOnNativeDataframe._set_index is not yet suported
            # self._modin_frame.index = index

    def _get_columns(self):
        """
        Return frame's columns.

        Returns
        -------
        pandas.Index
        """
        if self._modin_frame._has_unsupported_data:
            return default_axis_getter(1)(self)
        return self._modin_frame.columns

    def _set_columns(self, columns):
        """
        Set new columns.

        Parameters
        ----------
        columns : list-like
            New columns.
        """
        if self._modin_frame._has_unsupported_data:
            default_axis_setter(1)(self, columns)
        else:
            self._modin_frame = self._modin_frame._set_columns(columns)

    def fillna(
        self,
        squeeze_self=False,
        squeeze_value=False,
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

        new_modin_frame = self._modin_frame.concat(
            axis, other_modin_frames, join=join, sort=sort, ignore_index=ignore_index
        )
        return self.__constructor__(new_modin_frame)

    def drop(self, index=None, columns=None):
        assert index is None, "Only column drop is supported"
        return self.__constructor__(
            self._modin_frame.mask(
                row_labels=index, col_labels=self.columns.drop(columns)
            )
        )

    def dropna(self, axis=0, how="any", thresh=None, subset=None):
        if thresh is not None or axis != 0:
            raise NotImplementedError(
                "OmniSci's dropna does not support 'thresh' and 'axis' parameters."
            )

        if subset is None:
            subset = self.columns
        return self.__constructor__(
            self._modin_frame.dropna(subset=subset, how=how),
            shape_hint=self._shape_hint,
        )

    def dt_year(self):
        return self.__constructor__(
            self._modin_frame.dt_extract("year"), self._shape_hint
        )

    def dt_month(self):
        return self.__constructor__(
            self._modin_frame.dt_extract("month"), self._shape_hint
        )

    def dt_day(self):
        return self.__constructor__(
            self._modin_frame.dt_extract("day"), self._shape_hint
        )

    def dt_hour(self):
        return self.__constructor__(
            self._modin_frame.dt_extract("hour"), self._shape_hint
        )

    def _bin_op(self, other, op_name, **kwargs):
        """
        Perform a binary operation on a frame.

        Parameters
        ----------
        other : any
            The second operand.
        op_name : str
            Operation name.
        **kwargs : dict
            Keyword args.

        Returns
        -------
        DFAlgQueryCompiler
            A new query compiler.
        """
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

    def mod(self, other, **kwargs):
        return self._bin_op(other, "mod", **kwargs)

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

    def __and__(self, other, **kwargs):
        return self._bin_op(other, "and", **kwargs)

    def __or__(self, other, **kwargs):
        return self._bin_op(other, "or", **kwargs)

    def reset_index(self, **kwargs):
        level = kwargs.get("level", None)
        if level is not None:
            raise NotImplementedError(
                "OmniSci's reset_index does not support 'level' parameter."
            )

        drop = kwargs.get("drop", False)
        shape_hint = self._shape_hint if drop else None

        return self.__constructor__(
            self._modin_frame.reset_index(drop), shape_hint=shape_hint
        )

    def astype(self, col_dtypes, **kwargs):
        return self.__constructor__(
            self._modin_frame.astype(col_dtypes), self._shape_hint
        )

    def setitem(self, axis, key, value):
        if axis == 1 or not isinstance(value, type(self)):
            raise NotImplementedError(
                f"OmniSci's setitem does not support such set of parameters: axis={axis}, value={value}."
            )
        return self._setitem(axis, key, value)

    _setitem = PandasQueryCompiler._setitem

    def insert(self, loc, column, value):
        if isinstance(value, type(self)):
            value.columns = [column]
            return self.insert_item(axis=1, loc=loc, value=value)

        if is_list_like(value):
            raise NotImplementedError(
                "OmniSci's insert does not support list-like values."
            )

        return self.__constructor__(self._modin_frame.insert(loc, column, value))

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        ignore_index = kwargs.get("ignore_index", False)
        na_position = kwargs.get("na_position", "last")
        return self.__constructor__(
            self._modin_frame.sort_rows(columns, ascending, ignore_index, na_position),
            self._shape_hint,
        )

    def columnarize(self):
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
        if self._shape_hint is not None:
            return True
        return len(self.columns) == 1 or len(self.index) == 1

    def cat_codes(self):
        return self.__constructor__(self._modin_frame.cat_codes(), self._shape_hint)

    def has_multiindex(self, axis=0):
        if axis == 0:
            return self._modin_frame.has_multiindex()
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    def get_index_name(self, axis=0):
        return self.columns.name if axis else self._modin_frame.get_index_name()

    def set_index_name(self, name, axis=0):
        if axis == 0:
            self._modin_frame = self._modin_frame.set_index_name(name)
        else:
            self.columns.name = name

    def get_index_names(self, axis=0):
        return self.columns.names if axis else self._modin_frame.get_index_names()

    def set_index_names(self, names=None, axis=0):
        if axis == 0:
            self._modin_frame = self._modin_frame.set_index_names(names)
        else:
            self.columns.names = names

    def free(self):
        return

    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    @property
    def dtypes(self):
        return self._modin_frame.dtypes
