import numpy as np
from numpy import nan
import pandas
from pandas.api.types import is_scalar
from pandas.compat import cPickle as pkl, numpy as numpy_compat, string_types, to_str
from pandas.core.common import count_not_none, _get_rename_function, _pipe
from pandas.core.dtypes.common import (
    is_list_like,
    is_dict_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
    is_dtype_equal,
    is_object_dtype,
)
from pandas.core.indexing import convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg
import re
import warnings

from modin.error_message import ErrorMessage

# Similar to pandas, sentinel value to use as kwarg in place of None when None has
# special meaning and needs to be distinguished from a user explicitly passing None.
sentinel = object()


class BasePandasDataset(object):
    """This object is the base for most of the common code that exists in
        DataFrame/Series. Since both objects share the same underlying representation,
        and the algorithms are the same, we use this object to define the general
        behavior of those objects and then use those objects to define the output type.
    """

    # Siblings are other objects that share the same query compiler. We use this list
    # to update inplace when there is a shallow copy.
    _siblings = []

    def _add_sibling(self, sibling):
        sibling._siblings = self._siblings + [self]
        self._siblings += [sibling]
        for sib in self._siblings:
            sib._siblings += [sibling]

    def _build_repr_df(self, num_rows, num_cols):
        # Add one here so that pandas automatically adds the dots
        # It turns out to be faster to extract 2 extra rows and columns than to
        # build the dots ourselves.
        num_rows_for_head = num_rows // 2 + 1
        num_cols_for_front = num_cols // 2 + 1

        if self.empty:
            return self._query_compiler.to_pandas()
        if len(self.index) <= num_rows:
            head = self._query_compiler
            tail = None
        else:
            head = self._query_compiler.head(num_rows_for_head)
            tail = self._query_compiler.tail(num_rows_for_head)

        if not hasattr(self, "columns") or len(self.columns) <= num_cols:
            head_front = head.to_pandas()
            # Creating these empty to make the concat logic simpler
            head_back = pandas.DataFrame()
            tail_back = pandas.DataFrame()

            if tail is not None:
                tail_front = tail.to_pandas()
            else:
                tail_front = pandas.DataFrame()
        else:
            head_front = head.front(num_cols_for_front).to_pandas()
            head_back = head.back(num_cols_for_front).to_pandas()

            if tail is not None:
                tail_front = tail.front(num_cols_for_front).to_pandas()
                tail_back = tail.back(num_cols_for_front).to_pandas()
            else:
                tail_front = tail_back = pandas.DataFrame()

        head_for_repr = pandas.concat([head_front, head_back], axis=1)
        tail_for_repr = pandas.concat([tail_front, tail_back], axis=1)

        return pandas.concat([head_for_repr, tail_for_repr])

    def _update_inplace(self, new_query_compiler):
        """Updates the current DataFrame inplace.

        Args:
            new_query_compiler: The new QueryCompiler to use to manage the data
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        for sib in self._siblings:
            sib._query_compiler = new_query_compiler
        old_query_compiler.free()

    def _validate_other(
        self,
        other,
        axis,
        numeric_only=False,
        numeric_or_time_only=False,
        numeric_or_object_only=False,
        comparison_dtypes_only=False,
    ):
        """Helper method to check validity of other in inter-df operations"""
        # We skip dtype checking if the other is a scalar.
        if is_scalar(other):
            return other
        axis = self._get_axis_number(axis) if axis is not None else 1
        result = other
        if isinstance(other, BasePandasDataset):
            return other._query_compiler
        elif is_list_like(other):
            if axis == 0:
                if len(other) != len(self._query_compiler.index):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self._query_compiler.index), len(other))
                    )
            else:
                if len(other) != len(self._query_compiler.columns):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(
                            len(self._query_compiler.columns), len(other)
                        )
                    )
            if hasattr(other, "dtype"):
                other_dtypes = [other.dtype] * len(other)
            else:
                other_dtypes = [type(x) for x in other]
        else:
            other_dtypes = [
                type(other)
                for _ in range(
                    len(self._query_compiler.index)
                    if axis
                    else len(self._query_compiler.columns)
                )
            ]
        # Do dtype checking.
        if numeric_only:
            if not all(
                is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype)
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError("Cannot do operation on non-numeric dtypes")
        elif numeric_or_object_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (is_object_dtype(self_dtype) and is_object_dtype(other_dtype))
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError("Cannot do operation non-numeric dtypes")
        elif comparison_dtypes_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (
                    is_datetime_or_timedelta_dtype(self_dtype)
                    and is_datetime_or_timedelta_dtype(other_dtype)
                )
                or is_dtype_equal(self_dtype, other_dtype)
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError(
                    "Cannot do operation non-numeric objects with numeric objects"
                )
        elif numeric_or_time_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (
                    is_datetime_or_timedelta_dtype(self_dtype)
                    and is_datetime_or_timedelta_dtype(other_dtype)
                )
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError(
                    "Cannot do operation non-numeric objects with numeric objects"
                )
        return result

    def _binary_op(self, op, other, axis=None, **kwargs):
        axis = self._get_axis_number(axis) if axis is not None else 1
        if kwargs.get("level", None) is not None:
            if isinstance(other, BasePandasDataset):
                other = other._to_pandas()
            return self._default_to_pandas(
                getattr(getattr(pandas, self.__name__), op), other, axis=axis, **kwargs
            )
        other = self._validate_other(other, axis, numeric_or_object_only=True)
        new_query_compiler = self._query_compiler.binary_op(
            op, other=other, axis=axis, **kwargs
        )
        return self._create_or_update_from_compiler(new_query_compiler)

    def _default_to_pandas(self, op, *args, **kwargs):
        """Helper method to use default pandas function"""
        empty_self_str = "" if not self.empty else " for empty DataFrame"
        ErrorMessage.default_to_pandas(
            "`{}.{}`{}".format(
                self.__name__,
                op if isinstance(op, str) else op.__name__,
                empty_self_str,
            )
        )
        if callable(op):
            result = op(self._to_pandas(), *args, **kwargs)
        elif isinstance(op, str):
            # The inner `getattr` is ensuring that we are treating this object (whether
            # it is a DataFrame, Series, etc.) as a pandas object. The outer `getattr`
            # will get the operation (`op`) from the pandas version of the class and run
            # it on the object after we have converted it to pandas.
            result = getattr(getattr(pandas, self.__name__), op)(
                self._to_pandas(), *args, **kwargs
            )
        # SparseDataFrames cannot be serialize by arrow and cause problems for Modin.
        # For now we will use pandas.
        if isinstance(result, type(self)) and not isinstance(
            result, (pandas.SparseDataFrame, pandas.SparseSeries)
        ):
            return self._create_or_update_from_compiler(
                result, inplace=kwargs.get("inplace", False)
            )
        elif isinstance(result, pandas.DataFrame):
            from .dataframe import DataFrame

            return DataFrame(result)
        elif isinstance(result, pandas.Series):
            from .series import Series

            return Series(result)
        else:
            try:
                if (
                    isinstance(result, (list, tuple))
                    and len(result) == 2
                    and isinstance(result[0], pandas.DataFrame)
                ):
                    # Some operations split the DataFrame into two (e.g. align). We need to wrap
                    # both of the returned results
                    if isinstance(result[1], pandas.DataFrame):
                        second = self.__constructor__(result[1])
                    else:
                        second = result[1]
                    return self.__constructor__(result[0]), second
                else:
                    return result
            except TypeError:
                return result

    def _get_axis_number(self, axis):
        return (
            getattr(pandas, self.__name__)()._get_axis_number(axis)
            if axis is not None
            else 0
        )

    def __constructor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def abs(self):
        """Apply an absolute value function to all numeric columns.

        Returns:
            A new DataFrame with the applied absolute value.
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.abs())

    def _set_index(self, new_index):
        """Set the index for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._query_compiler.index = new_index

    def _get_index(self):
        """Get the index for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.index

    index = property(_get_index, _set_index)

    def add(self, other, axis="columns", level=None, fill_value=None):
        """Add this DataFrame to another or a scalar/list.

        Args:
            other: What to add this this DataFrame.
            axis: The axis to apply addition over. Only applicaable to Series
                or list 'other'.
            level: A level in the multilevel axis to add over.
            fill_value: The value to fill NaN.

        Returns:
            A new DataFrame with the applied addition.
        """
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    def agg(self, func, axis=0, *args, **kwargs):
        return self.aggregate(func, axis=axis, *args, **kwargs)

    def aggregate(self, func, axis=0, *args, **kwargs):
        axis = self._get_axis_number(axis)
        result = None

        if axis == 0:
            try:
                result = self._aggregate(func, _axis=axis, *args, **kwargs)
            except TypeError:
                pass
        if result is None:
            kwargs.pop("is_transform", None)
            return self.apply(func, axis=axis, args=args, **kwargs)
        return result

    def _aggregate(self, arg, *args, **kwargs):
        _axis = kwargs.pop("_axis", 0)
        kwargs.pop("_level", None)

        if isinstance(arg, string_types):
            kwargs.pop("is_transform", None)
            return self._string_function(arg, *args, **kwargs)

        # Dictionaries have complex behavior because they can be renamed here.
        elif isinstance(arg, dict):
            return self._default_to_pandas("agg", arg, *args, **kwargs)
        elif is_list_like(arg) or callable(arg):
            kwargs.pop("is_transform", None)
            return self.apply(arg, axis=_axis, args=args, **kwargs)
        else:
            raise TypeError("type {} is not callable".format(type(arg)))

    def _string_function(self, func, *args, **kwargs):
        assert isinstance(func, string_types)
        f = getattr(self, func, None)
        if f is not None:
            if callable(f):
                return f(*args, **kwargs)
            assert len(args) == 0
            assert (
                len([kwarg for kwarg in kwargs if kwarg not in ["axis", "_level"]]) == 0
            )
            return f
        f = getattr(np, func, None)
        if f is not None:
            return self._default_to_pandas("agg", func, *args, **kwargs)
        raise ValueError("{} is an unknown string function".format(func))

    def _get_dtypes(self):
        if hasattr(self, "dtype"):
            return [self.dtype]
        else:
            return list(self.dtypes)

    def align(
        self,
        other,
        join="outer",
        axis=None,
        level=None,
        copy=True,
        fill_value=None,
        method=None,
        limit=None,
        fill_axis=0,
        broadcast_axis=None,
    ):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas(
            "align",
            other,
            join=join,
            axis=axis,
            level=level,
            copy=copy,
            fill_value=fill_value,
            method=method,
            limit=limit,
            fill_axis=fill_axis,
            broadcast_axis=broadcast_axis,
        )

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Return whether all elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies df.all(axis=1)
                to the transpose of df.
        """
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            self.__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool]]
                return data_for_compute.all(
                    axis=axis, bool_only=False, skipna=skipna, level=level, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            result = self._reduce_dimension(
                self._query_compiler.all(
                    axis=0, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
            if isinstance(result, BasePandasDataset):
                return result.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            return result

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Return whether any elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies on the column partitions,
                otherwise operates on row partitions
        """
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            self.__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool]]
                return data_for_compute.all(
                    axis=axis, bool_only=None, skipna=skipna, level=level, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            result = self._reduce_dimension(
                self._query_compiler.any(
                    axis=0, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
            if isinstance(result, BasePandasDataset):
                return result.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            return result

    def apply(
        self,
        func,
        axis=0,
        broadcast=None,
        raw=False,
        reduce=None,
        result_type=None,
        convert_dtype=True,
        args=(),
        **kwds
    ):
        """Apply a function along input axis of DataFrame.

        Args:
            func: The function to apply
            axis: The axis over which to apply the func.
            broadcast: Whether or not to broadcast.
            raw: Whether or not to convert to a Series.
            reduce: Whether or not to try to apply reduction procedures.

        Returns:
            Series or DataFrame, depending on func.
        """
        axis = self._get_axis_number(axis)
        ErrorMessage.non_verified_udf()
        if isinstance(func, string_types):
            if axis == 1:
                kwds["axis"] = axis
            result = self._string_function(func, *args, **kwds)
            # Sometimes we can return a scalar here
            if isinstance(result, BasePandasDataset):
                return result._query_compiler
            return result
        elif isinstance(func, dict):
            if axis == 1:
                raise TypeError(
                    "(\"'dict' object is not callable\", "
                    "'occurred at index {0}'".format(self.index[0])
                )
            if len(self.columns) != len(set(self.columns)):
                warnings.warn(
                    "duplicate column names not supported with apply().",
                    FutureWarning,
                    stacklevel=2,
                )
        elif not callable(func) and not is_list_like(func):
            raise TypeError("{} object is not callable".format(type(func)))
        query_compiler = self._query_compiler.apply(func, axis, *args, **kwds)
        return query_compiler

    def as_blocks(self, copy=True):
        return self._default_to_pandas("as_blocks", copy=copy)

    def as_matrix(self, columns=None):
        """Convert the frame to its Numpy-array representation.

        Args:
            columns: If None, return all columns, otherwise,
                returns specified columns.

        Returns:
            values: ndarray
        """
        if columns is None:
            return self.to_numpy()
        return self.__getitem__(columns).to_numpy()

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        return self._default_to_pandas(
            "asfreq",
            freq,
            method=method,
            how=how,
            normalize=normalize,
            fill_value=fill_value,
        )

    def asof(self, where, subset=None):
        return self._default_to_pandas("asof", where, subset=subset)

    def astype(self, dtype, copy=True, errors="raise", **kwargs):
        col_dtypes = {}
        if isinstance(dtype, dict):
            if (
                not set(dtype.keys()).issubset(set(self._query_compiler.columns))
                and errors == "raise"
            ):
                raise KeyError(
                    "Only a column name can be used for the key in"
                    "a dtype mappings argument."
                )
            col_dtypes = dtype
        else:
            for column in self._query_compiler.columns:
                col_dtypes[column] = dtype

        new_query_compiler = self._query_compiler.astype(col_dtypes, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler, not copy)

    @property
    def at(self, axis=None):
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def at_time(self, time, asof=False, axis=None):
        return self._default_to_pandas("at_time", time, asof=asof, axis=axis)

    def between_time(
        self, start_time, end_time, include_start=True, include_end=True, axis=None
    ):
        return self._default_to_pandas(
            "between_time",
            start_time,
            end_time,
            include_start=include_start,
            include_end=include_end,
            axis=axis,
        )

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='bfill')"""
        return self.fillna(
            method="bfill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )

    def bool(self):
        """Return the bool of a single element PandasObject.

        This must be a boolean scalar value, either True or False.  Raise a
        ValueError if the PandasObject does not have exactly 1 element, or that
        element is not boolean
        """
        shape = self.shape
        if shape != (1,) and shape != (1, 1):
            raise ValueError(
                """The PandasObject does not have exactly
                                1 element. Return the bool of a single
                                element PandasObject. The truth value is
                                ambiguous. Use a.empty, a.item(), a.any()
                                or a.all()."""
            )
        else:
            return self._to_pandas().bool()

    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        # validate inputs
        if axis is not None:
            axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        if is_list_like(lower) or is_list_like(upper):
            if axis is None:
                raise ValueError("Must specify axis = 0 or 1")
            self._validate_other(lower, axis)
            self._validate_other(upper, axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = numpy_compat.function.validate_clip_with_axis(axis, args, kwargs)
        # any np.nan bounds are treated as None
        if lower is not None and np.any(np.isnan(lower)):
            lower = None
        if upper is not None and np.any(np.isnan(upper)):
            upper = None
        new_query_compiler = self._query_compiler.clip(
            lower=lower, upper=upper, axis=axis, inplace=inplace, *args, **kwargs
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def clip_lower(self, threshold, axis=None, inplace=False):
        return self.clip(lower=threshold, axis=axis, inplace=inplace)

    def clip_upper(self, threshold, axis=None, inplace=False):
        return self.clip(upper=threshold, axis=axis, inplace=inplace)

    def combine(self, other, func, fill_value=None, **kwargs):
        if isinstance(other, type(self)):
            other = other._to_pandas()
        return self._default_to_pandas(
            "combine", other, func, fill_value=fill_value, **kwargs
        )

    def combine_first(self, other):
        if isinstance(other, type(self)):
            other = other._to_pandas()
        return self._default_to_pandas("combine_first", other=other)

    def compound(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas(
            "compound", axis=axis, skipna=skipna, level=level
        )

    def copy(self, deep=True):
        """Creates a shallow copy of the DataFrame.

        Returns:
            A new DataFrame pointing to the same partitions as this one.
        """
        if deep:
            return self.__constructor__(query_compiler=self._query_compiler.copy())
        new_obj = self.__constructor__(query_compiler=self._query_compiler)
        self._add_sibling(new_obj)
        return new_obj

    def count(self, axis=0, level=None, numeric_only=False):
        """Get the count of non-null objects in the DataFrame.

        Arguments:
            axis: 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            level: If the axis is a MultiIndex (hierarchical), count along a
                particular level, collapsing into a DataFrame.
            numeric_only: Include only float, int, boolean data

        Returns:
            The count, in a Series (or DataFrame if level is specified).
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        return self._reduce_dimension(
            self._query_compiler.count(
                axis=axis, level=level, numeric_only=numeric_only
            )
        )

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative maximum across the DataFrame.

        Args:
            axis (int): The axis to take maximum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative maximum of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if axis:
            self._validate_dtypes()
        return self.__constructor__(
            query_compiler=self._query_compiler.cummax(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative minimum across the DataFrame.

        Args:
            axis (int): The axis to cummin on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative minimum of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if axis:
            self._validate_dtypes()
        return self.__constructor__(
            query_compiler=self._query_compiler.cummin(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative product across the DataFrame.

        Args:
            axis (int): The axis to take product on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative product of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cumprod(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative sum across the DataFrame.

        Args:
            axis (int): The axis to take sum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative sum of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cumsum(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generates descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding NaN values.

        Args:
            percentiles (list-like of numbers, optional):
                The percentiles to include in the output.
            include: White-list of data types to include in results
            exclude: Black-list of data types to exclude in results

        Returns: Series/DataFrame of summary statistics
        """
        if include is not None and (isinstance(include, np.dtype) or include != "all"):
            if not is_list_like(include):
                include = [include]
            include = [
                np.dtype(i)
                if not (isinstance(i, type) and i.__module__ == "numpy")
                else i
                for i in include
            ]
            if not any(
                (isinstance(inc, np.dtype) and inc == d)
                or (
                    not isinstance(inc, np.dtype)
                    and inc.__subclasscheck__(getattr(np, d.__str__()))
                )
                for d in self._get_dtypes()
                for inc in include
            ):
                # This is the error that pandas throws.
                raise ValueError("No objects to concatenate")
        if exclude is not None:
            if not is_list_like(exclude):
                exclude = [exclude]
            exclude = [np.dtype(e) for e in exclude]
            if all(
                (isinstance(exc, np.dtype) and exc == d)
                or (
                    not isinstance(exc, np.dtype)
                    and exc.__subclasscheck__(getattr(np, d.__str__()))
                )
                for d in self._get_dtypes()
                for exc in exclude
            ):
                # This is the error that pandas throws.
                raise ValueError("No objects to concatenate")
        if percentiles is not None:
            pandas.DataFrame()._check_percentile(percentiles)
        return self.__constructor__(
            query_compiler=self._query_compiler.describe(
                percentiles=percentiles, include=include, exclude=exclude
            )
        )

    def diff(self, periods=1, axis=0):
        """Finds the difference between elements on the axis requested

        Args:
            periods: Periods to shift for forming difference
            axis: Take difference over rows or columns

        Returns:
            DataFrame with the diff applied
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.diff(periods=periods, axis=axis)
        )

    def dot(self, other):
        from .dataframe import DataFrame

        self_labels = self.columns if isinstance(self, DataFrame) else self.index
        if isinstance(other, BasePandasDataset):
            common = self_labels.union(other.index)
            if len(common) > len(self_labels) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")
            if isinstance(self, DataFrame) and isinstance(other, DataFrame):
                other = other._to_pandas()
                return self._default_to_pandas("dot", other)
        else:
            other = np.asarray(other)
            self_dim = self.shape[1] if len(self.shape) > 1 else self.shape[0]
            if self_dim != other.shape[0]:
                raise ValueError(
                    "Dot product shape mismatch, {} vs {}".format(
                        self.shape, other.shape
                    )
                )

        if isinstance(other, BasePandasDataset):
            other = other.reindex(index=self_labels)._query_compiler
        return self._reduce_dimension(query_compiler=self._query_compiler.dot(other))

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Return new object with labels in requested axis removed.
        Args:
            labels: Index or column labels to drop.
            axis: Whether to drop labels from the index (0 / 'index') or
                columns (1 / 'columns').
            index, columns: Alternative to specifying axis (labels, axis=1 is
                equivalent to columns=labels).
            level: For MultiIndex
            inplace: If True, do operation inplace and return None.
            errors: If 'ignore', suppress error and existing labels are
                dropped.
        Returns:
            dropped : type of caller
        """
        # TODO implement level
        if level is not None:
            return self._default_to_pandas(
                "drop",
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=inplace,
                errors=errors,
            )

        inplace = validate_bool_kwarg(inplace, "inplace")
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis = pandas.DataFrame()._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pandas.DataFrame()._construct_axes_from_arguments(
                (index, columns), {}
            )
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        # TODO Clean up this error checking
        if "index" not in axes:
            axes["index"] = None
        elif axes["index"] is not None:
            if not is_list_like(axes["index"]):
                axes["index"] = [axes["index"]]
            if errors == "raise":
                non_existant = [obj for obj in axes["index"] if obj not in self.index]
                if len(non_existant):
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["index"] = [obj for obj in axes["index"] if obj in self.index]
                # If the length is zero, we will just do nothing
                if not len(axes["index"]):
                    axes["index"] = None

        if "columns" not in axes:
            axes["columns"] = None
        elif axes["columns"] is not None:
            if not is_list_like(axes["columns"]):
                axes["columns"] = [axes["columns"]]
            if errors == "raise":
                non_existant = [
                    obj for obj in axes["columns"] if obj not in self.columns
                ]
                if len(non_existant):
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["columns"] = [
                    obj for obj in axes["columns"] if obj in self.columns
                ]
                # If the length is zero, we will just do nothing
                if not len(axes["columns"]):
                    axes["columns"] = None

        new_query_compiler = self._query_compiler.drop(
            index=axes["index"], columns=axes["columns"]
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False):
        """Create a new DataFrame from the removed NA values from this one.

        Args:
            axis (int, tuple, or list): The axis to apply the drop.
            how (str): How to drop the NA values.
                'all': drop the label if all values are NA.
                'any': drop the label if any values are NA.
            thresh (int): The minimum number of NAs to require.
            subset ([label]): Labels to consider from other axis.
            inplace (bool): Change this DataFrame or return a new DataFrame.
                True: Modify the data for this DataFrame, return None.
                False: Create a new DataFrame and return it.

        Returns:
            If inplace is set to True, returns None, otherwise returns a new
            DataFrame with the dropna applied.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if is_list_like(axis):
            axis = [self._get_axis_number(ax) for ax in axis]
            result = self

            for ax in axis:
                result = result.dropna(axis=ax, how=how, thresh=thresh, subset=subset)
            return self._create_or_update_from_compiler(result._query_compiler, inplace)

        axis = self._get_axis_number(axis)
        if how is not None and how not in ["any", "all"]:
            raise ValueError("invalid how option: %s" % how)
        if how is None and thresh is None:
            raise TypeError("must specify how or thresh")
        if subset is not None:
            if axis == 1:
                indices = self.index.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
            else:
                indices = self.columns.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
        new_query_compiler = self._query_compiler.dropna(
            axis=axis, how=how, thresh=thresh, subset=subset
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def droplevel(self, level, axis=0):
        """Return index with requested level(s) removed.

        Args:
            level: The level to drop

        Returns:
            Index or MultiIndex
        """
        return self._default_to_pandas("droplevel", level, axis=axis)

    def drop_duplicates(self, keep="first", inplace=False, **kwargs):
        """Return DataFrame with duplicate rows removed, optionally only considering certain columns

            Args:
                subset : column label or sequence of labels, optional
                    Only consider certain columns for identifying duplicates, by
                    default use all of the columns
                keep : {'first', 'last', False}, default 'first'
                    - ``first`` : Drop duplicates except for the first occurrence.
                    - ``last`` : Drop duplicates except for the last occurrence.
                    - False : Drop all duplicates.
                inplace : boolean, default False
                    Whether to drop duplicates in place or to return a copy

            Returns:
                deduplicated : DataFrame
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if kwargs.get("subset", None) is not None:
            duplicates = self.duplicated(keep=keep, **kwargs)
        else:
            duplicates = self.duplicated(keep=keep, **kwargs)
        indices, = duplicates.values.nonzero()
        return self.drop(index=self.index[indices], inplace=inplace)

    def duplicated(self, keep="first", **kwargs):
        return self._default_to_pandas("duplicated", keep=keep, **kwargs)

    def eq(self, other, axis="columns", level=None):
        """Checks element-wise that this is equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the eq over.
            level: The Multilevel index level to apply eq over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("eq", other, axis=axis, level=level)

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        axis=0,
    ):
        return self._default_to_pandas(
            "ewm",
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
        )

    def expanding(self, min_periods=1, center=False, axis=0):
        return self._default_to_pandas(
            "expanding", min_periods=min_periods, center=center, axis=axis
        )

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for fillna(method='ffill')
        """
        return self.fillna(
            method="ffill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
        **kwargs
    ):
        """Fill NA/NaN values using the specified method.

        Args:
            value: Value to use to fill holes. This value cannot be a list.

            method: Method to use for filling holes in reindexed Series pad.
                ffill: propagate last valid observation forward to next valid
                backfill.
                bfill: use NEXT valid observation to fill gap.

            axis: 0 or 'index', 1 or 'columns'.

            inplace: If True, fill in place. Note: this will modify any other
                views on this object.

            limit: If method is specified, this is the maximum number of
                consecutive NaN values to forward/backward fill. In other
                words, if there is a gap with more than this number of
                consecutive NaNs, it will only be partially filled. If method
                is not specified, this is the maximum number of entries along
                the entire axis where NaNs will be filled. Must be greater
                than 0 if not None.

            downcast: A dict of item->dtype of what to downcast if possible,
                or the string 'infer' which will try to downcast to an
                appropriate equal type.

        Returns:
            filled: DataFrame
        """
        # TODO implement value passed as DataFrame/Series
        if isinstance(value, BasePandasDataset):
            new_query_compiler = self._default_to_pandas(
                "fillna",
                value=value._to_pandas(),
                method=method,
                axis=axis,
                inplace=False,
                limit=limit,
                downcast=downcast,
                **kwargs
            )._query_compiler
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = self._get_axis_number(axis) if axis is not None else 0
        if isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                'you passed a "{0}"'.format(type(value).__name__)
            )
        if value is None and method is None:
            raise ValueError("must specify a fill method or value")
        if value is not None and method is not None:
            raise ValueError("cannot specify both a fill method and value")
        if method is not None and method not in ["backfill", "bfill", "pad", "ffill"]:
            expecting = "pad (ffill) or backfill (bfill)"
            msg = "Invalid fill method. Expecting {expecting}. Got {method}".format(
                expecting=expecting, method=method
            )
            raise ValueError(msg)
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            elif limit <= 0:
                raise ValueError("Limit must be greater than 0")

        new_query_compiler = self._query_compiler.fillna(
            value=value,
            method=method,
            axis=axis,
            inplace=False,
            limit=limit,
            downcast=downcast,
            **kwargs
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """Subset rows or columns based on their labels

        Args:
            items (list): list of labels to subset
            like (string): retain labels where `arg in label == True`
            regex (string): retain labels matching regex input
            axis: axis to filter on

        Returns:
            A new DataFrame with the filter applied.
        """
        nkw = count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` are mutually exclusive"
            )
        if nkw == 0:
            raise TypeError("Must pass either `items`, `like`, or `regex`")
        if axis is None:
            axis = "columns"  # This is the default info axis for dataframes

        axis = self._get_axis_number(axis)
        labels = self.columns if axis else self.index

        if items is not None:
            bool_arr = labels.isin(items)
        elif like is not None:

            def f(x):
                return like in to_str(x)

            bool_arr = labels.map(f).tolist()
        else:

            def f(x):
                return matcher.search(to_str(x)) is not None

            matcher = re.compile(regex)
            bool_arr = labels.map(f).tolist()
        if not axis:
            return self[bool_arr]
        return self[self.columns[bool_arr]]

    def first(self, offset):
        return self._default_to_pandas("first", offset)

    def first_valid_index(self):
        """Return index for first non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._query_compiler.first_valid_index()

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    @classmethod
    def from_csv(
        cls,
        path,
        header=0,
        sep=",",
        index_col=0,
        parse_dates=True,
        encoding=None,
        tupleize_cols=None,
        infer_datetime_format=False,
    ):
        from .io import read_csv

        return read_csv(
            path,
            header=header,
            sep=sep,
            index_col=index_col,
            parse_dates=parse_dates,
            encoding=encoding,
            tupleize_cols=tupleize_cols,
            infer_datetime_format=infer_datetime_format,
        )

    def ge(self, other, axis="columns", level=None):
        """Checks element-wise that this is greater than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("ge", other, axis=axis, level=level)

    def get(self, key, default=None):
        """Get item from object for given key (DataFrame column, Panel
        slice, etc.). Returns default value if not found.

        Args:
            key (DataFrame column, Panel slice) : the key for which value
            to get

        Returns:
            value (type of items contained in object) : A value that is
            stored at the key
        """
        if key in self.keys():
            return self.__getitem__(key)
        else:
            return default

    def get_dtype_counts(self):
        """Get the counts of dtypes in this object.

        Returns:
            The counts of dtypes in this object.
        """
        if hasattr(self, "dtype"):
            return pandas.Series({str(self.dtype): 1})
        result = self.dtypes.value_counts()
        result.index = result.index.map(lambda x: str(x))
        return result

    def get_ftype_counts(self):
        """Get the counts of ftypes in this object.

        Returns:
            The counts of ftypes in this object.
        """
        if hasattr(self, "ftype"):
            return pandas.Series({self.ftype: 1})
        return self.ftypes.value_counts().sort_index()

    def get_values(self):
        return self._default_to_pandas("get_values")

    def gt(self, other, axis="columns", level=None):
        """Checks element-wise that this is greater than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("gt", other, axis=axis, level=level)

    def head(self, n=5):
        """Get the first n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the first n rows of the DataFrame.
        """
        if n >= len(self.index):
            return self.copy()
        return self.__constructor__(query_compiler=self._query_compiler.head(n))

    @property
    def iat(self, axis=None):
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        """Get the index of the first occurrence of the max value of the axis.

        Args:
            axis (int): Identify the max over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each maximum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self._get_dtypes()):
            raise TypeError("reduction operation 'argmax' not allowed for this dtype")
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmax(axis=axis, skipna=skipna)
        )

    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        """Get the index of the first occurrence of the min value of the axis.

        Args:
            axis (int): Identify the min over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each minimum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self._get_dtypes()):
            raise TypeError("reduction operation 'argmin' not allowed for this dtype")
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmin(axis=axis, skipna=skipna)
        )

    def infer_objects(self):
        return self._default_to_pandas("infer_objects")

    def isin(self, values):
        """Fill a DataFrame with booleans for cells contained in values.

        Args:
            values (iterable, DataFrame, Series, or dict): The values to find.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is in values.
            True: cell is contained in values.
            False: otherwise
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.isin(values=values)
        )

    def isna(self):
        """Fill a DataFrame with booleans for cells containing NA.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is NA.
            True: cell contains NA.
            False: otherwise.
        """
        return self.__constructor__(query_compiler=self._query_compiler.isna())

    isnull = isna

    @property
    def ix(self, axis=None):
        raise ErrorMessage.not_implemented("ix is not implemented.")

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._default_to_pandas(
            "kurt",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs
        )

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._default_to_pandas(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs
        )

    def last(self, offset):
        return self._default_to_pandas("last", offset)

    def last_valid_index(self):
        """Return index for last non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._query_compiler.last_valid_index()

    def le(self, other, axis="columns", level=None):
        """Checks element-wise that this is less than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the le over.
            level: The Multilevel index level to apply le over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("le", other, axis=axis, level=level)

    def lt(self, other, axis="columns", level=None):
        """Checks element-wise that this is less than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the lt over.
            level: The Multilevel index level to apply lt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("lt", other, axis=axis, level=level)

    @property
    def loc(self):
        """Purely label-location based indexer for selection by label.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def mad(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas("mad", axis=axis, skipna=skipna, level=level)

    def mask(
        self,
        cond,
        other=nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        raise_on_error=None,
    ):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas(
            "mask",
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
            raise_on_error=raise_on_error,
        )

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Perform max across the DataFrame.

        Args:
            axis (int): The axis to take the max on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The max of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        data = self._validate_dtypes_min_max(axis, numeric_only)
        return data._reduce_dimension(
            data._query_compiler.max(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Computes mean across the DataFrame.

        Args:
            axis (int): The axis to take the mean on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The mean of the DataFrame. (Pandas series)
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        data = self._validate_dtypes_sum_prod_mean(
            axis, numeric_only, ignore_axis=False
        )
        return data._reduce_dimension(
            data._query_compiler.mean(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Computes median across the DataFrame.

        Args:
            axis (int): The axis to take the median on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The median of the DataFrame. (Pandas series)
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        return self._reduce_dimension(
            self._query_compiler.median(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def memory_usage(self, index=True, deep=False):
        """Returns the memory usage of each column in bytes

        Args:
            index (bool): Whether to include the memory usage of the DataFrame's
                index in returned Series. Defaults to True
            deep (bool): If True, introspect the data deeply by interrogating
            objects dtypes for system-level memory consumption. Defaults to False

        Returns:
            A Series where the index are the column names and the values are
            the memory usage of each of the columns in bytes. If `index=true`,
            then the first value of the Series will be 'Index' with its memory usage.
        """
        assert not index, "Internal Error. Index must be evaluated in child class"
        return self._reduce_dimension(
            self._query_compiler.memory_usage(index=index, deep=deep)
        )

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Perform min across the DataFrame.

        Args:
            axis (int): The axis to take the min on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The min of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        data = self._validate_dtypes_min_max(axis, numeric_only)
        return data._reduce_dimension(
            data._query_compiler.min(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def mod(self, other, axis="columns", level=None, fill_value=None):
        """Mods this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the mod against this.
            axis: The axis to mod over.
            level: The Multilevel index level to apply mod over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Mod applied.
        """
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    def mode(self, axis=0, numeric_only=False, dropna=True):
        """Perform mode across the DataFrame.

        Args:
            axis (int): The axis to take the mode on.
            numeric_only (bool): if True, only apply to numeric columns.

        Returns:
            DataFrame: The mode of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.mode(
                axis=axis, numeric_only=numeric_only, dropna=dropna
            )
        )

    def mul(self, other, axis="columns", level=None, fill_value=None):
        """Multiplies this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the multiply against this.
            axis: The axis to multiply over.
            level: The Multilevel index level to apply multiply over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Multiply applied.
        """
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    multiply = mul

    def ne(self, other, axis="columns", level=None):
        """Checks element-wise that this is not equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the ne over.
            level: The Multilevel index level to apply ne over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("ne", other, axis=axis, level=level)

    def notna(self):
        """Perform notna across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return self.__constructor__(query_compiler=self._query_compiler.notna())

    notnull = notna

    def nunique(self, axis=0, dropna=True):
        """Return Series with number of distinct
           observations over requested axis.

        Args:
            axis : {0 or 'index', 1 or 'columns'}, default 0
            dropna : boolean, default True

        Returns:
            nunique : Series
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        return self._reduce_dimension(
            self._query_compiler.nunique(axis=axis, dropna=dropna)
        )

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, **kwargs):
        return self._default_to_pandas(
            "pct_change",
            periods=periods,
            fill_method=fill_method,
            limit=limit,
            freq=freq,
            **kwargs
        )

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, *args, **kwargs)

        Args:
            func: function to apply to the df.
            args: positional arguments passed into ``func``.
            kwargs: a dictionary of keyword arguments passed into ``func``.

        Returns:
            object: the return type of ``func``.
        """
        return _pipe(self, func, *args, **kwargs)

    def pop(self, item):
        """Pops an item from this DataFrame and returns it.

        Args:
            item (str): Column label to be popped

        Returns:
            A Series containing the popped values. Also modifies this
            DataFrame.
        """
        result = self[item]
        del self[item]
        return result

    def pow(self, other, axis="columns", level=None, fill_value=None):
        """Pow this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the pow against this.
            axis: The axis to pow over.
            level: The Multilevel index level to apply pow over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Pow applied.
        """
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        """Return the product of the values for the requested axis

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            level : int or level name, default None
            numeric_only : boolean, default None
            min_count : int, default 0

        Returns:
            prod : Series or DataFrame (if level specified)
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        data = self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=True)
        return data._reduce_dimension(
            data._query_compiler.prod(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs
            )
        )

    product = prod
    radd = add

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation="linear"):
        """Return values at the given quantile over requested axis,
            a la numpy.percentile.

        Args:
            q (float): 0 <= q <= 1, the quantile(s) to compute
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                Specifies which interpolation method to use

        Returns:
            quantiles : Series or DataFrame
                    If q is an array, a DataFrame will be returned where the
                    index is q, the columns are the columns of self, and the
                    values are the quantiles.

                    If q is a float, a Series will be returned where the
                    index is the columns of self and the values
                    are the quantiles.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0

        def check_dtype(t):
            return is_numeric_dtype(t) or is_datetime_or_timedelta_dtype(t)

        if not numeric_only:
            # If not numeric_only and columns, then check all columns are either
            # numeric, timestamp, or timedelta
            if not axis and not all(check_dtype(t) for t in self._get_dtypes()):
                raise TypeError("can't multiply sequence by non-int of type 'float'")
            # If over rows, then make sure that all dtypes are equal for not
            # numeric_only
            elif axis:
                for i in range(1, len(self._get_dtypes())):
                    pre_dtype = self._get_dtypes()[i - 1]
                    curr_dtype = self._get_dtypes()[i]
                    if not is_dtype_equal(pre_dtype, curr_dtype):
                        raise TypeError(
                            "Cannot compare type '{0}' with type '{1}'".format(
                                pre_dtype, curr_dtype
                            )
                        )
        else:
            # Normally pandas returns this near the end of the quantile, but we
            # can't afford the overhead of running the entire operation before
            # we error.
            if not any(is_numeric_dtype(t) for t in self._get_dtypes()):
                raise ValueError("need at least one array to concatenate")

        # check that all qs are between 0 and 1
        pandas.DataFrame()._check_percentile(q)
        axis = self._get_axis_number(axis)

        if isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list)):
            return self.__constructor__(
                query_compiler=self._query_compiler.quantile_for_list_of_values(
                    q=q,
                    axis=axis,
                    numeric_only=numeric_only,
                    interpolation=interpolation,
                )
            )
        else:
            return self._reduce_dimension(
                self._query_compiler.quantile_for_single_value(
                    q=q,
                    axis=axis,
                    numeric_only=numeric_only,
                    interpolation=interpolation,
                )
            )

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
    ):
        """
        Compute numerical data ranks (1 through n) along axis.
        Equal values are assigned a rank that is the [method] of
        the ranks of those values.

        Args:
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            method: {'average', 'min', 'max', 'first', 'dense'}
                Specifies which method to use for equal vals
            numeric_only (boolean)
                Include only float, int, boolean data.
            na_option: {'keep', 'top', 'bottom'}
                Specifies how to handle NA options
            ascending (boolean):
                Decedes ranking order
            pct (boolean):
                Computes percentage ranking of data
        Returns:
            A new DataFrame
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.rank(
                axis=axis,
                method=method,
                numeric_only=numeric_only,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=np.nan,
        limit=None,
        tolerance=None,
    ):
        axis = self._get_axis_number(axis) if axis is not None else 0
        if (
            level is not None
            or (
                (columns is not None or axis == 1)
                and isinstance(self.columns, pandas.MultiIndex)
            )
            or (
                (index is not None or axis == 0)
                and isinstance(self.index, pandas.MultiIndex)
            )
        ):
            return self._default_to_pandas(
                "reindex",
                labels=labels,
                index=index,
                columns=columns,
                axis=axis,
                method=method,
                copy=copy,
                level=level,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels
        new_query_compiler = None
        if index is not None:
            if not isinstance(index, pandas.Index):
                index = pandas.Index(index)
            if not index.equals(self.index):
                new_query_compiler = self._query_compiler.reindex(
                    0,
                    index,
                    method=method,
                    fill_value=fill_value,
                    limit=limit,
                    tolerance=tolerance,
                )
        if new_query_compiler is None:
            new_query_compiler = self._query_compiler
        final_query_compiler = None
        if columns is not None:
            if not isinstance(columns, pandas.Index):
                columns = pandas.Index(columns)
            if not columns.equals(self.columns):
                final_query_compiler = new_query_compiler.reindex(
                    1,
                    columns,
                    method=method,
                    fill_value=fill_value,
                    limit=limit,
                    tolerance=tolerance,
                )
        if final_query_compiler is None:
            final_query_compiler = new_query_compiler
        return self._create_or_update_from_compiler(final_query_compiler, not copy)

    def reindex_axis(
        self,
        labels,
        axis=0,
        method=None,
        level=None,
        copy=True,
        limit=None,
        fill_value=nan,
    ):
        return self._default_to_pandas(
            "reindex_axis",
            labels,
            axis=axis,
            method=method,
            level=level,
            copy=copy,
            limit=limit,
            fill_value=fill_value,
        )

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas(
            "reindex_like",
            other,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )

    def rename_axis(
        self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False
    ):
        kwargs = {
            "index": index,
            "columns": columns,
            "axis": axis,
            "copy": copy,
            "inplace": inplace,
        }
        axes, kwargs = getattr(pandas, self.__name__)()._construct_axes_from_arguments(
            (), kwargs, sentinel=sentinel
        )
        if axis is not None:
            axis = self._get_axis_number(axis)
        else:
            axis = 0
        inplace = validate_bool_kwarg(inplace, "inplace")

        if mapper is not None:
            # Use v0.23 behavior if a scalar or list
            non_mapper = is_scalar(mapper) or (
                is_list_like(mapper) and not is_dict_like(mapper)
            )
            if non_mapper:
                return self._set_axis_name(mapper, axis=axis, inplace=inplace)
            else:
                # Deprecated (v0.21) behavior is if mapper is specified,
                # and not a list or scalar, then call rename
                msg = (
                    "Using 'rename_axis' to alter labels is deprecated. "
                    "Use '.rename' instead"
                )
                warnings.warn(msg, FutureWarning, stacklevel=3)
                axis = pandas.DataFrame()._get_axis_name(axis)
                d = {"copy": copy, "inplace": inplace, axis: mapper}
                return self.rename(**d)
        else:
            # Use new behavior.  Means that index and/or columns is specified
            result = self if inplace else self.copy(deep=copy)

            for axis in axes:
                if axes[axis] is None:
                    continue
                v = axes[axis]
                axis = self._get_axis_number(axis)
                non_mapper = is_scalar(v) or (is_list_like(v) and not is_dict_like(v))
                if non_mapper:
                    newnames = v
                else:
                    f = _get_rename_function(v)
                    curnames = self.index.names if axis == 0 else self.columns.names
                    newnames = [f(name) for name in curnames]
                result._set_axis_name(newnames, axis=axis, inplace=True)
            if not inplace:
                return result

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):
        return self._default_to_pandas(
            "replace",
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )

    def resample(
        self,
        rule,
        how=None,
        axis=0,
        fill_method=None,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        limit=None,
        base=0,
        on=None,
        level=None,
    ):
        return self._default_to_pandas(
            "resample",
            rule,
            how=how,
            axis=axis,
            fill_method=fill_method,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            loffset=loffset,
            limit=limit,
            base=base,
            on=on,
            level=level,
        )

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        """Reset this index to default and create column from current index.

        Args:
            level: Only remove the given levels from the index. Removes all
                levels by default
            drop: Do not try to insert index into DataFrame columns. This
                resets the index to the default integer index.
            inplace: Modify the DataFrame in place (do not create a new object)
            col_level : If the columns have multiple levels, determines which
                level the labels are inserted into. By default it is inserted
                into the first level.
            col_fill: If the columns have multiple levels, determines how the
                other levels are named. If None then the index name is
                repeated.

        Returns:
            A new DataFrame if inplace is False, None otherwise.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # TODO Implement level
        if level is not None or isinstance(self.index, pandas.MultiIndex):
            new_query_compiler = self._default_to_pandas(
                "reset_index",
                level=level,
                drop=drop,
                inplace=False,
                col_level=col_level,
                col_fill=col_fill,
            )._query_compiler
        # Error checking for matching Pandas. Pandas does not allow you to
        # insert a dropped index into a DataFrame if these columns already
        # exist.
        elif (
            not drop
            and not isinstance(self.index, pandas.MultiIndex)
            and all(n in self.columns for n in ["level_0", "index"])
        ):
            raise ValueError("cannot insert level_0, already exists")
        else:
            new_query_compiler = self._query_compiler.reset_index(
                drop=drop, level=level
            )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rfloordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def rmod(self, other, axis="columns", level=None, fill_value=None):
        """Mod this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the div against this.
            axis: The axis to div over.
            level: The Multilevel index level to apply div over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the rdiv applied.
        """
        return self._binary_op(
            "rmod", other, axis=axis, level=level, fill_value=fill_value
        )

    rmul = mul

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):
        return self._default_to_pandas(
            "rolling",
            window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
        )

    def round(self, decimals=0, *args, **kwargs):
        """Round each element in the DataFrame.

        Args:
            decimals: The number of decimals to round to.

        Returns:
             A new DataFrame.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.round(decimals=decimals, **kwargs)
        )

    def rpow(self, other, axis="columns", level=None, fill_value=None):
        """Pow this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the pow against this.
            axis: The axis to pow over.
            level: The Multilevel index level to apply pow over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Pow applied.
        """
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        """Subtract a DataFrame/Series/scalar from this DataFrame.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: The axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        """Div this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the div against this.
            axis: The axis to div over.
            level: The Multilevel index level to apply div over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the rdiv applied.
        """
        return self._binary_op(
            "rtruediv", other, axis=axis, level=level, fill_value=fill_value
        )

    rdiv = rtruediv

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
    ):
        """Returns a random sample of items from an axis of object.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
                Default = 1 if frac = None.
            frac: Fraction of axis items to return. Cannot be used with n.
            replace: Sample with or without replacement. Default = False.
            weights: Default 'None' results in equal probability weighting.
                If passed a Series, will align with target object on index.
                Index values in weights not found in sampled object will be
                ignored and index values in sampled object not in weights will
                be assigned weights of zero. If called on a DataFrame, will
                accept the name of a column when axis = 0. Unless weights are
                a Series, weights must be same length as axis being sampled.
                If weights do not sum to 1, they will be normalized to sum
                to 1. Missing values in the weights column will be treated as
                zero. inf and -inf values not allowed.
            random_state: Seed for the random number generator (if int), or
                numpy RandomState object.
            axis: Axis to sample. Accepts axis number or name.

        Returns:
            A new Dataframe
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if axis:
            axis_labels = self.columns
            axis_length = len(axis_labels)
        else:
            # Getting rows requires indices instead of labels. RangeIndex provides this.
            axis_labels = pandas.RangeIndex(len(self.index))
            axis_length = len(axis_labels)
        if weights is not None:
            # Index of the weights Series should correspond to the index of the
            # Dataframe in order to sample
            if isinstance(weights, BasePandasDataset):
                weights = weights.reindex(self.axes[axis])
            # If weights arg is a string, the weights used for sampling will
            # the be values in the column corresponding to that string
            if isinstance(weights, string_types):
                if axis == 0:
                    try:
                        weights = self[weights]
                    except KeyError:
                        raise KeyError("String passed to weights not a valid column")
                else:
                    raise ValueError(
                        "Strings can only be passed to "
                        "weights when sampling from rows on "
                        "a DataFrame"
                    )
            weights = pandas.Series(weights, dtype="float64")

            if len(weights) != axis_length:
                raise ValueError(
                    "Weights and axis to be sampled must be of same length"
                )
            if (weights == np.inf).any() or (weights == -np.inf).any():
                raise ValueError("weight vector may not include `inf` values")
            if (weights < 0).any():
                raise ValueError("weight vector many not include negative values")
            # weights cannot be NaN when sampling, so we must set all nan
            # values to 0
            weights = weights.fillna(0)
            # If passed in weights are not equal to 1, renormalize them
            # otherwise numpy sampling function will error
            weights_sum = weights.sum()
            if weights_sum != 1:
                if weights_sum != 0:
                    weights = weights / weights_sum
                else:
                    raise ValueError("Invalid weights: weights sum to zero")
            weights = weights.values

        if n is None and frac is None:
            # default to n = 1 if n and frac are both None (in accordance with
            # Pandas specification)
            n = 1
        elif n is not None and frac is None and n % 1 != 0:
            # n must be an integer
            raise ValueError("Only integers accepted as `n` values")
        elif n is None and frac is not None:
            # compute the number of samples based on frac
            n = int(round(frac * axis_length))
        elif n is not None and frac is not None:
            # Pandas specification does not allow both n and frac to be passed
            # in
            raise ValueError("Please enter a value for `frac` OR `n`, not both")
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide positive value."
            )
        if n == 0:
            # This returns an empty object, and since it is a weird edge case that
            # doesn't need to be distributed, we default to pandas for n=0.
            return self._default_to_pandas(
                "sample",
                n=n,
                frac=frac,
                replace=replace,
                weights=weights,
                random_state=random_state,
                axis=axis,
            )
        if random_state is not None:
            # Get a random number generator depending on the type of
            # random_state that is passed in
            if isinstance(random_state, int):
                random_num_gen = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.randomState):
                random_num_gen = random_state
            else:
                # random_state must be an int or a numpy RandomState object
                raise ValueError(
                    "Please enter an `int` OR a "
                    "np.random.RandomState for random_state"
                )
            # choose random numbers and then get corresponding labels from
            # chosen axis
            sample_indices = random_num_gen.choice(
                np.arange(0, axis_length), size=n, replace=replace, p=weights
            )
            samples = axis_labels[sample_indices]
        else:
            # randomly select labels from chosen axis
            samples = np.random.choice(
                a=axis_labels, size=n, replace=replace, p=weights
            )
        if axis:
            query_compiler = self._query_compiler.getitem_column_array(samples)
            return self.__constructor__(query_compiler=query_compiler)
        else:
            query_compiler = self._query_compiler.getitem_row_array(samples)
            return self.__constructor__(query_compiler=query_compiler)

    def sem(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._default_to_pandas(
            "sem",
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs
        )

    def select(self, crit, axis=0):
        return self._default_to_pandas("select", crit, axis=axis)

    def set_axis(self, labels, axis=0, inplace=None):
        """Assign desired index to given axis.

        Args:
            labels (pandas.Index or list-like): The Index to assign.
            axis (string or int): The axis to reassign.
            inplace (bool): Whether to make these modifications inplace.

        Returns:
            If inplace is False, returns a new DataFrame, otherwise None.
        """
        if is_scalar(labels):
            warnings.warn(
                'set_axis now takes "labels" as first argument, and '
                '"axis" as named parameter. The old form, with "axis" as '
                'first parameter and "labels" as second, is still supported '
                "but will be deprecated in a future version of pandas.",
                FutureWarning,
                stacklevel=2,
            )
            labels, axis = axis, labels
        if inplace is None:
            warnings.warn(
                "set_axis currently defaults to operating inplace.\nThis "
                "will change in a future version of pandas, use "
                "inplace=True to avoid this warning.",
                FutureWarning,
                stacklevel=2,
            )
            inplace = True
        if inplace:
            setattr(self, pandas.DataFrame()._get_axis_name(axis), labels)
        else:
            obj = self.copy()
            obj.set_axis(labels, axis=axis, inplace=True)
            return obj

    def set_value(self, index, col, value, takeable=False):
        return self._default_to_pandas(
            "set_value", index, col, value, takeable=takeable
        )

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        return self._default_to_pandas(
            "shift", periods=periods, freq=freq, axis=axis, fill_value=fill_value
        )

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Return unbiased skew over requested axis Normalized by N-1

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            Exclude NA/null values when computing the result.
            level : int or level name, default None
            numeric_only : boolean, default None

        Returns:
            skew : Series or DataFrame (if level specified)
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        return self._reduce_dimension(
            self._query_compiler.skew(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def slice_shift(self, periods=1, axis=0):
        return self._default_to_pandas("slice_shift", periods=periods, axis=axis)

    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        by=None,
    ):
        """Sort a DataFrame by one of the indices (columns or index).

        Args:
            axis: The axis to sort over.
            level: The MultiIndex level to sort over.
            ascending: Ascending or descending
            inplace: Whether or not to update this DataFrame inplace.
            kind: How to perform the sort.
            na_position: Where to position NA on the sort.
            sort_remaining: On Multilevel Index sort based on all levels.
            by: (Deprecated) argument to pass to sort_values.

        Returns:
            A sorted DataFrame
        """
        axis = self._get_axis_number(axis)
        if level is not None or (
            (axis == 0 and isinstance(self.index, pandas.MultiIndex))
            or (axis == 1 and isinstance(self.columns, pandas.MultiIndex))
        ):
            new_query_compiler = self._default_to_pandas(
                "sort_index",
                axis=axis,
                level=level,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
            )._query_compiler
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        if by is not None:
            warnings.warn(
                "by argument to sort_index is deprecated, "
                "please use .sort_values(by=...)",
                FutureWarning,
                stacklevel=2,
            )
            if level is not None:
                raise ValueError("unable to simultaneously sort by and level")
            return self.sort_values(by, axis=axis, ascending=ascending, inplace=inplace)
        new_query_compiler = self._query_compiler.sort_index(
            axis=axis, ascending=ascending, kind=kind, na_position=na_position
        )
        if inplace:
            self._update_inplace(new_query_compiler=new_query_compiler)
        else:
            return self.__constructor__(query_compiler=new_query_compiler)

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
    ):
        """Sorts by a column/row or list of columns/rows.

        Args:
            by: A list of labels for the axis to sort over.
            axis: The axis to sort.
            ascending: Sort in ascending or descending order.
            inplace: If true, do the operation inplace.
            kind: How to sort.
            na_position: Where to put np.nan values.

        Returns:
             A sorted DataFrame.
        """
        axis = self._get_axis_number(axis)
        if not is_list_like(by):
            by = [by]
        # Currently, sort_values will just reindex based on the sorted values.
        # TODO create a more efficient way to sort
        if axis == 0:
            broadcast_value_dict = {col: self[col]._to_pandas() for col in by}
            broadcast_values = pandas.DataFrame(broadcast_value_dict, index=self.index)
            new_index = broadcast_values.sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
            ).index
            return self.reindex(index=new_index, copy=not inplace)
        else:
            broadcast_value_list = [
                self[row :: len(self.index)]._to_pandas() for row in by
            ]
            index_builder = list(zip(broadcast_value_list, by))
            broadcast_values = pandas.concat(
                [row for row, idx in index_builder], copy=False
            )
            broadcast_values.columns = self.columns
            new_columns = broadcast_values.sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
            ).columns
            return self.reindex(columns=new_columns, copy=not inplace)

    def std(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        """Computes standard deviation across the DataFrame.

        Args:
            axis (int): The axis to take the std on.
            skipna (bool): True to skip NA values, false otherwise.
            ddof (int): degrees of freedom

        Returns:
            The std of the DataFrame (Pandas Series)
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        return self._reduce_dimension(
            self._query_compiler.std(
                axis=axis,
                skipna=skipna,
                level=level,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def sub(self, other, axis="columns", level=None, fill_value=None):
        """Subtract a DataFrame/Series/scalar from this DataFrame.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: The axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self._binary_op(
            "sub", other, axis=axis, level=level, fill_value=fill_value
        )

    subtract = sub

    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        """Perform a sum across the DataFrame.

        Args:
            axis (int): The axis to sum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The sum of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        data = self._validate_dtypes_sum_prod_mean(
            axis, numeric_only, ignore_axis=False
        )
        return data._reduce_dimension(
            data._query_compiler.sum(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs
            )
        )

    def swapaxes(self, axis1, axis2, copy=True):
        return self._default_to_pandas("swapaxes", axis1, axis2, copy=copy)

    def swaplevel(self, i=-2, j=-1, axis=0):
        return self._default_to_pandas("swaplevel", i=i, j=j, axis=axis)

    def take(self, indices, axis=0, convert=None, is_copy=True, **kwargs):
        return self._default_to_pandas(
            "take", indices, axis=axis, convert=convert, is_copy=is_copy, **kwargs
        )

    def tail(self, n=5):
        """Get the last n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the last n rows of this DataFrame.
        """
        if n >= len(self.index):
            return self.copy()
        return self.__constructor__(query_compiler=self._query_compiler.tail(n))

    def to_clipboard(self, excel=True, sep=None, **kwargs):  # pragma: no cover
        return self._default_to_pandas("to_clipboard", excel=excel, sep=sep, **kwargs)

    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        line_terminator=None,
        chunksize=None,
        tupleize_cols=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        *args,
        **kwargs
    ):  # pragma: no cover

        kwargs = {
            "path_or_buf": path_or_buf,
            "sep": sep,
            "na_rep": na_rep,
            "float_format": float_format,
            "columns": columns,
            "header": header,
            "index": index,
            "index_label": index_label,
            "mode": mode,
            "encoding": encoding,
            "compression": compression,
            "quoting": quoting,
            "quotechar": quotechar,
            "line_terminator": line_terminator,
            "chunksize": chunksize,
            "tupleize_cols": tupleize_cols,
            "date_format": date_format,
            "doublequote": doublequote,
            "escapechar": escapechar,
            "decimal": decimal,
        }
        return self._default_to_pandas("to_csv", **kwargs)

    def to_dense(self):  # pragma: no cover
        return self._default_to_pandas("to_dense")

    def to_dict(self, orient="dict", into=dict):  # pragma: no cover
        return self._default_to_pandas("to_dict", orient=orient, into=into)

    def to_excel(
        self,
        excel_writer,
        sheet_name="Sheet1",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells=True,
        encoding=None,
        inf_rep="inf",
        verbose=True,
        freeze_panes=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_excel",
            excel_writer,
            sheet_name,
            na_rep,
            float_format,
            columns,
            header,
            index,
            index_label,
            startrow,
            startcol,
            engine,
            merge_cells,
            encoding,
            inf_rep,
            verbose,
            freeze_panes,
        )

    def to_hdf(self, path_or_buf, key, format="table", **kwargs):  # pragma: no cover
        return self._default_to_pandas(
            "to_hdf", path_or_buf, key, format=format, **kwargs
        )

    def to_json(
        self,
        path_or_buf=None,
        orient=None,
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=False,
        compression="infer",
        index=True,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_json",
            path_or_buf,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
        )

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        bold_rows=False,
        column_format=None,
        longtable=None,
        escape=None,
        encoding=None,
        decimal=".",
        multicolumn=None,
        multicolumn_format=None,
        multirow=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_latex",
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            bold_rows=bold_rows,
            column_format=column_format,
            longtable=longtable,
            escape=escape,
            encoding=encoding,
            decimal=decimal,
            multicolumn=multicolumn,
            multicolumn_format=multicolumn_format,
            multirow=multirow,
        )

    def to_msgpack(
        self, path_or_buf=None, encoding="utf-8", **kwargs
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_msgpack", path_or_buf=path_or_buf, encoding=encoding, **kwargs
        )

    def to_numpy(self, dtype=None, copy=False):
        """Convert the DataFrame to a NumPy array.

        Args:
            dtype: The dtype to pass to numpy.asarray()
            copy: Whether to ensure that the returned value is a not a view on another
                array.

        Returns:
            A numpy array.
        """
        arr = self._query_compiler.to_numpy()
        if dtype is not None:
            return np.asarray(arr, dtype)
        return arr

    # TODO(williamma12): When this gets implemented, have the series one call this.
    def to_period(self, freq=None, axis=0, copy=True):  # pragma: no cover
        return self._default_to_pandas("to_period", freq=freq, axis=axis, copy=copy)

    def to_pickle(
        self, path, compression="infer", protocol=pkl.HIGHEST_PROTOCOL
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_pickle", path, compression=compression, protocol=protocol
        )

    def to_sparse(self, fill_value=None, kind="block"):
        return self._default_to_pandas("to_sparse", fill_value=fill_value, kind=kind)

    def to_string(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        line_width=None,
    ):
        return self._default_to_pandas(
            "to_string",
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            line_width=line_width,
        )

    def to_sql(
        self,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        new_query_compiler = self._query_compiler
        # writing the index to the database by inserting it to the DF
        if index:
            if not index_label:
                index_label = "index"
            new_query_compiler = new_query_compiler.insert(0, index_label, self.index)
            # so pandas._to_sql will not write the index to the database as well
            index = False

        from modin.data_management.factories import BaseFactory

        BaseFactory.to_sql(
            new_query_compiler,
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    # TODO(williamma12): When this gets implemented, have the series one call this.
    def to_timestamp(self, freq=None, how="start", axis=0, copy=True):
        return self._default_to_pandas(
            "to_timestamp", freq=freq, how=how, axis=axis, copy=copy
        )

    def to_xarray(self):
        return self._default_to_pandas("to_xarray")

    def truediv(self, other, axis="columns", level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._binary_op(
            "truediv", other, axis=axis, level=level, fill_value=fill_value
        )

    div = divide = truediv

    def truncate(self, before=None, after=None, axis=None, copy=True):
        return self._default_to_pandas(
            "truncate", before=before, after=after, axis=axis, copy=copy
        )

    def tshift(self, periods=1, freq=None, axis=0):
        return self._default_to_pandas("tshift", periods=periods, freq=freq, axis=axis)

    def transform(self, func, axis=0, *args, **kwargs):
        kwargs["is_transform"] = True
        result = self.agg(func, axis=axis, *args, **kwargs)
        try:
            assert len(result) == len(self)
        except Exception:
            raise ValueError("transforms cannot produce aggregated results")
        return result

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        return self._default_to_pandas(
            "tz_convert", tz, axis=axis, level=level, copy=copy
        )

    def tz_localize(
        self, tz, axis=0, level=None, copy=True, ambiguous="raise", nonexistent="raise"
    ):
        return self._default_to_pandas(
            "tz_localize",
            tz,
            axis=axis,
            level=level,
            copy=copy,
            ambiguous=ambiguous,
            nonexistent=nonexistent,
        )

    def unstack(self, level=-1, fill_value=None):
        return self._default_to_pandas("unstack", level=level, fill_value=fill_value)

    def var(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        """Computes variance across the DataFrame.

        Args:
            axis (int): The axis to take the variance on.
            skipna (bool): True to skip NA values, false otherwise.
            ddof (int): degrees of freedom

        Returns:
            The variance of the DataFrame.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        return self._reduce_dimension(
            self._query_compiler.var(
                axis=axis,
                skipna=skipna,
                level=level,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs
            )
        )

    def __abs__(self):
        """Creates a modified DataFrame by taking the absolute value.

        Returns:
            A modified DataFrame
        """
        return self.abs()

    def __and__(self, other):
        return self._binary_op("__and__", other, axis=0)

    def __array__(self, dtype=None):
        arr = self.to_numpy(dtype)
        return arr

    def __array_wrap__(self, result, context=None):
        """TODO: This is very inefficient. __array__ and as_matrix have been
        changed to call the more efficient to_numpy, but this has been left
        unchanged since we are not sure of its purpose.
        """
        return self._default_to_pandas("__array_wrap__", result, context=context)

    def __copy__(self, deep=True):
        """Make a copy of this object.

        Args:
            deep: Boolean, deep copy or not.
                  Currently we do not support deep copy.

        Returns:
            A Modin Series/DataFrame object.
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        """Make a -deep- copy of this object.

        Note: This is equivalent to copy(deep=True).

        Args:
            memo: No effect. Just to comply with Pandas API.

        Returns:
            A Modin Series/DataFrame object.
        """
        return self.copy(deep=True)

    def __eq__(self, other):
        return self.eq(other)

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas("__finalize__", other, method=method, **kwargs)

    def __ge__(self, other):
        return self.ge(other)

    def __getitem__(self, key):
        if len(self) == 0:
            return self._default_to_pandas("__getitem__", key)
        # see if we can slice the rows
        # This lets us reuse code in Pandas to error check
        indexer = convert_to_index_sliceable(
            getattr(pandas, self.__name__)(index=self.index), key
        )
        if indexer is not None:
            return self._getitem_slice(indexer)
        else:
            return self._getitem(key)

    def _getitem_slice(self, key):
        # If there is no step, we can fasttrack the codepath to use existing logic from
        # head and tail, which is already pretty fast.
        if key.step is None:
            if key.start is None and key.stop is None:
                return self.copy()

            def compute_offset(value):
                return value - len(self) if value >= 0 else value

            # Head is a negative number, Tail is a positive number
            if key.start is None:
                return self.head(compute_offset(key.stop))
            elif key.stop is None:
                return self.tail(-compute_offset(key.start))
            return self.head(compute_offset(key.stop)).tail(-compute_offset(key.start))
        # We convert to a RangeIndex because getitem_row_array is expecting a list
        # of indices, and RangeIndex will give us the exact indices of each boolean
        # requested.
        key = pandas.RangeIndex(len(self.index))[key]
        return self.__constructor__(
            query_compiler=self._query_compiler.getitem_row_array(key)
        )

    def __getstate__(self):
        return self._default_to_pandas("__getstate__")

    def __gt__(self, other):
        return self.gt(other)

    def __invert__(self):
        if not all(is_numeric_dtype(d) for d in self._get_dtypes()):
            raise TypeError(
                "bad operand type for unary ~: '{}'".format(
                    next(d for d in self._get_dtypes() if not is_numeric_dtype(d))
                )
            )
        return self.__constructor__(query_compiler=self._query_compiler.invert())

    def __le__(self, other):
        return self.le(other)

    def __len__(self):
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def __lt__(self, other):
        return self.lt(other)

    def __ne__(self, other):
        return self.ne(other)

    def __neg__(self):
        """Computes an element wise negative DataFrame

        Returns:
            A modified DataFrame where every element is the negation of before
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.negative())

    def __nonzero__(self):
        raise ValueError(
            "The truth value of a {0} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )

    __bool__ = __nonzero__

    def __or__(self, other):
        return self._binary_op("__or__", other, axis=0)

    def __sizeof__(self):
        return self._default_to_pandas("__sizeof__")

    def __str__(self):  # pragma: no cover
        return repr(self)

    def __xor__(self, other):
        return self._binary_op("__xor__", other, axis=0)

    @property
    def blocks(self):
        def blocks(df):
            """Defined because properties do not have a __name__"""
            return df.blocks

        return self._default_to_pandas(blocks)

    @property
    def is_copy(self):
        warnings.warn(
            "Attribute `is_copy` is deprecated and will be removed in a "
            "future version.",
            FutureWarning,
        )
        # Pandas doesn't do anything so neither do we.
        return

    @property
    def size(self):
        """Get the number of elements in the DataFrame.

        Returns:
            The number of elements in the DataFrame.
        """
        return len(self._query_compiler.index) * len(self._query_compiler.columns)

    @property
    def values(self):
        """Create a numpy array with the values from this object.

        Returns:
            The numpy representation of this object.
        """
        return self.to_numpy()

    @property
    def __name__(self):
        return type(self).__name__

    def __getattribute__(self, item):
        default_behaviors = [
            "__init__",
            "__class__",
            "index",
            "_get_index",
            "_set_index",
            "empty",
            "index",
            "columns",
            "name",
            "_get_name",
            "_set_name",
            "dtypes",
            "dtype",
            "_default_to_pandas",
            "_query_compiler",
            "_to_pandas",
            "_build_repr_df",
            "_reduce_dimension",
            "__repr__",
            "__len__",
        ]
        if item not in default_behaviors:
            method = object.__getattribute__(self, item)
            is_callable = callable(method)
            # We default to pandas on empty DataFrames. This avoids a large amount of
            # pain in underlying implementation and returns a result immediately rather
            # than dealing with the edge cases that empty DataFrames have.
            if self.empty and is_callable:

                def default_handler(*args, **kwargs):
                    return self._default_to_pandas(item, *args, **kwargs)

                return default_handler
        return object.__getattribute__(self, item)
