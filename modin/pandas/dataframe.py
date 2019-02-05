from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.api.types import is_scalar
from pandas.compat import to_str, string_types, numpy as numpy_compat, cPickle as pkl
import pandas.core.common as com
from pandas.core.dtypes.common import (
    _get_dtype_from_object,
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
    is_dtype_equal,
    is_object_dtype,
    is_integer_dtype,
)
from pandas.core.index import _ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer, convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg

import itertools
import functools
import numpy as np
import re
import sys
import warnings

from modin.error_message import ErrorMessage
from .utils import from_pandas, to_pandas, _inherit_docstrings
from .iterator import PartitionIterator
from .series import SeriesView


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame, pandas.DataFrame.__init__]
)
class DataFrame(object):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        query_compiler=None,
    ):
        """Distributed DataFrame object backed by Pandas dataframes.

        Args:
            data (numpy ndarray (structured or homogeneous) or dict):
                Dict can contain Series, arrays, constants, or list-like
                objects.
            index (pandas.Index, list, ObjectID): The row index for this
                DataFrame.
            columns (pandas.Index): The column names for this DataFrame, in
                pandas Index object.
            dtype: Data type to force. Only a single dtype is allowed.
                If None, infer
            copy (boolean): Copy data from inputs.
                Only affects DataFrame / 2d ndarray input.
            query_compiler: A query compiler object to manage distributed computation.
        """
        if isinstance(data, DataFrame):
            self._query_compiler = data._query_compiler
            return

        # Check type of data and use appropriate constructor
        if data is not None or query_compiler is None:

            pandas_df = pandas.DataFrame(
                data=data, index=index, columns=columns, dtype=dtype, copy=copy
            )

            self._query_compiler = from_pandas(pandas_df)._query_compiler
        else:
            self._query_compiler = query_compiler

    def __str__(self):  # pragma: no cover
        return repr(self)

    def _build_repr_df(self, num_rows, num_cols):
        # Add one here so that pandas automatically adds the dots
        # It turns out to be faster to extract 2 extra rows and columns than to
        # build the dots ourselves.
        num_rows_for_head = num_rows // 2 + 1
        num_cols_for_front = num_cols // 2 + 1

        if len(self.index) <= num_rows:
            head = self._query_compiler
            tail = None
        else:
            head = self._query_compiler.head(num_rows_for_head)
            tail = self._query_compiler.tail(num_rows_for_head)

        if len(self.columns) <= num_cols:
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

    def __repr__(self):
        # In the future, we can have this be configurable, just like Pandas.
        num_rows = 60
        num_cols = 30

        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _repr_html_(self):  # pragma: no cover
        """repr function for rendering in Jupyter Notebooks like Pandas
        Dataframes.

        Returns:
            The HTML representation of a Dataframe.
        """
        # In the future, we can have this be configurable, just like Pandas.
        num_rows = 60
        num_cols = 20

        # We use pandas _repr_html_ to get a string of the HTML representation
        # of the dataframe.
        result = self._build_repr_df(num_rows, num_cols)._repr_html_()
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # We split so that we insert our correct dataframe dimensions.
            return result.split("<p>")[
                0
            ] + "<p>{0} rows x {1} columns</p>\n</div>".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _get_index(self):
        """Get the index for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.index

    def _get_columns(self):
        """Get the columns for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_index(self, new_index):
        """Set the index for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._query_compiler.index = new_index

    def _set_columns(self, new_columns):
        """Set the columns for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._query_compiler.columns = new_columns

    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    def _validate_eval_query(self, expr, **kwargs):
        """Helper function to check the arguments to eval() and query()

        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.
        """
        if isinstance(expr, str) and expr == "":
            raise ValueError("expr cannot be an empty string")

        if isinstance(expr, str) and "@" in expr:
            ErrorMessage.not_implemented("Local variables not yet supported in eval.")

        if isinstance(expr, str) and "not" in expr:
            if "parser" in kwargs and kwargs["parser"] == "python":
                ErrorMessage.not_implemented("'Not' nodes are not implemented.")

    @property
    def size(self):
        """Get the number of elements in the DataFrame.

        Returns:
            The number of elements in the DataFrame.
        """
        return len(self.index) * len(self.columns)

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this DataFrame.
        """
        # DataFrames have an invariant that requires they be 2 dimensions.
        return 2

    @property
    def ftypes(self):
        """Get the ftypes for this DataFrame.

        Returns:
            The ftypes for this DataFrame.
        """
        # The ftypes are common across all partitions.
        # The first partition will be enough.
        dtypes = self.dtypes.copy()
        ftypes = ["{0}:dense".format(str(dtype)) for dtype in dtypes.values]
        result = pandas.Series(ftypes, index=self.columns)
        return result

    @property
    def dtypes(self):
        """Get the dtypes for this DataFrame.

        Returns:
            The dtypes for this DataFrame.
        """
        return self._query_compiler.dtypes

    @property
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns:
            True if the DataFrame is empty.
            False otherwise.
        """
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def values(self):
        """Create a numpy array with the values from this DataFrame.

        Returns:
            The numpy representation of this DataFrame.
        """
        return to_pandas(self).values

    @property
    def axes(self):
        """Get the axes for the DataFrame.

        Returns:
            The axes for the DataFrame.
        """
        return [self.index, self.columns]

    @property
    def shape(self):
        """Get the size of each of the dimensions in the DataFrame.

        Returns:
            A tuple with the size of each dimension as they appear in axes().
        """
        return len(self.index), len(self.columns)

    def _update_inplace(self, new_query_compiler):
        """Updates the current DataFrame inplace.

        Args:
            new_query_compiler: The new QueryCompiler to use to manage the data
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        old_query_compiler.free()

    def add_prefix(self, prefix):
        """Add a prefix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):
        """Add a suffix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(self, func):
        """Apply a function to a DataFrame elementwise.

        Args:
            func (callable): The function to apply.
        """
        if not callable(func):
            raise ValueError("'{0}' object is not callable".format(type(func)))
        ErrorMessage.non_verified_udf()
        return DataFrame(query_compiler=self._query_compiler.applymap(func))

    def copy(self, deep=True):
        """Creates a shallow copy of the DataFrame.

        Returns:
            A new DataFrame pointing to the same partitions as this one.
        """
        return DataFrame(query_compiler=self._query_compiler.copy())

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        **kwargs
    ):
        """Apply a groupby to this DataFrame. See _groupby() remote task.
        Args:
            by: The value to groupby.
            axis: The axis to groupby.
            level: The level of the groupby.
            as_index: Whether or not to store result as index.
            sort: Whether or not to sort the result by the index.
            group_keys: Whether or not to group the keys.
            squeeze: Whether or not to squeeze.
        Returns:
            A new DataFrame resulting from the groupby.
        """
        axis = pandas.DataFrame()._get_axis_number(axis)
        idx_name = None
        if callable(by):
            by = by(self.index)
        elif isinstance(by, string_types):
            idx_name = by
            by = self.__getitem__(by).values.tolist()
        elif is_list_like(by):
            if isinstance(by, pandas.Series):
                idx_name = by.name
                by = by.values.tolist()

            mismatch = (
                len(by) != len(self) if axis == 0 else len(by) != len(self.columns)
            )

            if all(obj in self for obj in by) and mismatch:
                # In the future, we will need to add logic to handle this, but for now
                # we default to pandas in this case.
                pass
            elif mismatch:
                raise KeyError(next(x for x in by if x not in self))

        from .groupby import DataFrameGroupBy

        return DataFrameGroupBy(
            self,
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            squeeze,
            idx_name,
            **kwargs
        )

    def sum(
        self,
        axis=None,
        skipna=True,
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=False)

        return self._query_compiler.sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    def abs(self):
        """Apply an absolute value function to all numeric columns.

        Returns:
            A new DataFrame with the applied absolute value.
        """
        self._validate_dtypes(numeric_only=True)

        return DataFrame(query_compiler=self._query_compiler.abs())

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
        return DataFrame(query_compiler=self._query_compiler.isin(values=values))

    def isna(self):
        """Fill a DataFrame with booleans for cells containing NA.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is NA.
            True: cell contains NA.
            False: otherwise.
        """
        return DataFrame(query_compiler=self._query_compiler.isna())

    isnull = isna

    def keys(self):
        """Get the info axis for the DataFrame.

        Returns:
            A pandas Index for this DataFrame.
        """
        return self.columns

    def transpose(self, *args, **kwargs):
        """Transpose columns and rows for the DataFrame.

        Returns:
            A new DataFrame transposed from this DataFrame.
        """
        return DataFrame(query_compiler=self._query_compiler.transpose(*args, **kwargs))

    T = property(transpose)

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
            axis = [pandas.DataFrame()._get_axis_number(ax) for ax in axis]
            result = self

            for ax in axis:
                result = result.dropna(axis=ax, how=how, thresh=thresh, subset=subset)
            return self._create_dataframe_from_compiler(result._query_compiler, inplace)

        axis = pandas.DataFrame()._get_axis_number(axis)
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
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.add,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_or_object_only=True)
        new_query_compiler = self._query_compiler.add(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def agg(self, func, axis=0, *args, **kwargs):
        return self.aggregate(func, axis, *args, **kwargs)

    def aggregate(self, func, axis=0, *args, **kwargs):
        axis = pandas.DataFrame()._get_axis_number(axis)

        result = None

        if axis == 0:
            try:
                result = self._aggregate(func, axis=axis, *args, **kwargs)
            except TypeError:
                pass

        if result is None:
            kwargs.pop("is_transform", None)
            return self.apply(func, axis=axis, args=args, **kwargs)

        return result

    def _aggregate(self, arg, *args, **kwargs):
        _axis = kwargs.pop("_axis", None)
        if _axis is None:
            _axis = getattr(self, "axis", 0)
        kwargs.pop("_level", None)

        if isinstance(arg, string_types):
            return self._string_function(arg, *args, **kwargs)

        # Dictionaries have complex behavior because they can be renamed here.
        elif isinstance(arg, dict):
            return self._default_to_pandas(pandas.DataFrame.agg, arg, *args, **kwargs)
        elif is_list_like(arg) or callable(arg):
            return self.apply(arg, axis=_axis, args=args, **kwargs)
        else:
            # TODO Make pandas error
            raise ValueError("type {} is not callable".format(type(arg)))

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
            return self._default_to_pandas(pandas.DataFrame.agg, func, *args, **kwargs)

        raise ValueError("{} is an unknown string function".format(func))

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
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.align,
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

    def all(self, axis=0, bool_only=None, skipna=None, level=None, **kwargs):
        """Return whether all elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies df.all(axis=1)
                to the transpose of df.
        """
        if axis is not None:
            axis = pandas.DataFrame()._get_axis_number(axis)
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
        return self._query_compiler.all(
            axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
        )

    def any(self, axis=0, bool_only=None, skipna=None, level=None, **kwargs):
        """Return whether any elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies on the column partitions,
                otherwise operates on row partitions
        """
        if axis is not None:
            axis = pandas.DataFrame()._get_axis_number(axis)
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
        return self._query_compiler.any(
            axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
        )

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        """Append another DataFrame/list/Series to this one.

        Args:
            other: The object to append to this.
            ignore_index: Ignore the index on appending.
            verify_integrity: Verify the integrity of the index on completion.

        Returns:
            A new DataFrame containing the concatenated values.
        """
        if isinstance(other, (pandas.Series, dict)):
            if isinstance(other, dict):
                other = pandas.Series(other)
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True"
                    " or if the Series has a name"
                )

            if other.name is None:
                index = None
            else:
                # other must have the same index name as self, otherwise
                # index name will be reset
                index = pandas.Index([other.name], name=self.index.name)

            # Create a Modin DataFrame from this Series for ease of development
            other = DataFrame(pandas.DataFrame(other).T, index=index)._query_compiler
        elif isinstance(other, list):
            if not isinstance(other[0], DataFrame):
                other = pandas.DataFrame(other)
                if (self.columns.get_indexer(other.columns) >= 0).all():
                    other = DataFrame(other.loc[:, self.columns])._query_compiler
                else:
                    other = DataFrame(other)._query_compiler
            else:
                other = [obj._query_compiler for obj in other]
        else:
            other = other._query_compiler

        # If ignore_index is False, by definition the Index will be correct.
        # We also do this first to ensure that we don't waste compute/memory.
        if verify_integrity and not ignore_index:
            appended_index = self.index.append(other.index)
            is_valid = next((False for idx in appended_index.duplicated() if idx), True)
            if not is_valid:
                raise ValueError(
                    "Indexes have overlapping values: {}".format(
                        appended_index[appended_index.duplicated()]
                    )
                )

        query_compiler = self._query_compiler.concat(
            0, other, ignore_index=ignore_index, sort=sort
        )
        return DataFrame(query_compiler=query_compiler)

    def apply(
        self, func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds
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
        axis = pandas.DataFrame()._get_axis_number(axis)
        ErrorMessage.non_verified_udf()

        if isinstance(func, string_types):
            if axis == 1:
                kwds["axis"] = axis
            return getattr(self, func)(*args, **kwds)
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
        elif is_list_like(func):
            if axis == 1:
                raise TypeError(
                    "(\"'list' object is not callable\", "
                    "'occurred at index {0}'".format(self.index[0])
                )
        elif not callable(func):
            return

        query_compiler = self._query_compiler.apply(func, axis, *args, **kwds)
        if isinstance(query_compiler, pandas.Series):
            return query_compiler
        return DataFrame(query_compiler=query_compiler)

    def as_blocks(self, copy=True):
        return self._default_to_pandas(pandas.DataFrame.as_blocks, copy=copy)

    def as_matrix(self, columns=None):
        """Convert the frame to its Numpy-array representation.

        Args:
            columns: If None, return all columns, otherwise,
                returns specified columns.

        Returns:
            values: ndarray
        """
        # TODO this is very inefficient, also see __array__
        return to_pandas(self).as_matrix(columns)

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        return self._default_to_pandas(
            pandas.DataFrame.asfreq,
            freq,
            method=method,
            how=how,
            normalize=normalize,
            fill_value=fill_value,
        )

    def asof(self, where, subset=None):
        return self._default_to_pandas(pandas.DataFrame.asof, where, subset=subset)

    def assign(self, **kwargs):
        return self._default_to_pandas(pandas.DataFrame.assign, **kwargs)

    def astype(self, dtype, copy=True, errors="raise", **kwargs):
        col_dtypes = {}
        if isinstance(dtype, dict):
            if not set(dtype.keys()).issubset(set(self.columns)) and errors == "raise":
                raise KeyError(
                    "Only a column name can be used for the key in"
                    "a dtype mappings argument."
                )
            col_dtypes = dtype

        else:
            for column in self.columns:
                col_dtypes[column] = dtype

        new_query_compiler = self._query_compiler.astype(col_dtypes, **kwargs)
        return self._create_dataframe_from_compiler(new_query_compiler, not copy)

    def at_time(self, time, asof=False):
        return self._default_to_pandas(pandas.DataFrame.at_time, time, asof=asof)

    def between_time(self, start_time, end_time, include_start=True, include_end=True):
        return self._default_to_pandas(
            pandas.DataFrame.between_time,
            start_time,
            end_time,
            include_start=include_start,
            include_end=include_end,
        )

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='bfill')"""
        new_df = self.fillna(
            method="bfill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )
        if not inplace:
            return new_df

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
            return to_pandas(self).bool()

    def boxplot(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize=None,
        rot=0,
        grid=True,
        figsize=None,
        layout=None,
        return_type=None,
        **kwargs
    ):
        return to_pandas(self).boxplot(
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            **kwargs
        )

    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        # validate inputs
        if axis is not None:
            axis = pandas.DataFrame()._get_axis_number(axis)
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
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

    def clip_lower(self, threshold, axis=None, inplace=False):
        return self.clip(lower=threshold, axis=axis, inplace=inplace)

    def clip_upper(self, threshold, axis=None, inplace=False):
        return self.clip(upper=threshold, axis=axis, inplace=inplace)

    def combine(self, other, func, fill_value=None, overwrite=True):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.combine,
            other,
            func,
            fill_value=fill_value,
            overwrite=overwrite,
        )

    def combine_first(self, other):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(pandas.DataFrame.combine_first, other=other)

    def compound(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas(
            pandas.DataFrame.compound, axis=axis, skipna=skipna, level=level
        )

    def consolidate(self, inplace=False):
        return self._default_to_pandas(pandas.DataFrame.consolidate, inplace=inplace)

    def convert_objects(
        self,
        convert_dates=True,
        convert_numeric=False,
        convert_timedeltas=True,
        copy=True,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.convert_objects,
            convert_dates=convert_dates,
            convert_numeric=convert_numeric,
            convert_timedeltas=convert_timedeltas,
            copy=copy,
        )

    def corr(self, method="pearson", min_periods=1):
        return self._default_to_pandas(
            pandas.DataFrame.corr, method=method, min_periods=min_periods
        )

    def corrwith(self, other, axis=0, drop=False):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.corrwith, other, axis=axis, drop=drop
        )

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return self._query_compiler.count(
            axis=axis, level=level, numeric_only=numeric_only
        )

    def cov(self, min_periods=None):
        return self._default_to_pandas(pandas.DataFrame.cov, min_periods=min_periods)

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative maximum across the DataFrame.

        Args:
            axis (int): The axis to take maximum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative maximum of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if axis:
            self._validate_dtypes()
        return DataFrame(
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if axis:
            self._validate_dtypes()
        return DataFrame(
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes(numeric_only=True)
        return DataFrame(
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes(numeric_only=True)
        return DataFrame(
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
        if include is not None:
            if not is_list_like(include):
                include = [include]
            include = [np.dtype(i) for i in include]
        if exclude is not None:
            if not is_list_like(include):
                exclude = [exclude]
            exclude = [np.dtype(e) for e in exclude]
        if percentiles is not None:
            pandas.DataFrame()._check_percentile(percentiles)
        return DataFrame(
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
        axis = pandas.DataFrame()._get_axis_number(axis)
        return DataFrame(
            query_compiler=self._query_compiler.diff(periods=periods, axis=axis)
        )

    def div(self, other, axis="columns", level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.div,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.div(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def divide(self, other, axis="columns", level=None, fill_value=None):
        """Synonym for div.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self.div(other, axis, level, fill_value)

    def dot(self, other):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(pandas.DataFrame.dot, other)

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
                pandas.DataFrame.drop,
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
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

    def drop_duplicates(self, subset=None, keep="first", inplace=False):
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
        duplicates = self.duplicated(subset=subset, keep=keep)
        indices, = duplicates.nonzero()
        return self.drop(indices, inplace=inplace)

    def duplicated(self, subset=None, keep="first"):
        return self._default_to_pandas(
            pandas.DataFrame.duplicated, subset=subset, keep=keep
        )

    def eq(self, other, axis="columns", level=None):
        """Checks element-wise that this is equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the eq over.
            level: The Multilevel index level to apply eq over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.eq, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis)
        new_query_compiler = self._query_compiler.eq(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def equals(self, other):
        """
        Checks if other DataFrame is elementwise equal to the current one

        Returns:
            Boolean: True if equal, otherwise False
        """
        if isinstance(other, pandas.DataFrame):
            # Copy into a Ray DataFrame to simplify logic below
            other = DataFrame(other)
        if not self.index.equals(other.index) or not self.columns.equals(other.columns):
            return False

        return all(self.eq(other).all())

    def eval(self, expr, inplace=False, **kwargs):
        """Evaluate a Python expression as a string using various backends.
        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.

            parser: The parser to use to construct the syntax tree from the
                expression. The default of 'pandas' parses code slightly
                different than standard Python. Alternatively, you can parse
                an expression using the 'python' parser to retain strict
                Python semantics. See the enhancing performance documentation
                for more details.

            engine: The engine used to evaluate the expression.

            truediv: Whether to use true division, like in Python >= 3

            local_dict: A dictionary of local variables, taken from locals()
                by default.

            global_dict: A dictionary of global variables, taken from
                globals() by default.

            resolvers: A list of objects implementing the __getitem__ special
                method that you can use to inject an additional collection
                of namespaces to use for variable lookup. For example, this is
                used in the query() method to inject the index and columns
                variables that refer to their respective DataFrame instance
                attributes.

            level: The number of prior stack frames to traverse and add to
                the current scope. Most users will not need to change this
                parameter.

            target: This is the target object for assignment. It is used when
                there is variable assignment in the expression. If so, then
                target must support item assignment with string keys, and if a
                copy is being returned, it must also support .copy().

            inplace: If target is provided, and the expression mutates target,
                whether to modify target inplace. Otherwise, return a copy of
                target with the mutation.
        Returns:
            ndarray, numeric scalar, DataFrame, Series
        """
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.eval(expr, **kwargs)

        if isinstance(new_query_compiler, pandas.Series):
            return new_query_compiler
        else:
            return self._create_dataframe_from_compiler(new_query_compiler, inplace)

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
            pandas.DataFrame.ewm,
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
            pandas.DataFrame.expanding,
            min_periods=min_periods,
            center=center,
            axis=axis,
        )

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='ffill')
        """
        new_df = self.fillna(
            method="ffill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )
        if not inplace:
            return new_df

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
        # TODO implement value passed as DataFrame
        if isinstance(value, pandas.DataFrame) or isinstance(value, pandas.Series):
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.fillna,
                value=value,
                method=method,
                axis=axis,
                inplace=False,
                limit=limit,
                downcast=downcast,
                **kwargs
            )._query_compiler
            return self._create_dataframe_from_compiler(new_query_compiler, inplace)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

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

        new_query_compiler = self._query_compiler.fillna(
            value=value,
            method=method,
            axis=axis,
            inplace=False,
            limit=limit,
            downcast=downcast,
            **kwargs
        )
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

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
        nkw = com._count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` "
                "are mutually exclusive"
            )
        if nkw == 0:
            raise TypeError("Must pass either `items`, `like`, or `regex`")
        if axis is None:
            axis = "columns"  # This is the default info axis for dataframes

        axis = pandas.DataFrame()._get_axis_number(axis)
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
        return self._default_to_pandas(pandas.DataFrame.first, offset)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.floordiv,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.floordiv(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    @classmethod
    def from_csv(
        cls,
        path,
        header=0,
        sep=", ",
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

    @classmethod
    def from_dict(cls, data, orient="columns", dtype=None):
        ErrorMessage.default_to_pandas("`from_dict`")
        return from_pandas(pandas.DataFrame.from_dict(data, orient=orient, dtype=dtype))

    @classmethod
    def from_items(cls, items, columns=None, orient="columns"):
        ErrorMessage.default_to_pandas("`from_items`")
        return from_pandas(
            pandas.DataFrame.from_items(items, columns=columns, orient=orient)
        )

    @classmethod
    def from_records(
        cls,
        data,
        index=None,
        exclude=None,
        columns=None,
        coerce_float=False,
        nrows=None,
    ):
        ErrorMessage.default_to_pandas("`from_records`")
        return from_pandas(
            pandas.DataFrame.from_records(
                data,
                index=index,
                exclude=exclude,
                columns=columns,
                coerce_float=coerce_float,
                nrows=nrows,
            )
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
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.ge, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis, comparison_dtypes_only=True)
        new_query_compiler = self._query_compiler.ge(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

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
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def get_dtype_counts(self):
        """Get the counts of dtypes in this object.

        Returns:
            The counts of dtypes in this object.
        """
        result = self.dtypes.value_counts()
        result.index = result.index.map(lambda x: str(x))
        return result

    def get_ftype_counts(self):
        """Get the counts of ftypes in this object.

        Returns:
            The counts of ftypes in this object.
        """
        return self.ftypes.value_counts().sort_index()

    def get_value(self, index, col, takeable=False):
        return self._default_to_pandas(
            pandas.DataFrame.get_value, index, col, takeable=takeable
        )

    def get_values(self):
        return self._default_to_pandas(pandas.DataFrame.get_values)

    def gt(self, other, axis="columns", level=None):
        """Checks element-wise that this is greater than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.gt, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis, comparison_dtypes_only=True)
        new_query_compiler = self._query_compiler.gt(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def head(self, n=5):
        """Get the first n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the first n rows of the DataFrame.
        """
        if n >= len(self.index):
            return self.copy()
        return DataFrame(query_compiler=self._query_compiler.head(n))

    def hist(
        self,
        column=None,
        by=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        yrot=None,
        ax=None,
        sharex=False,
        sharey=False,
        figsize=None,
        layout=None,
        bins=10,
        **kwargs
    ):
        return self._default_to_pandas(
            pandas.DataFrame.hist,
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            **kwargs
        )

    def idxmax(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the max value of the axis.

        Args:
            axis (int): Identify the max over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each maximum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self.dtypes):
            raise TypeError("reduction operation 'argmax' not allowed for this dtype")
        return self._query_compiler.idxmax(axis=axis, skipna=skipna)

    def idxmin(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the min value of the axis.

        Args:
            axis (int): Identify the min over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each minimum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self.dtypes):
            raise TypeError("reduction operation 'argmax' not allowed for this dtype")
        return self._query_compiler.idxmin(axis=axis, skipna=skipna)

    def infer_objects(self):
        return self._default_to_pandas(pandas.DataFrame.infer_objects)

    def info(
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None
    ):
        """Print a concise summary of a DataFrame, which includes the index
        dtype and column dtypes, non-null values and memory usage.

        Args:
            verbose (bool, optional): Whether to print the full summary. Defaults
                to true

            buf (writable buffer): Where to send output. Defaults to sys.stdout

            max_cols (int, optional): When to switch from verbose to truncated
                output. By defualt, this is 100.

            memory_usage (bool, str, optional): Specifies whether the total memory
                usage of the DataFrame elements (including index) should be displayed.
                True always show memory usage. False never shows memory usage. A value
                of 'deep' is equivalent to "True with deep introspection". Memory usage
                is shown in human-readable units (base-2 representation). Without deep
                introspection a memory estimation is made based in column dtype and
                number of rows assuming values consume the same memory amount for
                corresponding dtypes. With deep memory introspection, a real memory
                usage calculation is performed at the cost of computational resources.
                Defaults to True.

            null_counts (bool, optional): Whetehr to show the non-null counts. By
                default, this is shown only when the frame is smaller than 100 columns
                and 1690785 rows. A value of True always shows the counts and False
                never shows the counts.

        Returns:
            Prints the summary of a DataFrame and returns None.
        """
        # We will default to pandas because it will be faster than doing two passes
        # over the data
        buf = sys.stdout if not buf else buf
        import io

        with io.StringIO() as tmp_buf:
            self._default_to_pandas(
                pandas.DataFrame.info,
                verbose=verbose,
                buf=tmp_buf,
                max_cols=max_cols,
                memory_usage=memory_usage,
                null_counts=null_counts,
            )
            result = tmp_buf.getvalue()
            result = result.replace(
                "pandas.core.frame.DataFrame", "modin.pandas.dataframe.DataFrame"
            )
            buf.write(result)
        return None

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert column into DataFrame at specified location.
        Args:
            loc (int): Insertion index. Must verify 0 <= loc <= len(columns).
            column (hashable object): Label of the inserted column.
            value (int, Series, or array-like): The values to insert.
            allow_duplicates (bool): Whether to allow duplicate column names.
        """
        if isinstance(value, (DataFrame, pandas.DataFrame)):
            if len(value.columns) != 1:
                raise ValueError("Wrong number of items passed 2, placement implies 1")
            value = value.iloc[:, 0]
        if len(self.index) == 0:
            try:
                value = pandas.Series(value)
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    "Cannot insert into a DataFrame with no defined index "
                    "and a value that cannot be converted to a "
                    "Series"
                )
            new_index = value.index.copy()
            new_columns = self.columns.insert(loc, column)
            new_query_compiler = DataFrame(
                value, index=new_index, columns=new_columns
            )._query_compiler
        elif len(self.columns) == 0 and loc == 0:
            new_query_compiler = DataFrame(
                data=value, columns=[column], index=self.index
            )._query_compiler
        else:
            if not is_list_like(value):
                value = np.full(len(self.index), value)
            if not isinstance(value, pandas.Series) and len(value) != len(self.index):
                raise ValueError("Length of values does not match length of index")
            if not allow_duplicates and column in self.columns:
                raise ValueError("cannot insert {0}, already exists".format(column))
            if loc > len(self.columns):
                raise IndexError(
                    "index {0} is out of bounds for axis 0 with size {1}".format(
                        loc, len(self.columns)
                    )
                )
            if loc < 0:
                raise ValueError("unbounded slice")
            new_query_compiler = self._query_compiler.insert(loc, column, value)

        self._update_inplace(new_query_compiler=new_query_compiler)

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction="forward",
        downcast=None,
        **kwargs
    ):
        return self._default_to_pandas(
            pandas.DataFrame.interpolate,
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            downcast=downcast,
            **kwargs
        )

    def iterrows(self):
        """Iterate over DataFrame rows as (index, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the rows of the frame.
        """
        index_iter = iter(self.index)

        def iterrow_builder(df):
            df.columns = self.columns
            df.index = [next(index_iter)]
            return df.iterrows()

        partition_iterator = PartitionIterator(self._query_compiler, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the columns of the frame.
        """
        col_iter = iter(self.columns)

        def items_builder(df):
            df.columns = [next(col_iter)]
            df.index = self.index
            return df.items()

        partition_iterator = PartitionIterator(self._query_compiler, 1, items_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Returns the same thing as .items()

        Returns:
            A generator that iterates over the columns of the frame.
        """
        return self.items()

    def itertuples(self, index=True, name="Pandas"):
        """Iterate over DataFrame rows as namedtuples.

        Args:
            index (boolean, default True): If True, return the index as the
                first element of the tuple.
            name (string, default "Pandas"): The name of the returned
            namedtuples or None to return regular tuples.
        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A tuple representing row data. See args for varying tuples.
        """
        index_iter = iter(self.index)

        def itertuples_builder(df):
            df.columns = self.columns
            df.index = [next(index_iter)]
            return df.itertuples(index=index, name=name)

        partition_iterator = PartitionIterator(
            self._query_compiler, 0, itertuples_builder
        )
        for v in partition_iterator:
            yield v

    def join(self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):
        """Join two or more DataFrames, or a DataFrame with a collection.

        Args:
            other: What to join this DataFrame with.
            on: A column name to use from the left for the join.
            how: What type of join to conduct.
            lsuffix: The suffix to add to column names that match on left.
            rsuffix: The suffix to add to column names that match on right.
            sort: Whether or not to sort.

        Returns:
            The joined DataFrame.
        """

        if on is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.join,
                other,
                on=on,
                how=how,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                sort=sort,
            )
        if isinstance(other, pandas.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = DataFrame({other.name: other})
        if isinstance(other, DataFrame):
            # Joining the empty DataFrames with either index or columns is
            # fast. It gives us proper error checking for the edge cases that
            # would otherwise require a lot more logic.
            pandas.DataFrame(columns=self.columns).join(
                pandas.DataFrame(columns=other.columns),
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            ).columns

            return DataFrame(
                query_compiler=self._query_compiler.join(
                    other._query_compiler,
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
                )
            )
        else:
            # This constraint carried over from Pandas.
            if on is not None:
                raise ValueError(
                    "Joining multiple DataFrames only supported for joining on index"
                )
            # See note above about error checking with an empty join.
            pandas.DataFrame(columns=self.columns).join(
                [pandas.DataFrame(columns=obj.columns) for obj in other],
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            ).columns

            return DataFrame(
                query_compiler=self._query_compiler.join(
                    [obj._query_compiler for obj in other],
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
                )
            )

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._default_to_pandas(
            pandas.DataFrame.kurt,
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs
        )

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._default_to_pandas(
            pandas.DataFrame.kurtosis,
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs
        )

    def last(self, offset):
        return self._default_to_pandas(pandas.DataFrame.last, offset)

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
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.le, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis, comparison_dtypes_only=True)
        new_query_compiler = self._query_compiler.le(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def lookup(self, row_labels, col_labels):
        return self._default_to_pandas(pandas.DataFrame.lookup, row_labels, col_labels)

    def lt(self, other, axis="columns", level=None):
        """Checks element-wise that this is less than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the lt over.
            level: The Multilevel index level to apply lt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.lt, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis, comparison_dtypes_only=True)
        new_query_compiler = self._query_compiler.lt(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def mad(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas(
            pandas.DataFrame.mad, axis=axis, skipna=skipna, level=level
        )

    def mask(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        raise_on_error=None,
    ):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.mask,
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes_min_max(axis, numeric_only)

        return self._query_compiler.max(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Computes mean across the DataFrame.

        Args:
            axis (int): The axis to take the mean on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The mean of the DataFrame. (Pandas series)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=False)

        return self._query_compiler.mean(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Computes median across the DataFrame.

        Args:
            axis (int): The axis to take the median on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The median of the DataFrame. (Pandas series)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)

        return self._query_compiler.median(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.melt,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
            col_level=col_level,
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
        result = self._query_compiler.memory_usage(index=index, deep=deep)
        result.index = self.columns
        if index:
            index_value = self.index.memory_usage(deep=deep)
            return pandas.Series(index_value, index=["Index"]).append(result)

        return result

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    ):
        """Database style join, where common columns in "on" are merged.

        Args:
            right: The DataFrame to merge against.
            how: What type of join to use.
            on: The common column name(s) to join on. If None, and left_on and
                right_on  are also None, will default to all commonly named
                columns.
            left_on: The column(s) on the left to use for the join.
            right_on: The column(s) on the right to use for the join.
            left_index: Use the index from the left as the join keys.
            right_index: Use the index from the right as the join keys.
            sort: Sort the join keys lexicographically in the result.
            suffixes: Add this suffix to the common names not in the "on".
            copy: Does nothing in our implementation
            indicator: Adds a column named _merge to the DataFrame with
                metadata from the merge about each row.
            validate: Checks if merge is a specific type.

        Returns:
             A merged Dataframe
        """

        if not isinstance(right, DataFrame):
            raise ValueError(
                "can not merge DataFrame with instance of type "
                "{}".format(type(right))
            )
        if left_index is False or right_index is False:
            if isinstance(right, DataFrame):
                right = right._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.merge,
                right,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                sort=sort,
                suffixes=suffixes,
                copy=copy,
                indicator=indicator,
                validate=validate,
            )
        if left_index and right_index:
            return self.join(
                right, how=how, lsuffix=suffixes[0], rsuffix=suffixes[1], sort=sort
            )

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Perform min across the DataFrame.

        Args:
            axis (int): The axis to take the min on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The min of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes_min_max(axis, numeric_only)

        return self._query_compiler.min(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.mod,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.mod(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def mode(self, axis=0, numeric_only=False):
        """Perform mode across the DataFrame.

        Args:
            axis (int): The axis to take the mode on.
            numeric_only (bool): if True, only apply to numeric columns.

        Returns:
            DataFrame: The mode of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis)
        return DataFrame(
            query_compiler=self._query_compiler.mode(
                axis=axis, numeric_only=numeric_only
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.mul,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.mul(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def multiply(self, other, axis="columns", level=None, fill_value=None):
        """Synonym for mul.

        Args:
            other: The object to use to apply the multiply against this.
            axis: The axis to multiply over.
            level: The Multilevel index level to apply multiply over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Multiply applied.
        """
        return self.mul(other, axis, level, fill_value)

    def ne(self, other, axis="columns", level=None):
        """Checks element-wise that this is not equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the ne over.
            level: The Multilevel index level to apply ne over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.ne, other, axis=axis, level=level
            )
        other = self._validate_other(other, axis)
        new_query_compiler = self._query_compiler.ne(
            other=other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def nlargest(self, n, columns, keep="first"):
        return self._default_to_pandas(pandas.DataFrame.nlargest, n, columns, keep=keep)

    def notna(self):
        """Perform notna across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return DataFrame(query_compiler=self._query_compiler.notna())

    def notnull(self):
        """Perform notnull across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return DataFrame(query_compiler=self._query_compiler.notnull())

    def nsmallest(self, n, columns, keep="first"):
        return self._default_to_pandas(
            pandas.DataFrame.nsmallest, n, columns, keep=keep
        )

    def nunique(self, axis=0, dropna=True):
        """Return Series with number of distinct
           observations over requested axis.

        Args:
            axis : {0 or 'index', 1 or 'columns'}, default 0
            dropna : boolean, default True

        Returns:
            nunique : Series
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return self._query_compiler.nunique(axis=axis, dropna=dropna)

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, **kwargs):
        return self._default_to_pandas(
            pandas.DataFrame.pct_change,
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
        return com._pipe(self, func, *args, **kwargs)

    def pivot(self, index=None, columns=None, values=None):
        return self._default_to_pandas(
            pandas.DataFrame.pivot, index=index, columns=columns, values=values
        )

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
    ):
        return self._default_to_pandas(
            pandas.DataFrame.pivot_table,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
        )

    @property
    def plot(
        self,
        x=None,
        y=None,
        kind="line",
        ax=None,
        subplots=False,
        sharex=None,
        sharey=False,
        layout=None,
        figsize=None,
        use_index=True,
        title=None,
        grid=None,
        legend=True,
        style=None,
        logx=False,
        logy=False,
        loglog=False,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        rot=None,
        fontsize=None,
        colormap=None,
        table=False,
        yerr=None,
        xerr=None,
        secondary_y=False,
        sort_columns=False,
        **kwargs
    ):
        return to_pandas(self).plot

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.pow,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.pow(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=1,
        **kwargs
    ):
        """Return the product of the values for the requested axis

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            level : int or level name, default None
            numeric_only : boolean, default None
            min_count : int, default 1

        Returns:
            prod : Series or DataFrame (if level specified)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=True)
        return self._query_compiler.prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    def product(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=1,
        **kwargs
    ):
        """Return the product of the values for the requested axis

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            level : int or level name, default None
            numeric_only : boolean, default None
            min_count : int, default 1

        Returns:
            product : Series or DataFrame (if level specified)
        """
        return self.prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        def check_dtype(t):
            return is_numeric_dtype(t) or is_datetime_or_timedelta_dtype(t)

        if not numeric_only:
            # If not numeric_only and columns, then check all columns are either
            # numeric, timestamp, or timedelta
            if not axis and not all(check_dtype(t) for t in self.dtypes):
                raise TypeError("can't multiply sequence by non-int of type 'float'")
            # If over rows, then make sure that all dtypes are equal for not
            # numeric_only
            elif axis:
                for i in range(1, len(self.dtypes)):
                    pre_dtype = self.dtypes[i - 1]
                    curr_dtype = self.dtypes[i]
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
            if not any(is_numeric_dtype(t) for t in self.dtypes):
                raise ValueError("need at least one array to concatenate")

        # check that all qs are between 0 and 1
        pandas.DataFrame()._check_percentile(q)
        axis = pandas.DataFrame()._get_axis_number(axis)

        if isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list)):
            return DataFrame(
                query_compiler=self._query_compiler.quantile_for_list_of_values(
                    q=q,
                    axis=axis,
                    numeric_only=numeric_only,
                    interpolation=interpolation,
                )
            )
        else:
            return self._query_compiler.quantile_for_single_value(
                q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation
            )

    def query(self, expr, inplace=False, **kwargs):
        """Queries the Dataframe with a boolean expression

        Returns:
            A new DataFrame if inplace=False
        """
        ErrorMessage.non_verified_udf()
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.query(expr, **kwargs)
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

    def radd(self, other, axis="columns", level=None, fill_value=None):
        return self.add(other, axis, level, fill_value)

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
        axis = pandas.DataFrame()._get_axis_number(axis)
        return DataFrame(
            query_compiler=self._query_compiler.rank(
                axis=axis,
                method=method,
                numeric_only=numeric_only,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )

    def rdiv(self, other, axis="columns", level=None, fill_value=None):
        """Div this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the div against this.
            axis: The axis to div over.
            level: The Multilevel index level to apply div over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the rdiv applied.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.rdiv,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.rdiv(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

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
        if (
            level is not None
            or (
                isinstance(self.columns, pandas.MultiIndex)
                and (columns is not None or axis == 1)
            )
            or (
                isinstance(self.index, pandas.MultiIndex)
                and (index is not None or axis == 0)
            )
        ):
            return self._default_to_pandas(
                pandas.DataFrame.reindex,
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels
        if index is not None:
            new_query_compiler = self._query_compiler.reindex(
                0,
                index,
                method=method,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )
        else:
            new_query_compiler = self._query_compiler
        if columns is not None:
            final_query_compiler = new_query_compiler.reindex(
                1,
                columns,
                method=method,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )
        else:
            final_query_compiler = new_query_compiler
        return self._create_dataframe_from_compiler(final_query_compiler, not copy)

    def reindex_axis(
        self,
        labels,
        axis=0,
        method=None,
        level=None,
        copy=True,
        limit=None,
        fill_value=np.nan,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.reindex_axis,
            labels,
            axis=axis,
            method=method,
            level=level,
            copy=copy,
            limit=limit,
            fill_value=fill_value,
        )

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.reindex_like,
            other,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
    ):
        """Alters axes labels.

        Args:
            mapper, index, columns: Transformations to apply to the axis's
                values.
            axis: Axis to target with mapper.
            copy: Also copy underlying data.
            inplace: Whether to return a new DataFrame.
            level: Only rename a specific level of a MultiIndex.

        Returns:
            If inplace is False, a new DataFrame with the updated axes.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # We have to do this with the args because of how rename handles
        # kwargs. It doesn't ignore None values passed in, so we have to filter
        # them ourselves.
        args = locals()
        kwargs = {k: v for k, v in args.items() if v is not None and k != "self"}
        # inplace should always be true because this is just a copy, and we
        # will use the results after.
        kwargs["inplace"] = True
        df_to_rename = pandas.DataFrame(index=self.index, columns=self.columns)
        df_to_rename.rename(**kwargs)

        if inplace:
            obj = self
        else:
            obj = self.copy()
        obj.index = df_to_rename.index
        obj.columns = df_to_rename.columns

        if not inplace:
            return obj

    def rename_axis(self, mapper, axis=0, copy=True, inplace=False):
        axes_is_columns = axis == 1 or axis == "columns"
        renamed = self if inplace else self.copy()
        if axes_is_columns:
            renamed.columns.name = mapper
        else:
            renamed.index.name = mapper
        if not inplace:
            return renamed

    def _set_axis_name(self, name, axis=0, inplace=False):
        """Alter the name or names of the axis.

        Args:
            name: Name for the Index, or list of names for the MultiIndex
            axis: 0 or 'index' for the index; 1 or 'columns' for the columns
            inplace: Whether to modify `self` directly or return a copy

        Returns:
            Type of caller or None if inplace=True.
        """
        axes_is_columns = axis == 1 or axis == "columns"
        renamed = self if inplace else self.copy()
        if axes_is_columns:
            renamed.columns.set_names(name)
        else:
            renamed.index.set_names(name)
        if not inplace:
            return renamed

    def reorder_levels(self, order, axis=0):
        return self._default_to_pandas(
            pandas.DataFrame.reorder_levels, order, axis=axis
        )

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
            pandas.DataFrame.replace,
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
            pandas.DataFrame.resample,
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
        if level is not None:
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.reset_index,
                level=level,
                drop=drop,
                inplace=inplace,
                col_level=col_level,
                col_fill=col_fill,
            )
            return self._create_dataframe_from_compiler(new_query_compiler, inplace)
        # Error checking for matching Pandas. Pandas does not allow you to
        # insert a dropped index into a DataFrame if these columns already
        # exist.
        if (
            not drop
            and not isinstance(self.index, pandas.MultiIndex)
            and all(n in self.columns for n in ["level_0", "index"])
        ):
            raise ValueError("cannot insert level_0, already exists")
        new_query_compiler = self._query_compiler.reset_index(drop=drop, level=level)
        return self._create_dataframe_from_compiler(new_query_compiler, inplace)

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.rfloordiv,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.rfloordiv(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.rmod,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.rmod(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def rmul(self, other, axis="columns", level=None, fill_value=None):
        return self.mul(other, axis, level, fill_value)

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
            pandas.DataFrame.rolling,
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
        return DataFrame(
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.rpow,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        # Check to make sure integers are not raised to negative integer powers
        if (
            is_integer_dtype(type(other))
            and other < 0
            and all(is_integer_dtype(t) for t in self.dtypes)
        ):
            raise ValueError("Integers to negative integer powers are not allowed.")
        new_query_compiler = self._query_compiler.rpow(
            other=other, axis=axis, level=level, fill_value=fill_value
        )

        return self._create_dataframe_from_compiler(new_query_compiler)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.rsub,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_or_time_only=True)
        new_query_compiler = self._query_compiler.rsub(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        return self.rdiv(other, axis, level, fill_value)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
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
            if isinstance(weights, pandas.Series):
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
            # An Empty DataFrame is returned if the number of samples is 0.
            # The Empty Dataframe should have either columns or index specified
            # depending on which axis is passed in.
            return DataFrame(
                columns=[] if axis == 1 else self.columns,
                index=self.index if axis == 1 else [],
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
            return DataFrame(query_compiler=query_compiler)
        else:
            query_compiler = self._query_compiler.getitem_row_array(samples)
            return DataFrame(query_compiler=query_compiler)

    def select(self, crit, axis=0):
        return self._default_to_pandas(pandas.DataFrame.select, crit, axis=axis)

    def select_dtypes(self, include=None, exclude=None):
        # Validates arguments for whether both include and exclude are None or
        # if they are disjoint. Also invalidates string dtypes.
        pandas.DataFrame().select_dtypes(include, exclude)

        if include and not is_list_like(include):
            include = [include]
        elif include is None:
            include = []
        if exclude and not is_list_like(exclude):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        sel = tuple(map(set, (include, exclude)))
        include, exclude = map(lambda x: set(map(_get_dtype_from_object, x)), sel)
        include_these = pandas.Series(not bool(include), index=self.columns)
        exclude_these = pandas.Series(not bool(exclude), index=self.columns)

        def is_dtype_instance_mapper(column, dtype):
            return column, functools.partial(issubclass, dtype.type)

        for column, f in itertools.starmap(
            is_dtype_instance_mapper, self.dtypes.iteritems()
        ):
            if include:  # checks for the case of empty include or exclude
                include_these[column] = any(map(f, include))
            if exclude:
                exclude_these[column] = not any(map(f, exclude))

        dtype_indexer = include_these & exclude_these
        indicate = [
            i for i in range(len(dtype_indexer.values)) if not dtype_indexer.values[i]
        ]
        return self.drop(columns=self.columns[indicate], inplace=False)

    def sem(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._default_to_pandas(
            pandas.DataFrame.sem,
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs
        )

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

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ):
        """Set the DataFrame index using one or more existing columns.

        Args:
            keys: column label or list of column labels / arrays.
            drop (boolean): Delete columns to be used as the new index.
            append (boolean): Whether to append columns to existing index.
            inplace (boolean): Modify the DataFrame in place.
            verify_integrity (boolean): Check the new index for duplicates.
                Otherwise defer the check until necessary. Setting to False
                will improve the performance of this method

        Returns:
            If inplace is set to false returns a new DataFrame, otherwise None.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]
        if inplace:
            frame = self
        else:
            frame = self.copy()

        arrays = []
        names = []
        if append:
            names = [x for x in self.index.names]
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)
        to_remove = []
        for col in keys:
            if isinstance(col, pandas.MultiIndex):
                # append all but the last column so we don't have to modify
                # the end of this loop
                for n in range(col.nlevels - 1):
                    arrays.append(col._get_level_values(n))

                level = col._get_level_values(col.nlevels - 1)
                names.extend(col.names)
            elif isinstance(col, pandas.Series):
                level = col._values
                names.append(col.name)
            elif isinstance(col, pandas.Index):
                level = col
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray, pandas.Index)):
                level = col
                names.append(None)
            else:
                level = frame[col]._values
                names.append(col)
                if drop:
                    to_remove.append(col)
            arrays.append(level)
        index = _ensure_index_from_sequences(arrays, names)

        if verify_integrity and not index.is_unique:
            duplicates = index.get_duplicates()
            raise ValueError("Index has duplicate keys: %s" % duplicates)

        for c in to_remove:
            del frame[c]
        # clear up memory usage
        index._cleanup()
        frame.index = index

        if not inplace:
            return frame

    def set_value(self, index, col, value, takeable=False):
        return self._default_to_pandas(
            pandas.DataFrame.set_value, index, col, value, takeable=takeable
        )

    def shift(self, periods=1, freq=None, axis=0):
        return self._default_to_pandas(
            pandas.DataFrame.shift, periods=periods, freq=freq, axis=axis
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)

        return self._query_compiler.skew(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def slice_shift(self, periods=1, axis=0):
        return self._default_to_pandas(
            pandas.DataFrame.slice_shift, periods=periods, axis=axis
        )

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
        axis = pandas.DataFrame()._get_axis_number(axis)
        if level is not None:
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.sort_index,
                axis=axis,
                level=level,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
            )
            return self._create_dataframe_from_compiler(new_query_compiler, inplace)
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
            return DataFrame(query_compiler=new_query_compiler)

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
        axis = pandas.DataFrame()._get_axis_number(axis)
        if not is_list_like(by):
            by = [by]
        # Currently, sort_values will just reindex based on the sorted values.
        # TODO create a more efficient way to sort
        if axis == 0:
            broadcast_value_dict = {col: self[col] for col in by}
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
                to_pandas(self[row :: len(self.index)]) for row in by
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

    def sortlevel(
        self, level=0, axis=0, ascending=True, inplace=False, sort_remaining=True
    ):
        return self._default_to_pandas(
            pandas.DataFrame.sortlevel,
            level=level,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            sort_remaining=sort_remaining,
        )

    def squeeze(self, axis=None):
        # Checks for 1x1 DF, passes into squeeze with approproate ndim
        if (
            self._query_compiler.data.shape[0] == 1
            and self._query_compiler.data.shape[1] == 1
        ):
            return self._query_compiler.squeeze(0, axis)
        # Checks for 1xN or Nx1 DF, passes into squeeze with appropriate ndim
        elif 1 in self._query_compiler.data.shape:
            return self._query_compiler.squeeze(1, axis)
        # NxN DF, don't need to pass into squeeze
        else:
            return self.copy()

    def stack(self, level=-1, dropna=True):
        return self._default_to_pandas(
            pandas.DataFrame.stack, level=level, dropna=dropna
        )

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)

        return self._query_compiler.std(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs
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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.sub,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_or_time_only=True)
        new_query_compiler = self._query_compiler.sub(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def subtract(self, other, axis="columns", level=None, fill_value=None):
        """Alias for sub.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: THe axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self.sub(other, axis, level, fill_value)

    def swapaxes(self, axis1, axis2, copy=True):
        return self._default_to_pandas(
            pandas.DataFrame.swapaxes, axis1, axis2, copy=copy
        )

    def swaplevel(self, i=-2, j=-1, axis=0):
        return self._default_to_pandas(pandas.DataFrame.swaplevel, i=i, j=j, axis=axis)

    def tail(self, n=5):
        """Get the last n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the last n rows of this DataFrame.
        """
        if n >= len(self.index):
            return self.copy()
        return DataFrame(query_compiler=self._query_compiler.tail(n))

    def take(self, indices, axis=0, convert=None, is_copy=True, **kwargs):
        return self._default_to_pandas(
            pandas.DataFrame.take,
            indices,
            axis=axis,
            convert=convert,
            is_copy=is_copy,
            **kwargs
        )

    def to_clipboard(self, excel=None, sep=None, **kwargs):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_clipboard, excel=excel, sep=sep, **kwargs
        )

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
        compression=None,
        quoting=None,
        quotechar='"',
        line_terminator="\n",
        chunksize=None,
        tupleize_cols=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
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
        return self._default_to_pandas(pandas.DataFrame.to_csv, **kwargs)

    def to_dense(self):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.to_dense)

    def to_dict(self, orient="dict", into=dict):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_dict, orient=orient, into=into
        )

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
            pandas.DataFrame.to_excel,
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

    def to_feather(self, fname):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.to_feather, fname)

    def to_gbq(
        self,
        destination_table,
        project_id,
        chunksize=10000,
        verbose=True,
        reauth=False,
        if_exists="fail",
        private_key=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_gbq,
            destination_table,
            project_id,
            chunksize=chunksize,
            verbose=verbose,
            reauth=reauth,
            if_exists=if_exists,
            private_key=private_key,
        )

    def to_hdf(self, path_or_buf, key, format="table", **kwargs):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_hdf, path_or_buf, key, format=format, **kwargs
        )

    def to_html(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="np.NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        bold_rows=True,
        classes=None,
        escape=True,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        notebook=False,
        decimal=".",
        border=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_html,
            buf,
            columns,
            col_space,
            header,
            index,
            na_rep,
            formatters,
            float_format,
            sparsify,
            index_names,
            justify,
            bold_rows,
            classes,
            escape,
            max_rows,
            max_cols,
            show_dimensions,
            notebook,
            decimal,
            border,
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
        compression=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_json,
            path_or_buf,
            orient,
            date_format,
            double_precision,
            force_ascii,
            date_unit,
            default_handler,
            lines,
            compression,
        )

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="np.NaN",
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
            pandas.DataFrame.to_latex,
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
            pandas.DataFrame.to_msgpack,
            path_or_buf=path_or_buf,
            encoding=encoding,
            **kwargs
        )

    def to_panel(self):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.to_panel)

    def to_parquet(
        self, fname, engine="auto", compression="snappy", **kwargs
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_parquet,
            fname,
            engine=engine,
            compression=compression,
            **kwargs
        )

    def to_period(self, freq=None, axis=0, copy=True):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_period, freq=freq, axis=axis, copy=copy
        )

    def to_pickle(
        self, path, compression="infer", protocol=pkl.HIGHEST_PROTOCOL
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_pickle, path, compression=compression, protocol=protocol
        )

    def to_records(self, index=True, convert_datetime64=True):
        return self._default_to_pandas(
            pandas.DataFrame.to_records,
            index=index,
            convert_datetime64=convert_datetime64,
        )

    def to_sparse(self, fill_value=None, kind="block"):
        return self._default_to_pandas(
            pandas.DataFrame.to_sparse, fill_value=fill_value, kind=kind
        )

    def to_sql(
        self,
        name,
        con,
        flavor=None,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_sql,
            name,
            con,
            flavor,
            schema,
            if_exists,
            index,
            index_label,
            chunksize,
            dtype,
        )

    def to_stata(
        self,
        fname,
        convert_dates=None,
        write_index=True,
        encoding="latin-1",
        byteorder=None,
        time_stamp=None,
        data_label=None,
        variable_labels=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_stata,
            fname,
            convert_dates,
            write_index,
            encoding,
            byteorder,
            time_stamp,
            data_label,
            variable_labels,
        )

    def to_string(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="np.NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        line_width=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.to_string,
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
            line_width=line_width,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
        )

    def to_timestamp(self, freq=None, how="start", axis=0, copy=True):
        return self._default_to_pandas(
            pandas.DataFrame.to_timestamp, freq=freq, how=how, axis=axis, copy=copy
        )

    def to_xarray(self):
        return self._default_to_pandas(pandas.DataFrame.to_xarray)

    def transform(self, func, *args, **kwargs):
        kwargs["is_transform"] = True
        result = self.agg(func, *args, **kwargs)
        try:
            result.columns = self.columns
            result.index = self.index
        except ValueError:
            raise ValueError("transforms cannot produce aggregated results")
        return result

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.truediv,
                other,
                axis=axis,
                level=level,
                fill_value=fill_value,
            )
        other = self._validate_other(other, axis, numeric_only=True)
        new_query_compiler = self._query_compiler.truediv(
            other=other, axis=axis, level=level, fill_value=fill_value
        )
        return self._create_dataframe_from_compiler(new_query_compiler)

    def truncate(self, before=None, after=None, axis=None, copy=True):
        return self._default_to_pandas(
            pandas.DataFrame.truncate, before=before, after=after, axis=axis, copy=copy
        )

    def tshift(self, periods=1, freq=None, axis=0):
        return self._default_to_pandas(
            pandas.DataFrame.tshift, periods=periods, freq=freq, axis=axis
        )

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        return self._default_to_pandas(
            pandas.DataFrame.tz_convert, tz, axis=axis, level=level, copy=copy
        )

    def tz_localize(self, tz, axis=0, level=None, copy=True, ambiguous="raise"):
        return self._default_to_pandas(
            pandas.DataFrame.tz_localize,
            tz,
            axis=axis,
            level=level,
            copy=copy,
            ambiguous=ambiguous,
        )

    def unstack(self, level=-1, fill_value=None):
        return self._default_to_pandas(
            pandas.DataFrame.unstack, level=level, fill_value=fill_value
        )

    def update(
        self, other, join="left", overwrite=True, filter_func=None, raise_conflict=False
    ):
        """Modify DataFrame in place using non-NA values from other.

        Args:
            other: DataFrame, or object coercible into a DataFrame
            join: {'left'}, default 'left'
            overwrite: If True then overwrite values for common keys in frame
            filter_func: Can choose to replace values other than NA.
            raise_conflict: If True, will raise an error if the DataFrame and
                other both contain data in the same place.

        Returns:
            None
        """
        if raise_conflict:
            return self._default_to_pandas(
                pandas.DataFrame.update,
                other,
                join=join,
                overwrite=overwrite,
                filter_func=filter_func,
                raise_conflict=raise_conflict,
            )
        if not isinstance(other, DataFrame):
            other = DataFrame(other)
        query_compiler = self._query_compiler.update(
            other._query_compiler,
            join=join,
            overwrite=overwrite,
            filter_func=filter_func,
            raise_conflict=raise_conflict,
        )
        self._update_inplace(new_query_compiler=query_compiler)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)

        return self._query_compiler.var(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs
        )

    def where(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        raise_on_error=None,
    ):
        """Replaces values not meeting condition with values in other.

        Args:
            cond: A condition to be met, can be callable, array-like or a
                DataFrame.
            other: A value or DataFrame of values to use for setting this.
            inplace: Whether or not to operate inplace.
            axis: The axis to apply over. Only valid when a Series is passed
                as other.
            level: The MultiLevel index level to apply over.
            errors: Whether or not to raise errors. Does nothing in Pandas.
            try_cast: Try to cast the result back to the input type.
            raise_on_error: Whether to raise invalid datatypes (deprecated).

        Returns:
            A new DataFrame with the replaced values.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(other, pandas.Series) and axis is None:
            raise ValueError("Must specify axis=0 or 1")
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            if isinstance(cond, DataFrame):
                cond = cond._query_compiler.to_pandas()
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.where,
                cond,
                other=other,
                inplace=False,
                axis=axis,
                level=level,
                errors=errors,
                try_cast=try_cast,
                raise_on_error=raise_on_error,
            )
            return self._create_dataframe_from_compiler(new_query_compiler, inplace)
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        cond = cond(self) if callable(cond) else cond

        if not isinstance(cond, DataFrame):
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            cond = DataFrame(cond, index=self.index, columns=self.columns)
        if isinstance(other, DataFrame):
            other = other._query_compiler
        elif isinstance(other, pandas.Series):
            other = other.reindex(self.index if not axis else self.columns)
        else:
            index = self.index if not axis else self.columns
            other = pandas.Series(other, index=index)
        query_compiler = self._query_compiler.where(
            cond._query_compiler, other, axis=axis, level=level
        )
        return self._create_dataframe_from_compiler(query_compiler, inplace)

    def xs(self, key, axis=0, level=None, drop_level=True):
        return self._default_to_pandas(
            pandas.DataFrame.xs, key, axis=axis, level=level, drop_level=drop_level
        )

    def __getitem__(self, key):
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
            A Pandas Series representing the value for the column.
        """
        key = com._apply_if_callable(key, self)
        # Shortcut if key is an actual column
        is_mi_columns = isinstance(self.columns, pandas.MultiIndex)
        try:
            if key in self.columns and not is_mi_columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass
        # see if we can slice the rows
        # This lets us reuse code in Pandas to error check
        indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
        if indexer is not None:
            return self._getitem_slice(indexer)
        if isinstance(key, (pandas.Series, np.ndarray, pandas.Index, list)):
            return self._getitem_array(key)
        elif isinstance(key, DataFrame):
            return self.where(key)
        elif is_mi_columns:
            return self._default_to_pandas(pandas.DataFrame.__getitem__, key)
            # return self._getitem_multilevel(key)
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key):
        return SeriesView(
            self._query_compiler.getitem_single_key(key), self, (slice(None), key)
        )

    def _getitem_array(self, key):
        if com.is_bool_indexer(key):
            if isinstance(key, pandas.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    "Item wrong length {} instead of {}.".format(
                        len(key), len(self.index)
                    )
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pandas.RangeIndex(len(self.index))[key]
            return DataFrame(query_compiler=self._query_compiler.getitem_row_array(key))
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return DataFrame(
                query_compiler=self._query_compiler.getitem_column_array(key)
            )

    def _getitem_slice(self, key):
        # We convert to a RangeIndex because getitem_row_array is expecting a list
        # of indices, and RangeIndex will give us the exact indices of each boolean
        # requested.
        key = pandas.RangeIndex(len(self.index))[key]
        return DataFrame(query_compiler=self._query_compiler.getitem_row_array(key))

    def __getattr__(self, key):
        """After regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.columns:
                return self[key]
            raise e

    def __setattr__(self, key, value):
        # We have to check for this first because we have to be able to set
        # _query_compiler before we check if the key is in self
        if key in ["_query_compiler"] or key in self.__dict__:
            pass
        elif key in self:
            self.__setitem__(key, value)
        elif isinstance(value, pandas.Series):
            warnings.warn(
                "Modin doesn't allow columns to be created via a new attribute name - see "
                "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access",
                UserWarning,
            )
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            return self._default_to_pandas(pandas.DataFrame.__setitem__, key, value)
        if key not in self.columns:
            self.insert(loc=len(self.columns), column=key, value=value)
        else:
            loc = self.columns.get_loc(key)
            self.__delitem__(key)
            self.insert(loc=loc, column=key, value=value)

    def __len__(self):
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def __unicode__(self):
        return self._default_to_pandas(pandas.DataFrame.__unicode__)

    def __invert__(self):
        return self._default_to_pandas(pandas.DataFrame.__invert__)

    def __hash__(self):
        return self._default_to_pandas(pandas.DataFrame.__hash__)

    def __iter__(self):
        """Iterate over the columns

        Returns:
            An Iterator over the columns of the DataFrame.
        """
        return iter(self.columns)

    def __contains__(self, key):
        """Searches columns for specific key

        Args:
            key : The column name

        Returns:
            Returns a boolean if the specified key exists as a column name
        """
        return self.columns.__contains__(key)

    def __nonzero__(self):
        raise ValueError(
            "The truth value of a {0} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )

    __bool__ = __nonzero__

    def __abs__(self):
        """Creates a modified DataFrame by taking the absolute value.

        Returns:
            A modified DataFrame
        """
        return self.abs()

    def __round__(self, decimals=0):
        return self._default_to_pandas(pandas.DataFrame.__round__, decimals=decimals)

    def __array__(self, dtype=None):
        # TODO: This is very inefficient and needs fix, also see as_matrix
        return to_pandas(self).__array__(dtype=dtype)

    def __array_wrap__(self, result, context=None):
        # TODO: This is very inefficient, see also __array__ and as_matrix
        return to_pandas(self).__array_wrap__(result, context=context)

    def __getstate__(self):
        return self._default_to_pandas(pandas.DataFrame.__getstate__)

    def __setstate__(self, state):
        return self._default_to_pandas(pandas.DataFrame.__setstate__, state)

    def __delitem__(self, key):
        """Delete a column by key. `del a[key]` for example.
           Operation happens in place.

           Notes: This operation happen on row and column partition
                  simultaneously. No rebuild.
        Args:
            key: key to delete
        """
        if key not in self:
            raise KeyError(key)
        self._update_inplace(new_query_compiler=self._query_compiler.delitem(key))

    def __finalize__(self, other, method=None, **kwargs):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.__finalize__, other, method=method, **kwargs
        )

    def __copy__(self, deep=True):
        """Make a copy using modin.DataFrame.copy method

        Args:
            deep: Boolean, deep copy or not.
                  Currently we do not support deep copy.

        Returns:
            A Ray DataFrame object.
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        """Make a -deep- copy using modin.DataFrame.copy method
           This is equivalent to copy(deep=True).

        Args:
            memo: No effect. Just to comply with Pandas API.

        Returns:
            A Ray DataFrame object.
        """
        return self.copy(deep=True)

    def __and__(self, other):
        return self.__bool__() and other

    def __or__(self, other):
        return self.__bool__() or other

    def __xor__(self, other):
        return self.__bool__() ^ other

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __add__(self, other, axis=None, level=None, fill_value=None):
        return self.add(other, axis=axis, level=level, fill_value=fill_value)

    def __iadd__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.add(other, axis=axis, level=level, fill_value=fill_value)

    def __radd__(self, other, axis=None, level=None, fill_value=None):
        return self.radd(other, axis=axis, level=level, fill_value=fill_value)

    def __mul__(self, other, axis=None, level=None, fill_value=None):
        return self.mul(other, axis=axis, level=level, fill_value=fill_value)

    def __imul__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.mul(other, axis=axis, level=level, fill_value=fill_value)

    def __rmul__(self, other, axis=None, level=None, fill_value=None):
        return self.rmul(other, axis=axis, level=level, fill_value=fill_value)

    def __pow__(self, other, axis=None, level=None, fill_value=None):
        return self.pow(other, axis=axis, level=level, fill_value=fill_value)

    def __ipow__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.pow(other, axis=axis, level=level, fill_value=fill_value)

    def __rpow__(self, other, axis=None, level=None, fill_value=None):
        return self.rpow(other, axis=axis, level=level, fill_value=fill_value)

    def __sub__(self, other, axis=None, level=None, fill_value=None):
        return self.sub(other, axis=axis, level=level, fill_value=fill_value)

    def __isub__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.sub(other, axis=axis, level=level, fill_value=fill_value)

    def __rsub__(self, other, axis=None, level=None, fill_value=None):
        return self.rsub(other, axis=axis, level=level, fill_value=fill_value)

    def __floordiv__(self, other, axis=None, level=None, fill_value=None):
        return self.floordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __ifloordiv__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.floordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __rfloordiv__(self, other, axis=None, level=None, fill_value=None):
        return self.rfloordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __truediv__(self, other, axis=None, level=None, fill_value=None):
        return self.truediv(other, axis=axis, level=level, fill_value=fill_value)

    def __itruediv__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.truediv(other, axis=axis, level=level, fill_value=fill_value)

    def __rtruediv__(self, other, axis=None, level=None, fill_value=None):
        return self.rtruediv(other, axis=axis, level=level, fill_value=fill_value)

    def __mod__(self, other, axis=None, level=None, fill_value=None):
        return self.mod(other, axis=axis, level=level, fill_value=fill_value)

    def __imod__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.mod(other, axis=axis, level=level, fill_value=fill_value)

    def __rmod__(self, other, axis=None, level=None, fill_value=None):
        return self.rmod(other, axis=axis, level=level, fill_value=fill_value)

    def __div__(self, other, axis=None, level=None, fill_value=None):
        return self.div(other, axis=axis, level=level, fill_value=fill_value)

    def __rdiv__(self, other, axis=None, level=None, fill_value=None):
        return self.rdiv(other, axis=axis, level=level, fill_value=fill_value)

    def __neg__(self):
        """Computes an element wise negative DataFrame

        Returns:
            A modified DataFrame where every element is the negation of before
        """
        self._validate_dtypes(numeric_only=True)
        return DataFrame(query_compiler=self._query_compiler.negative())

    def __sizeof__(self):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.__sizeof__)

    @property
    def __doc__(self):  # pragma: no cover
        def __doc__(df):
            """Defined because properties do not have a __name__"""
            return df.__doc__

        return self._default_to_pandas(__doc__)

    @property
    def blocks(self):
        def blocks(df):
            """Defined because properties do not have a __name__"""
            return df.blocks

        return self._default_to_pandas(blocks)

    @property
    def style(self):
        def style(df):
            """Defined because properties do not have a __name__"""
            return df.style

        return self._default_to_pandas(style)

    @property
    def iat(self, axis=None):
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    @property
    def loc(self):
        """Purely label-location based indexer for selection by label.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    @property
    def is_copy(self):
        def is_copy(df):
            """Defined because properties do not have a __name__"""
            return df.is_copy

        return self._default_to_pandas(is_copy)

    @property
    def at(self, axis=None):
        from .indexing import _LocIndexer

        return _LocIndexer(self)

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

    def _create_dataframe_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

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
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 1
        result = other
        if isinstance(other, DataFrame):
            return other._query_compiler
        elif is_list_like(other):
            other_dtypes = [type(x) for x in other]
            if axis == 0:
                if len(other) != len(self.index):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.index), len(other))
                    )
            else:
                if len(other) != len(self.columns):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.columns), len(other))
                    )
        else:
            other_dtypes = [
                type(other)
                for _ in range(len(self.index) if axis else len(self.columns))
            ]

        # Do dtype checking
        if numeric_only:
            if not all(
                is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype)
                for self_dtype, other_dtype in zip(self.dtypes, other_dtypes)
            ):
                raise TypeError("Cannot do operation on non-numeric dtypes")
        elif numeric_or_object_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (is_object_dtype(self_dtype) and is_object_dtype(other_dtype))
                for self_dtype, other_dtype in zip(self.dtypes, other_dtypes)
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
                for self_dtype, other_dtype in zip(self.dtypes, other_dtypes)
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
                for self_dtype, other_dtype in zip(self.dtypes, other_dtypes)
            ):
                raise TypeError(
                    "Cannot do operation non-numeric objects with numeric objects"
                )

        return result

    def _validate_dtypes(self, numeric_only=False):
        """Helper method to check that all the dtypes are the same"""
        dtype = self.dtypes[0]
        for t in self.dtypes:
            if numeric_only and not is_numeric_dtype(t):
                raise TypeError("{0} is not a numeric data type".format(t))
            elif not numeric_only and t != dtype:
                raise TypeError(
                    "Cannot compare type '{0}' with type '{1}'".format(t, dtype)
                )

    def _validate_dtypes_min_max(self, axis, numeric_only):
        # If our DataFrame has both numeric and non-numeric dtypes then
        # comparisons between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            axis
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot compare Numeric and Non-Numeric Types")

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        """Raises TypeErrors for sum, prod, and mean where necessary"""
        # We cannot add datetime types, so if we are summing a column with
        # dtype datetime64 and cannot ignore non-numeric types, we must throw a
        # TypeError.
        if (
            not axis
            and numeric_only is False
            and any(dtype == np.dtype("datetime64[ns]") for dtype in self.dtypes)
        ):
            raise TypeError("Cannot add Timestamp Types")

        # If our DataFrame has both numeric and non-numeric dtypes then
        # operations between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            (axis or ignore_axis)
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot operate on Numeric and Non-Numeric Types")

    def _default_to_pandas(self, op, *args, **kwargs):
        """Helper method to use default pandas function"""
        ErrorMessage.default_to_pandas("`{}`".format(op.__name__))
        result = op(self._query_compiler.to_pandas(), *args, **kwargs)
        # SparseDataFrames cannot be serialize by arrow and cause problems for Modin.
        # For now we will use pandas.
        if isinstance(result, pandas.DataFrame) and not isinstance(
            result, pandas.SparseDataFrame
        ):
            return DataFrame(result)
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
                        second = DataFrame(result[1])
                    else:
                        second = result[1]
                    return DataFrame(result[0]), second
                else:
                    return result
            except TypeError:
                return result
