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

from modin.data_management.functions.default_methods import (
    DataFrameDefault,
    SeriesDefault,
    DateTimeDefault,
    StrDefault,
    BinaryDefault,
    ResampleDefault,
    RollingDefault,
    CatDefault,
    GroupByDefault,
)
from modin.error_message import ErrorMessage

from pandas.core.dtypes.common import is_scalar
from pandas.util._decorators import doc, Appender, Substitution
import pandas.core.resample
import pandas
import numpy as np
from typing import List, Hashable


def _get_axis(axis):
    """Build index labels getter of the specified axis."""

    def axis_getter(self):
        ErrorMessage.default_to_pandas(f"DataFrame.get_axis({axis})")
        return self.to_pandas().axes[axis]

    return axis_getter


def _set_axis(axis):
    """Build index labels setter of the specified axis."""

    def axis_setter(self, labels):
        new_qc = DataFrameDefault.register(pandas.DataFrame.set_axis)(
            self, axis=axis, labels=labels
        )
        self.__dict__.update(new_qc.__dict__)

    return axis_setter


_add_one_column_warning = Appender(
    """.. warning::
    This method is supported only by one-column query compilers."""
)


def add_refer_to(method):
    add_note = Appender(
        # TODO: add direct hyper-link to the corresponding documentation when
        # it will be generated.
        """
        Notes
        -----
        Please refer to ``modin.pandas.%(method)s`` for more information
        about parameters and output format.
        """,
        join="",
    )
    sub_method = Substitution(method=method)

    def decorator(func):
        return sub_method(add_note(func))

    return decorator


def _doc_dt(prop, dt_type, method):
    template = """
    Get {prop} for each {dt_type} value.

    Returns
    -------
    BaseQueryCompiler
        New `QueryCompiler` with the same shape as `self`, where each element is
        {prop} for the corresponding {dt_type} value.
    """

    doc_adder = doc(template, prop=prop, dt_type=dt_type)
    refer_to_appender = add_refer_to(f"Series.dt.{method}")

    def decorator(func):
        return _add_one_column_warning(refer_to_appender(doc_adder(func)))

    return decorator


def _doc_dt_timestamp(prop, method):
    return _doc_dt(prop, "date-time", method)


def _doc_dt_interval(prop, method):
    return _doc_dt(prop, "interval", method)


class BaseQueryCompiler(abc.ABC):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    @abc.abstractmethod
    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """
        Do fallback to pandas for the passed function.

        Parameters
        ----------
        pandas_op : callable(pandas.DataFrame, *args, **kwargs) -> [pandas.DataFrame, pandas.Series, numpy.ndarray]
            Function to apply to the casted to pandas frame.
        *args : args
            Positional arguments to pass to `pandas_op`.
        **kwargs
            Key-value arguments to pass to `pandas_op`.

        Returns
        -------
        BaseQueryCompiler
            The result of the `pandas_op`, converted back to `BaseQueryCompiler`.
        """
        pass

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.

    lazy_execution = False

    # Metadata modification abstract methods
    def add_prefix(self, prefix, axis=1):
        """
        Add string prefix to the index labels along specified axis.

        Parameters
        ----------
        prefix : str
            The string to add before each label.
        axis : {0, 1}, default: 1
            Axis to add prefix along. 0 is for index and 1 is for columns.

        Returns
        -------
        BaseQueryCompiler
            New query compiler with updated labels.
        """
        if axis:
            return DataFrameDefault.register(pandas.DataFrame.add_prefix)(
                self, prefix=prefix
            )
        else:
            return SeriesDefault.register(pandas.Series.add_prefix)(self, prefix=prefix)

    def add_suffix(self, suffix, axis=1):
        """
        Add string suffix to the index labels along specified axis.

        Parameters
        ----------
        prefix : str
            The string to add after each label.
        axis : {0, 1}, default: 1
            Axis to add suffix along. 0 is for index and 1 is for columns.

        Returns
        -------
        BaseQueryCompiler
            New query compiler with updated labels.
        """
        if axis:
            return DataFrameDefault.register(pandas.DataFrame.add_suffix)(
                self, suffix=suffix
            )
        else:
            return SeriesDefault.register(pandas.Series.add_suffix)(self, suffix=suffix)

    # END Metadata modification abstract methods

    # Abstract copy

    def copy(self):
        """
        Make a copy of this object.

        Returns
        -------
        BaseQueryCompiler
            Copy of self.

        Note
        ----
        For copy, we don't want a situation where we modify the metadata of the
        copies if we end up modifying something here. We copy all of the metadata
        to prevent that.
        """
        return DataFrameDefault.register(pandas.DataFrame.copy)(self)

    # END Abstract copy

    # Abstract join and append helper functions

    def concat(self, axis, other, **kwargs):
        """
        Concatenate `self` with passed query compilers along specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to concatenate along. 0 is for index and 1 is for columns.
        other : BaseQueryCompiler or list of such.
            Objects to concatenate with `self`.
        join : {'outer', 'inner', 'right', 'left'}, default: 'outer'
            Type of join that will be used if indeces on the other axis are different.
            (If specified, have to be passed via `kwargs`).
        ignore_index : bool, default: False
            If `True`, do not use the index values along the concatenation axis.
            (If specified, have to be passed via `kwargs`).
            The resulting axis will be labeled 0, …, n - 1.
        sort : bool, default: False
            Whether or not to sort non-concatenation axis.
            (If specified, have to be passed via `kwargs`).
        **kwargs : kwargs
            Additional parameters in the glory of compatibility. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            Concatenated objects.
        """
        concat_join = ["inner", "outer"]

        def concat(df, axis, other, **kwargs):
            kwargs.pop("join_axes", None)
            ignore_index = kwargs.get("ignore_index", False)
            if kwargs.get("join", "outer") in concat_join:
                if not isinstance(other, list):
                    other = [other]
                other = [df] + other
                result = pandas.concat(other, axis=axis, **kwargs)
            else:
                if isinstance(other, (list, np.ndarray)) and len(other) == 1:
                    other = other[0]
                ignore_index = kwargs.pop("ignore_index", None)
                kwargs["how"] = kwargs.pop("join", None)
                result = df.join(other, rsuffix="r_", **kwargs)
            if ignore_index:
                if axis == 0:
                    result = result.reset_index(drop=True)
                else:
                    result.columns = pandas.RangeIndex(len(result.columns))
            return result

        return DataFrameDefault.register(concat)(self, axis=axis, other=other, **kwargs)

    # END Abstract join and append helper functions

    # Data Management Methods
    @abc.abstractmethod
    def free(self):
        """Trigger a cleanup of this object."""
        # TODO create a way to clean up this object.
        pass

    @abc.abstractmethod
    def finalize(self):
        """Finalize constructing the dataframe calling all deferred functions which were used to build it."""
        pass

    # END Data Management Methods

    # To/From Pandas
    @abc.abstractmethod
    def to_pandas(self):
        """
        Convert Modin DataFrame to pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            The QueryCompiler converted to pandas.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pandas(cls, df, data_cls):
        """
        Build `QueryCompiler` from pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas DataFrame to convert from.
        data_cls : cls
            `BasePandasFrame` object to convert to.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` containing data from the pandas DataFrame.
        """
        pass

    # END To/From Pandas

    # From Arrow
    @classmethod
    @abc.abstractmethod
    def from_arrow(cls, at, data_cls):
        """
        Build `QueryCompiler` from Arrow Table.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.
        data_cls : cls
            `BasePandasFrame` object to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END From Arrow

    # To NumPy

    def to_numpy(self, **kwargs):
        """
        Convert Modin DataFrame to NumPy array.

        Returns
        -------
        np.ndarray
            The QueryCompiler converted to Numpy array.
        """
        return DataFrameDefault.register(pandas.DataFrame.to_numpy)(self, **kwargs)

    # END To NumPy

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    __doc_arithmetic_operator_desc = """
    Perform element-wise {operation} (self {sign} other).
    """

    __doc_arithmetic_roperator_desc = """
    Perform element-wise {operation} (other {sign} self).
    """

    __doc_binary_operator_signature_template = """
    If axes are note equal, first perform frames allignment.

    Parameters
    ----------
    {params}

    Returns
    -------
    BaseQueryCompiler
        Result of binary operation.
    """

    __doc_binary_operator_params = """other : BaseQueryCompiler, scalar or array-like
        Other operand of the binary operation.
    broadcast : bool, default: False
        If `other` is a one-column query compiler, indicates whether it is a Series or not.
        Frames and Series have to be processed differently, however we can't distinguish them
        at the query compiler level, so this parameter is a hint that passed from a high level API.
    """

    __doc_comparison_operator_params = "".join(
        [
            __doc_binary_operator_params,
            """level : int or label
        In case of MultiIndex match index values on the passed level.
    axis : int
        Axis to match indice along for 1D `other` (list or QueryCompiler that represents Series).
        """,
        ]
    )

    __doc_arithmetic_operator_params = "".join(
        [
            __doc_comparison_operator_params,
            """fill_value : float or None
        Value to fill missing elements in the result of frame allignment.""",
        ]
    )

    __doc_binary_operator = "".join(
        [
            __doc_arithmetic_operator_desc,
            __doc_binary_operator_signature_template.format(
                params=__doc_binary_operator_params
            ),
        ]
    )

    __doc_binary_roperator = "".join(
        [
            __doc_arithmetic_roperator_desc,
            __doc_binary_operator_signature_template.format(
                params=__doc_binary_operator_params
            ),
        ]
    )

    __doc_arithmetic_operator = "".join(
        [
            __doc_arithmetic_operator_desc,
            __doc_binary_operator_signature_template.format(
                params=__doc_arithmetic_operator_params
            ),
        ]
    )
    __doc_arithmetic_roperator = "".join(
        [
            __doc_arithmetic_roperator_desc,
            __doc_binary_operator_signature_template.format(
                params=__doc_arithmetic_operator_params
            ),
        ]
    )
    __doc_comparison_operator = "".join(
        [
            __doc_arithmetic_operator_desc,
            __doc_binary_operator_signature_template.format(
                params=__doc_comparison_operator_params
            ),
        ]
    )

    @doc(__doc_arithmetic_operator, operation="addition", sign="+")
    def add(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.add)(self, other=other, **kwargs)

    def combine(self, other, **kwargs):
        """
        Perform column-wise combine with another QueryCompiler with passed `func`.

        If axes are note equal, first perform frames allignment.

        Parameters
        ----------
        other : BaseQueryCompiler
            Left operand of the binary operation.
        func : callable(pandas.Series, pandas.Series) -> pandas.Series
            Function that takes two `pandas.Series` with alligned axes
            and return one `pandas.Series` - the result combination.
        fill_value : float or None
            Value to fill missing values with after frame alignment occurred.
        overwrite : bool
            If True, columns in `self` that do not exist in `other`
            will be overwritten with NaNs.

        Returns
        -------
        BaseQueryCompiler
            Result of combine.
        """
        return BinaryDefault.register(pandas.DataFrame.combine)(
            self, other=other, **kwargs
        )

    def combine_first(self, other, **kwargs):
        """
        Fill null elements of `self` with value in the same location in `other`.

        If axes are note equal, first perform frames allignment.

        Parameters
        ----------
        other : BaseQueryCompiler
            Provided frame to use to fill null values.

        Returns
        -------
        BaseQueryCompiler
        """
        return BinaryDefault.register(pandas.DataFrame.combine_first)(
            self, other=other, **kwargs
        )

    @doc(__doc_comparison_operator, operation="equality comparison", sign="==")
    def eq(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.eq)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_operator, operation="integer division", sign="//")
    def floordiv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.floordiv)(
            self, other=other, **kwargs
        )

    @doc(
        __doc_comparison_operator,
        operation="greater than or equal comparison",
        sign=">=",
    )
    def ge(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.ge)(self, other=other, **kwargs)

    @doc(__doc_comparison_operator, operation="greater than comparison", sign=">")
    def gt(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.gt)(self, other=other, **kwargs)

    @doc(
        __doc_comparison_operator, operation="less than or equal comparison", sign="<="
    )
    def le(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.le)(self, other=other, **kwargs)

    @doc(__doc_comparison_operator, operation="less than comparison", sign="<")
    def lt(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.lt)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_operator, operation="modulo", sign="%")
    def mod(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.mod)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_operator, operation="multiplication", sign="*")
    def mul(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.mul)(self, other=other, **kwargs)

    @add_refer_to("DataFrame.corr")
    def corr(self, **kwargs):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable(pandas.Series, pandas.Series) -> pandas.Series
            Method of correlation.
        min_periods : int
            Minimum number of observations required per pair of columns
            to have a valid result. If fewer than `min_periods` non-NA values
            are present the result will be NA.

        Returns
        -------
        BaseQueryCompiler
            Correlation matrix.
        """
        return DataFrameDefault.register(pandas.DataFrame.corr)(self, **kwargs)

    @add_refer_to("DataFrame.cov")
    def cov(self, **kwargs):
        """
        Compute pairwise covariance of columns, excluding NA/null values.

        Parameters
        ----------
        min_periods : int

        Returns
        -------
        BaseQueryCompiler
            Covariance matrix.
        """
        return DataFrameDefault.register(pandas.DataFrame.cov)(self, **kwargs)

    def dot(self, other, **kwargs):
        """
        Compute the matrix multiplication of self and other.

        Parameters
        ----------
        other : BaseQueryCompiler or NumPy array
            The other query compiler or NumPy array to matrix multiply with self.
        squeeze_self : boolean
            The flag to squeeze self.
        squeeze_other : boolean
            The flag to squeeze other (this flag is applied if `other` is query compiler).

        Returns
        -------
        BaseQueryCompiler
            A new query compiler that contains result of the matrix multiply.
        """
        if kwargs.get("squeeze_self", False):
            applyier = pandas.Series.dot
        else:
            applyier = pandas.DataFrame.dot
        return BinaryDefault.register(applyier)(self, other=other, **kwargs)

    @doc(__doc_comparison_operator, operation="not equal comparison", sign="!=")
    def ne(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.ne)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_operator, operation="exponential power", sign="**")
    def pow(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.pow)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_roperator, operation="integer division", sign="//")
    def rfloordiv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rfloordiv)(
            self, other=other, **kwargs
        )

    @doc(__doc_arithmetic_roperator, operation="modulo", sign="%")
    def rmod(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rmod)(
            self, other=other, **kwargs
        )

    @doc(__doc_arithmetic_roperator, operation="exponential power", sign="**")
    def rpow(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rpow)(
            self, other=other, **kwargs
        )

    @doc(__doc_arithmetic_roperator, operation="substraction", sign="-")
    def rsub(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rsub)(
            self, other=other, **kwargs
        )

    @doc(__doc_arithmetic_roperator, operation="division", sign="/")
    def rtruediv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rtruediv)(
            self, other=other, **kwargs
        )

    @doc(__doc_arithmetic_operator, operation="substraction", sign="-")
    def sub(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.sub)(self, other=other, **kwargs)

    @doc(__doc_arithmetic_operator, operation="division", sign="/")
    def truediv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.truediv)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_operator, operation="and", sign="&")
    def __and__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__and__)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_operator, operation="or", sign="|")
    def __or__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__or__)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_roperator, operation="and", sign="&")
    def __rand__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__rand__)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_roperator, operation="or", sign="|")
    def __ror__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__ror__)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_operator, operation="xor", sign="^")
    def __rxor__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__rxor__)(
            self, other=other, **kwargs
        )

    @doc(__doc_binary_operator, operation="xor", sign="^")
    def __xor__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__xor__)(
            self, other=other, **kwargs
        )

    # FIXME: query compiler shoudln't care about differences between Frame and Series.
    # We should combine `df_update` and `series_update` into one method.
    @add_refer_to("DataFrame.update")
    def df_update(self, other, **kwargs):
        """
        Update values of self using non-NA values of other at the corresponding positions.

        If axes are not equal, first perform frames allignment.

        Parameters
        ----------
        other : BaseQueryCompiler
            Frame to grab replacement values from.
        join : {"left"}
            Specify type of join to allign frames if axes are not equal.
        overwrite : bool
            Whether to overwrite every corresponding value of self, or only if it's NAN.
        filter_func : callable(pandas.Series, pandas.Series) -> numpy.ndarray<bool>
            Function that takes column of the self and return bool mask for values, that
            should be overwriten in the self frame.
        errors : {"raise", "ignore"}
            If "raise", will raise a `ValueError` if self and other both contain
            non-NA data in the same place.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with updated values.
        """
        return BinaryDefault.register(pandas.DataFrame.update, inplace=True)(
            self, other=other, **kwargs
        )

    def series_update(self, other, **kwargs):
        """
        Update values of self using values of other at the corresponding indices.

        Parameters
        ----------
        other : BaseQueryCompiler
            One-column query compiler with updated values.
        """
        return BinaryDefault.register(pandas.Series.update, inplace=True)(
            self, other=other, squeeze_self=True, squeeze_other=True, **kwargs
        )

    @add_refer_to("DataFrame.clip")
    def clip(self, lower, upper, **kwargs):
        """
        Trim values at input threshold.

        Parameters
        ----------
        lower : float or list-like
        upper : float or list-like
        axis : int
        inplace : bool
            Serves the compatibility purpose, have no effect on the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with values limited by the specified thresholds.
        """
        return DataFrameDefault.register(pandas.DataFrame.clip)(
            self, lower=lower, upper=upper, **kwargs
        )

    def where(self, cond, other, **kwargs):
        """
        Update values of self using values from `other` at positions where `cond` is True.

        Parameters
        ----------
        cond : BaseQueryCompiler
            Boolean mask. True - keep the self value, False - replace by `other` value.
        other : BaseQueryCompiler or pandas.Series
            Object to grab replacement values from.
        axis : int
            Axis to align frames along if axes of self, `cond` and `other` are not equal.
        level : int or label, optional
            Level of MultiIndex to align frames along if axes of self, `cond`
            and `other` are not equal. Currently `level` parameter is not implemented,
            so only `None` value is acceptable.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with updated data.
        """
        return DataFrameDefault.register(pandas.DataFrame.where)(
            self, cond=cond, other=other, **kwargs
        )

    @add_refer_to("DataFrame.merge")
    def merge(self, right, **kwargs):
        """
        Merge `QueryCompiler` objects with a database-style join.

        Parameters
        ----------
        right : BaseQueryCompiler
            `QueryCompiler` of the right frame to merge with.
        how : {"left", "right", "outer", "inner", "cross"}
        on : label or list of such
        left_on : label or list of such
        right_on : label or list of such
        left_index : bool
        right_index : bool
        sort : bool
        suffixes : list-like
        copy : bool
        indicator : bool or str
        validate : str

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` that contains result of the merge.
        """
        return DataFrameDefault.register(pandas.DataFrame.merge)(
            self, right=right, **kwargs
        )

    @add_refer_to("DataFrame.join")
    def join(self, right, **kwargs):
        """
        Join columns of another `QueryCompiler`.

        Parameters
        ----------
        right : BaseQueryCompiler
            `QueryCompiler` of the right frame to join with.
        on : label or list of such
        how : {"left", "right", "outer", "inner"}
        lsuffix : str
        rsuffix : str
        sort : bool

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` that contains result of the join.
        """
        return DataFrameDefault.register(pandas.DataFrame.join)(self, right, **kwargs)

    # END Abstract inter-data operations

    # Abstract Transpose
    def transpose(self, *args, **kwargs):
        """
        Transpose this `QueryCompiler`.

        Parameters
        ----------
        copy : bool
            Whether to copy the data after transposing.

        Returns
        -------
            Transposed new `QueryCompiler`.
        """
        return DataFrameDefault.register(pandas.DataFrame.transpose)(
            self, *args, **kwargs
        )

    def columnarize(self):
        """
        Transpose this `QueryCompiler` if it has a single row but multiple columns.

        This method should be called for `QueryCompilers` representing a Series object,
        i.e. ``self.is_series_like()`` should be True.

        Returns
        -------
        BaseQueryCompiler
            Transposed new QueryCompiler or self.
        """
        if len(self.columns) != 1 or (
            len(self.index) == 1 and self.index[0] == "__reduced__"
        ):
            return self.transpose()
        return self

    def is_series_like(self):
        """Return True if ``QueryCompiler`` has a single column or row."""
        return len(self.columns) == 1 or len(self.index) == 1

    # END Abstract Transpose

    # Abstract reindex/reset_index (may shuffle data)
    @add_refer_to("DataFrame.reindex")
    def reindex(self, axis, labels, **kwargs):
        """
        Allign `QueryCompiler` data with a new index-labels along specified axis.

        Parameters
        ----------
        axis : int
            Axis to align labels along. 0 is for index, 1 is for columns.
        labels : list-like
            Index-labels to align with.
        method : {None, "backfill"/"bfill", "pad"/"ffill", "nearest"}
            Method to use for filling holes in reindexed frame.
            Please refer to ``modin.pandas.DataFrame.reindex`` for more information.
        fill_value : scalar
            Value to use for missing values in the resulted frame.
        limit : int
        tolerance : int

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with aligned axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.reindex)(
            self, axis=axis, labels=labels, **kwargs
        )

    def reset_index(self, **kwargs):
        """
        Reset the index, or a level of it.

        Parameters
        ----------
        drop : bool
            Whether to drop the reseted index or insert it at the begining of the frame.
        level : int or label, optional
            Level to remove from index. Removes all levels by default.
        col_level : int or label
            If the columns have multiple levels, determines which level the labels
            are inserted into.
        col_fill : label
            If the columns have multiple levels, determines how the other levels
            are named.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with reseted index.
        """
        return DataFrameDefault.register(pandas.DataFrame.reset_index)(self, **kwargs)

    def set_index_from_columns(
        self, keys: List[Hashable], drop: bool = True, append: bool = False
    ):
        """
        Create new row labels from a list of columns.

        Parameters
        ----------
        keys : list of hashable
            The list of column names that will become the new index.
        drop : bool
            Whether or not to drop the columns provided in the `keys` argument.
        append : bool
            Whether or not to add the columns in `keys` as new levels appended to the
            existing index.

        Returns
        -------
        BaseQueryCompiler
            A new `QueryCompiler` with updated index.
        """
        return DataFrameDefault.register(pandas.DataFrame.set_index)(
            self, keys=keys, drop=drop, append=append
        )

    # END Abstract reindex/reset_index

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.

    def is_monotonic_increasing(self):
        """
        Return boolean if values in the object are monotonicly increasing.

        Returns
        -------
        bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_increasing)(self)

    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are monotonicly decreasing.

        Returns
        -------
        bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_decreasing)(self)

    __doc_numeric_method_template = """
    Get the {method} for each column or row.

    Parameters
    ----------
    {params}

    Returns
    -------
    BaseQueryCompiler
        One-column `QueryCompiler` with index labels of the specified axis,
        where each row contains the {method} for the corresponding
        row or column.

    Notes
    -----
    Please refer to ``modin.pandas.DataFrame.{link}`` for more information about parameters.
    """

    __doc_numeric_method_base_params = """axis : int
    level : int or label, optional
        Currently `level` parameter is not implemented, so only `None` value is acceptable.
    numeric_only : bool"""

    __doc_numeric_method_na_params = "\n".join(
        [__doc_numeric_method_base_params, "skipna : bool"]
    )

    __doc_numeric_method_mincount_params = "\n".join(
        [
            __doc_numeric_method_na_params,
            """min_count : int
        Currently `min_count` parameter is not implemented, so only `1` is acceptable.""",
        ]
    )

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_base_params,
        method="number of non-NaN values",
        link="count",
    )
    def count(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.count)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_na_params,
        method="maximum value",
        link="max",
    )
    def max(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.max)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_na_params,
        method="mean value",
        link="mean",
    )
    def mean(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.mean)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_na_params,
        method="minimum value",
        link="min",
    )
    def min(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.min)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_mincount_params,
        method="production",
        link="prod",
    )
    def prod(self, squeeze_self, axis, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        # TODO: rework to original implementation after pandas issue #41074 resolves if possible.
        def map_func(df, **kwargs):
            """Apply .prod to DataFrame or Series in depend on `squeeze_self.`"""
            if squeeze_self:
                result = df.squeeze(axis=1).prod(**kwargs)
                if is_scalar(result):
                    if axis:
                        return pandas.DataFrame(
                            [result], index=["__reduced__"], columns=["__reduced__"]
                        )
                    else:
                        return pandas.Series([result], index=[df.columns[0]])
                else:
                    return result
            else:
                return df.prod(**kwargs)

        return DataFrameDefault.register(
            map_func,
        )(self, axis=axis, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_mincount_params,
        method="sum",
        link="sum",
    )
    def sum(self, squeeze_self, axis, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        # TODO: rework to original implementation after pandas issue #41074 resolves if possible.
        def map_func(df, **kwargs):
            """Apply .sum to DataFrame or Series in depend on `squeeze_self.`"""
            if squeeze_self:
                result = df.squeeze(axis=1).sum(**kwargs)
                if is_scalar(result):
                    if axis:
                        return pandas.DataFrame(
                            [result], index=["__reduced__"], columns=["__reduced__"]
                        )
                    else:
                        return pandas.Series([result], index=[df.columns[0]])
                else:
                    return result
            else:
                return df.sum(**kwargs)

        return DataFrameDefault.register(
            map_func,
        )(self, axis=axis, **kwargs)

    @add_refer_to("to_datetime")
    def to_datetime(self, *args, **kwargs):
        """
        Convert columns of the `QueryCompiler` to the datetime dtype.

        Parameters
        ----------
        *args : args
        **kwargs : kwargs

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with all columns converted to datetime dtype.
        """
        return SeriesDefault.register(pandas.to_datetime)(self, *args, **kwargs)

    # END Abstract full Reduce operations

    # Abstract map partitions operations
    # These operations are operations that apply a function to every partition.
    def abs(self):
        """
        Get absolute numeric value of each element.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with absolute numeric value of each element.
        """
        return DataFrameDefault.register(pandas.DataFrame.abs)(self)

    def applymap(self, func):
        """
        Apply passed function elementwise.

        Parameters
        ----------
        func : callable(scalar) -> scalar
            Function to apply to each element of the `QueryCompiler`.

        Returns
        -------
        BaseQueryCompiler
            Transformed `QueryCompiler`.
        """
        return DataFrameDefault.register(pandas.DataFrame.applymap)(self, func=func)

    def conj(self, **kwargs):
        """
        Get the complex conjugate for every element of self.

        The complex conjugate of a complex number is obtained by changing the sign
        of its imaginary part. Note that only numeric data is allowed.

        Parameters
        ----------
        **kwargs : kwargs, optional

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with conjugate applied element-wise.

        Notes
        -----
        Please refer to ``numpy.conj`` for parameters description.
        """

        def conj(df, *args, **kwargs):
            return pandas.DataFrame(np.conj(df))

        return DataFrameDefault.register(conj)(self, **kwargs)

    # FIXME:
    #   1. High-level objects leaks to the query compiler level.
    #   2. Spread **kwargs into actual arguments.
    def isin(self, **kwargs):
        """
        Check for each element of self whether it's contained in passed values.

        Parameters
        ----------
        values : list-like, modin.pandas.Series, modin.pandas.DataFrame or dict
            Values to check elements of self in.

        Returns
        -------
        BaseQueryCompiler
            Boolean mask for self of whether an element at the corresponding
            position is contained in `values`.
        """
        return DataFrameDefault.register(pandas.DataFrame.isin)(self, **kwargs)

    def isna(self):
        """
        Check for each element of self whether it's NaN.

        Returns
        -------
        BaseQueryCompiler
            Boolean mask for self of whether an element at the corresponding
            position is NaN.
        """
        return DataFrameDefault.register(pandas.DataFrame.isna)(self)

    # FIXME: this method is not supposed to take any parameters.
    def negative(self, **kwargs):
        """
        Change the sign for every value of self.

        Parameters
        ----------
        **kwargs : kwargs
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler

        Notes
        -----
        Be aware, that all `QueryCompiler` values have to be numeric.
        """
        return DataFrameDefault.register(pandas.DataFrame.__neg__)(self, **kwargs)

    def notna(self):
        """
        Check for each element of `self` whether it's existing (non-missing) value.

        Returns
        -------
        BaseQueryCompiler
            Boolean mask for `self` of whether an element at the corresponding
            position is exists.
        """
        return DataFrameDefault.register(pandas.DataFrame.notna)(self)

    @add_refer_to("DataFrame.round")
    def round(self, **kwargs):
        """
        Round every numeric value up to specified number of decimals.

        Parameters
        ----------
        decimals : int or list-like
            Number of decimals to round each column to.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with rounded values.
        """
        return DataFrameDefault.register(pandas.DataFrame.round)(self, **kwargs)

    # FIXME:
    #   1. high-level objects leaks to the query compiler.
    #   2. remove `inplace` parameter.
    @add_refer_to("DataFrame.replace")
    def replace(self, **kwargs):
        """
        Replace values given in `to_replace` by `value`.

        Parameters
        ----------
        to_replace : scalar, list-like, regex, modin.pandas.Series, or None
        value: scalar, list-like, regex or dict
        inplace : False
            This parameter serves the compatibility purpose. Always have to be False.
        limit : int or None
        regex : bool or same types as `to_replace`
        method : {"pad", "ffill", "bfill", None}

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with all `to_replace` values replaced by `value`.
        """
        return DataFrameDefault.register(pandas.DataFrame.replace)(self, **kwargs)

    @_add_one_column_warning
    def series_view(self, **kwargs):
        """
        Reinterpret underlying data with new dtype.

        Parameters
        ----------
        dtype : dtype
            Data type to reinterpret underlying data with.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` of the same data in memory, with reinterpreted values.

        .. warning::
            Be aware, that if this method do fallback to pandas, then newly created
            `QueryCompiler` will be the copy of the original data.
        """
        return SeriesDefault.register(pandas.Series.view)(self, **kwargs)

    @_add_one_column_warning
    @add_refer_to("to_numeric")
    def to_numeric(self, *args, **kwargs):
        """
        Convert underlying data to numeric dtype.

        Parameters
        ----------
        errors : {"ignore", "raise", "coerce"}
        downcast : {"integer", "signed", "unsigned", "float", None}

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with converted to numeric values.
        """
        return SeriesDefault.register(pandas.to_numeric)(self, *args, **kwargs)

    # FIXME: get rid of `**kwargs` parameter.
    @_add_one_column_warning
    def unique(self, **kwargs):
        """
        Get unique values of `self`.

        Parameters
        ----------
        **kwargs : kwargs
            Serves compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with unique values.
        """
        return SeriesDefault.register(pandas.Series.unique)(self, **kwargs)

    @_add_one_column_warning
    @add_refer_to("Series.searchsorted")
    def searchsorted(self, **kwargs):
        """
        Find positions in a sorted `self` where `value` should be inserted to maintain order.

        Parameters
        ----------
        value : list-like
        side : {"left", "right"}
        sorter : list-like, optional

        Returns
        -------
        BaseQueryCompiler
            One-column `QueryCompiler` which contains indices to insert.
        """
        return SeriesDefault.register(pandas.Series.searchsorted)(self, **kwargs)

    # END Abstract map partitions operations

    @_add_one_column_warning
    @add_refer_to("Series.value_counts")
    def value_counts(self, **kwargs):
        """
        Count unique values of one-column `self`.

        Parameters
        ----------
        normalize : bool
        sort : bool
        ascending : bool
        bins : int, optional
        dropna : bool

        Returns
        -------
        BaseQueryCompiler
            One-column `QueryCompiler` which index labels is a unique elements of `self`
            and each row contains the number of times corresponding value was met in the `self`.
        """
        return SeriesDefault.register(pandas.Series.value_counts)(self, **kwargs)

    @add_refer_to("DataFrame.stack")
    def stack(self, level, dropna):
        """
        Stack the prescribed level(s) from columns to index.

        Parameters
        ----------
        level : int or label
        dropna : bool

        Returns
        -------
        BaseQueryCompiler
        """
        return DataFrameDefault.register(pandas.DataFrame.stack)(
            self, level=level, dropna=dropna
        )

    # Abstract map partitions across select indices
    def astype(self, col_dtypes, **kwargs):
        """
        Convert columns dtypes to given dtypes.

        Parameters
        ----------
        col_dtypes : dict
            Map for column names and new dtypes.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with updated dtypes.
        """
        return DataFrameDefault.register(pandas.DataFrame.astype)(
            self, dtype=col_dtypes, **kwargs
        )

    @property
    # FIXME:
    def dtypes(self):
        """
        Get columns dtypes.

        Returns
        -------
        pandas.Series
            Series with dtypes of each column.
        """
        return self.to_pandas().dtypes

    # END Abstract map partitions across select indices

    # Abstract column/row partitions reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def all(self, **kwargs):
        """
        Return whether all the elements are true, potentially over an axis.

        Parameters
        ----------
        axis : int, optional
        bool_only : bool, optional
        skipna : bool
        level : int or label

        Returns
        -------
        BaseQueryCompiler
            If axis was specified return one-column `QueryCompiler` with index labels
            of the specified axis, where each row contains boolean of whether all elements
            at the corresponding row or column are True. Otherwise return `QueryCompiler`
            with a single bool of whether all elements are True.
        """
        return DataFrameDefault.register(pandas.DataFrame.all)(self, **kwargs)

    def any(self, **kwargs):
        """
        Return whether any element is true, potentially over an axis.

        Parameters
        ----------
        axis : int, optional
        bool_only : bool, optional
        skipna : bool
        level : int or label

        Returns
        -------
        BaseQueryCompiler
            If axis was specified return one-column `QueryCompiler` with index labels
            of the specified axis, where each row contains boolean of whether any element
            at the corresponding row or column is True. Otherwise return `QueryCompiler`
            with a single bool of whether any element is True.
        """
        return DataFrameDefault.register(pandas.DataFrame.any)(self, **kwargs)

    def first_valid_index(self):
        """
        Return index label of first non-NaN/NULL value.

        Returns
        --------
        scalar
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.first_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    @add_refer_to("DataFrame.idxmax")
    def idxmax(self, **kwargs):
        """
        Get position of the first occurance of the maximum for each row or column.

        Parameters
        ----------
        axis : int
        skipna : bool

        Returns
        -------
        BaseQueryCompiler
            One-column `QueryCompiler` with index labels of the specified axis,
            where each row contains position of the maximum element for the
            corresponding row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmax)(self, **kwargs)

    @add_refer_to("DataFrame.idxmin")
    def idxmin(self, **kwargs):
        """
        Get position of the first occurance of the minimum for each row or column.

        Parameters
        ----------
        axis : int
        skipna : bool

        Returns
        -------
        BaseQueryCompiler
            One-column `QueryCompiler` with index labels of the specified axis,
            where each row contains position of the minimum element for the
            corresponding row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmin)(self, **kwargs)

    def last_valid_index(self):
        """
        Return index label of last non-NaN/NULL value.

        Returns
        --------
        scalar
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.last_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_na_params,
        method="median value",
        link="median",
    )
    def median(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.median)(self, **kwargs)

    @add_refer_to("DataFrame.memory_usage")
    def memory_usage(self, **kwargs):
        """
        Return the memory usage of each column in bytes.

        Parameters
        ----------
        index : bool
        deep : bool

        Returns
        -------
        BaseQueryCompiler
            One-column `QueryCompiler` with index labels of `self`, where each row
            contains the memory usage for the corresponding column.
        """
        return DataFrameDefault.register(pandas.DataFrame.memory_usage)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params="""axis : int
        dropna : bool""",
        method="the number of unique values",
        link="nunique",
    )
    def nunique(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.nunique)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params="""q : float
        axis : int
        numeric_only : bool
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}""",
        method="value at the given quantile",
        link="quantile",
    )
    def quantile_for_single_value(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_na_params,
        method="unbiased skew",
        link="skew",
    )
    def skew(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.skew)(self, **kwargs)

    __doc_numeric_method_ddof_params = "\n".join(
        [__doc_numeric_method_na_params, "ddof : int"]
    )

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_ddof_params,
        method="standard deviation of the mean",
        link="sem",
    )
    def sem(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.sem)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_ddof_params,
        method="standard deviation",
        link="std",
    )
    def std(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.std)(self, **kwargs)

    @doc(
        __doc_numeric_method_template,
        params=__doc_numeric_method_ddof_params,
        method="variance",
        link="var",
    )
    def var(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.var)(self, **kwargs)

    # END Abstract column/row partitions reduce operations

    # Abstract column/row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def describe(self, **kwargs):
        """
        Generate descriptive statistics.

        Parameters
        ----------
        percentiles : list-like
        include : "all" or list of dtypes, optional
        exclude : list of dtypes, optional
        datetime_is_numeric : bool

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` object containing the descriptive statistics
            of the underlying data.

        Notes
        -----
        For more information about parameters and output statistics format please
        refer to ``modin.pandas.DataFrame.describe``.
        """
        return DataFrameDefault.register(pandas.DataFrame.describe)(self, **kwargs)

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    __doc_cum_method = """
    Get cummulative {method} for every row or column.

    Parameters
    ----------
    axis : int
    skipna : bool

    Returns
    -------
    BaseQueryCompiler
        `QueryCompiler` of the same shape as `self`, where each element is the {method}
        of all the previous values in this row or column.

    Notes
    -----
    Please refer to ``modin.pandas.DataFrame.{link}`` for more information about parameters.
    """

    @doc(__doc_cum_method, method="sum", link="cumsum")
    def cumsum(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cumsum)(self, **kwargs)

    @doc(__doc_cum_method, method="maximum", link="cummax")
    def cummax(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cummax)(self, **kwargs)

    @doc(__doc_cum_method, method="minimum", link="cummin")
    def cummin(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cummin)(self, **kwargs)

    @doc(__doc_cum_method, method="product", link="cumprod")
    def cumprod(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cumprod)(self, **kwargs)

    @add_refer_to("DataFrame.diff")
    def diff(self, **kwargs):
        """
        First discrete difference of element.

        Parameters
        ----------
        periods : int
        axis : int

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` of the same shape as `self`, where each element is the difference
            between the corresponding value and the previous value in this row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.diff)(self, **kwargs)

    @add_refer_to("DataFrame.dropna")
    def dropna(self, **kwargs):
        """
        Remove missing values.

        Parameters
        ----------
        axis : int
        how : {"any", "all"}
        thresh : int, optional
        subset : list of labels

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with null values dropped along given axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.dropna)(self, **kwargs)

    @add_refer_to("DataFrame.nlargest")
    def nlargest(self, n=5, columns=None, keep="first"):
        """
        Return the first n rows ordered by columns in descending order.

        Parameters
        ----------
        n : int, default: 5
        columns : list of labels, optional
        keep : {"first", "last", "all"}

        Returns
        -------
        BaseQueryCompiler
        """
        if columns is None:
            return SeriesDefault.register(pandas.Series.nlargest)(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register(pandas.DataFrame.nlargest)(
                self, n=n, columns=columns, keep=keep
            )

    @add_refer_to("DataFrame.nsmallest")
    def nsmallest(self, n=5, columns=None, keep="first"):
        """
        Return the first n rows ordered by columns in ascending order.

        Parameters
        ----------
        n : int, default: 5
        columns : list of labels, optional
        keep : {"first", "last", "all"}

        Returns
        -------
        BaseQueryCompiler
        """
        if columns is None:
            return SeriesDefault.register(pandas.Series.nsmallest)(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register(pandas.DataFrame.nsmallest)(
                self, n=n, columns=columns, keep=keep
            )

    @add_refer_to("DataFrame.eval")
    def eval(self, expr, **kwargs):
        """
        Evaluate string expression on `QueryCompiler` columns.

        Parameters
        ----------
        expr : str
        **kwargs : kwargs

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` containing the result of evaluation.
        """
        return DataFrameDefault.register(pandas.DataFrame.eval)(
            self, expr=expr, **kwargs
        )

    @add_refer_to("DataFrame.mode")
    def mode(self, **kwargs):
        """
        Get the modes for every column or row.

        Parameters
        ----------
        axis : int
        numeric_only : bool
        dropna : bool

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with modes calculated alogn given axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.mode)(self, **kwargs)

    @add_refer_to("DataFrame.fillna")
    def fillna(self, **kwargs):
        """
        Replace NaN values with the method provided.

        Parameters
        ----------
        value : scalar or dict
        method : {"backfill", "bfill", "pad", "ffill", None}
        axis : int
        inplace : False
            This parameter serves the compatibility purpose. Always have to be False.
        limit : int, optional
        downcast : dict, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with all null values filled.
        """
        return DataFrameDefault.register(pandas.DataFrame.fillna)(self, **kwargs)

    @add_refer_to("DataFrame.query")
    def query(self, expr, **kwargs):
        """
        Query columns of the `QueryCompiler` with a boolean expression.

        Parameters
        ----------
        expr : str
        **kwargs : kwargs

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing the rows where the boolean expression is satisfied.
        """
        return DataFrameDefault.register(pandas.DataFrame.query)(
            self, expr=expr, **kwargs
        )

    @add_refer_to("DataFrame.rank")
    def rank(self, **kwargs):
        """
        Get numerical rank for each value along axis.

        Parameters
        ----------
        axis : int
        method : {"average", "min", "max", "first", "dense"}
        numeric_only : bool
        na_option : {"keep", "top", "bottom"}
        ascending : bool
        pct : bool

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` of the same shape as `self`, where each element is the
            numerical rank of the corresponding value along row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.rank)(self, **kwargs)

    @add_refer_to("DataFrame.sort_index")
    def sort_index(self, **kwargs):
        """
        Sort data by index or column labels.

        Parameters
        ----------
        axis : int
        level : int, label or list of such
        ascending : bool
        inplace : bool
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        sort_remaining : bool
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing the data sorted by columns or indices.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_index)(self, **kwargs)

    @add_refer_to("DataFrame.melt")
    def melt(self, *args, **kwargs):
        """
        Unpivot `QueryCompiler` data from wide to long format.

        Parameters
        ----------
        id_vars : list of labels, optional
        value_vars : list of labels, optional
        var_name : label
        value_name : label
        col_level : int or label
        ignore_index : bool

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with unpivoted data.
        """
        return DataFrameDefault.register(pandas.DataFrame.melt)(self, *args, **kwargs)

    @add_refer_to("DataFrame.sort_values")
    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        """
        Reorder the columns based on the lexicographic order of the given rows.

        Parameters
        ----------
        rows : label or list of labels
            The row or rows to sort by.
        ascending : bool
            Sort in ascending order (True) or descending order (False).
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` that contains result of the sort.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
            self, by=rows, axis=1, ascending=ascending, **kwargs
        )

    @add_refer_to("DataFrame.sort_values")
    def sort_rows_by_column_values(self, rows, ascending=True, **kwargs):
        """
        Reorder the rows based on the lexicographic order of the given columns.

        Parameters
        ----------
        columns : label or list of labels
            The column or columns to sort by.
        ascending : bool
            Sort in ascending order (True) or descending order (False).
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` that contains result of the sort.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
            self, by=rows, axis=0, ascending=ascending, **kwargs
        )

    # END Abstract map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    @doc(
        __doc_numeric_method_template,
        params="""q : list-like
        axis : int
        numeric_only : bool
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}""",
        method="value at the given quantile",
        link="quantile",
    )
    def quantile_for_list_of_values(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    # END Abstract map across rows/columns

    # Abstract __getitem__ methods
    def getitem_array(self, key):
        """
        Mask `QueryCompiler` with `key`.

        Parameters
        ----------
        key : BaseQueryCompiler, numpy.ndarray or list of column labels
            Boolean mask represented by `QueryCompiler` or `numpy.ndarray` of the same
            shape as `self`. Or enumeratable of columns to pick.

        Returns
        -------
        BaseQueryCompiler
            New masked `QueryCompiler`.
        """

        def getitem_array(df, key):
            return df[key]

        return DataFrameDefault.register(getitem_array)(self, key)

    def getitem_column_array(self, key, numeric=False):
        """
        Get column data for target labels.

        Parameters
        ----------
        key : list-like
            Target labels by which to retrieve data.
        numeric : bool
            Whether or not the key passed in represents the numeric index
            or the named index.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` that contains specified columns.
        """

        def get_column(df, key):
            if numeric:
                return df.iloc[:, key]
            else:
                return df[key]

        return DataFrameDefault.register(get_column)(self, key=key)

    def getitem_row_array(self, key):
        """
        Get row data for target indices.

        Parameters
        ----------
        key : list-like
            Numeric indices of the rows to pick.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` that contains specified rows.
        """

        def get_row(df, key):
            return df.iloc[key]

        return DataFrameDefault.register(get_row)(self, key=key)

    # END Abstract __getitem__ methods

    # Abstract insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    def insert(self, loc, column, value):
        """
        Insert new column.

        Parameters
        ----------
        loc : int
            Insertion position.
        column : label
            Label of the new column.
        value : One-column BaseQueryCompiler, 1D array or scalar
            Data to fill new column with.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` with new column inserted.
        """
        return DataFrameDefault.register(pandas.DataFrame.insert, inplace=True)(
            self, loc=loc, column=column, value=value
        )

    # END Abstract insert

    # Abstract drop
    def drop(self, index=None, columns=None):
        """
        Drop specified rows or columns.

        Parameters
        ----------
        index : list of labels
            Labels of rows to drop.
        columns : list of labels
            Labels of columns to drop.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with removed data.
        """
        if index is None and columns is None:
            return self
        else:
            return DataFrameDefault.register(pandas.DataFrame.drop)(
                self, index=index, columns=columns
            )

    # END drop

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        """
        Apply passed function across given axis.

        Parameters
        ----------
        func : callable(pandas.Series) -> scalar, str, list or dict of such
            The function to apply to each column or row.
        axis : int
            Target axis to apply the function along.
        *args : args
            Argument to pass to `func`.
        **kwargs : kwargs
            Arguments to pass to `func`.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` that contain the results of execution and built by
            the following rules:
            - Labels of specified axis is the passed functions names.
            - Labels of the opposite axis is preserved.
            - Each element is the result of execution corresponding function under
              corresponding row/column.
        """
        return DataFrameDefault.register(pandas.DataFrame.apply)(
            self, func=func, axis=axis, *args, **kwargs
        )

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    # FIXME: `map_args` and `reduce_args` leaked there from `PandasQueryCompiler.groupby_*`,
    # pandas backend implements groupby via MapReduce approach, but for other backends these
    # parameters make no sense, they shouldn't be presented in a base class.
    __doc_groupby_method = """
    Group `QueryCompiler` data and {aggregation} for every group.

    Parameters
    ----------
    by : BaseQueryCompiler, column or index label, Grouper or list of such
        Object that determine groups.
    axis : int
        Axis to group and apply aggregation function along.
    groupby_args : dict
        GroupBy parameters in the format of ``modin.pandas.DataFrame.groupby`` signature.
    map_args : dict
        If GroupBy implemented with MapReduce approach, specifies arguments to pass to
        the aggregation function at the map phase.
    reduce_args : dict, optional
        If GroupBy implemented with MapReduce approach, specifies arguments to pass to
        the aggregation function at the reduce phase.
    numeric_only : bool, default: True
        Whether or not to drop non-numeric columns before executing GroupBy.
    drop : bool, default: False
        If `by` is a `QueryCompiler` indicates whether or not by-data came
        from the `self`.

    Returns
    -------
    BaseQueryCompiler
        `QueryCompiler` containing the result of groupby aggregation built by the
        following rules:
            - Labels on the opposit of specified axis is preserved.
            - If groupby_args["as_index"] is True then labels on the specified axis
              is the group names, otherwise labels would be default: 0, 1 ... n.
            - If groupby_args["as_index"] is False, then first N columns/rows of the frame
              contain group names, where N is the columns/rows to group on.
            - Each element of `QueryCompiler` is the {result} for the
              corresponding group and column/row.
    """

    @doc(
        __doc_groupby_method,
        aggregation="count non-null values",
        result="number of non-null values",
    )
    def groupby_count(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.count)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(
        __doc_groupby_method,
        aggregation="check whether any element is True",
        result="boolean of whether there is any element which is True",
    )
    def groupby_any(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.any)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(
        __doc_groupby_method,
        aggregation="get the minimum value",
        result="minimum value",
    )
    def groupby_min(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.min)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(__doc_groupby_method, aggregation="compute production", result="product")
    def groupby_prod(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.prod)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(
        __doc_groupby_method,
        aggregation="get the maximum value",
        result="maximum value",
    )
    def groupby_max(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.max)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(
        __doc_groupby_method,
        aggregation="check whether all elements are True",
        result="boolean of whether all elements are True",
    )
    def groupby_all(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.all)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(__doc_groupby_method, aggregation="compute sum", result="sum")
    def groupby_sum(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.sum)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    @doc(
        __doc_groupby_method,
        aggregation="get the number of elements",
        result="number of elements",
    )
    def groupby_size(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.size)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
            method="size",
        )

    def groupby_agg(
        self,
        by,
        is_multi_by,
        axis,
        agg_func,
        agg_args,
        agg_kwargs,
        groupby_kwargs,
        drop=False,
    ):
        """
        Group `QueryCompiler` data and apply passed aggregation function.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        is_multi_by : bool
            If `by` is a `QueryCompiler` or list of such indicates whether it's
            grouping on multiple columns/rows.
        axis : int
            Axis to group and apply aggregation function along.
        agg_func : dict or callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the pandas GroupBy object.
        agg_args : dict
            Positional arguments to pass to the `agg_func`.
        agg_kwargs : dict
            Key arguments to pass to the `agg_func`.
        groupby_kwargs : dict
            GroupBy parameters in the format of ``modin.pandas.DataFrame.groupby`` signature.
        drop : bool, default: False
            If `by` is a `QueryCompiler` indicates whether or not by-data came
            from the `self`.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` containing the result of groupby aggregation.
        """
        if isinstance(by, type(self)) and len(by.columns) == 1:
            by = by.columns[0] if drop else by.to_pandas().squeeze()
        elif isinstance(by, type(self)):
            by = list(by.columns)

        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.aggregate)(
            self,
            by=by,
            is_multi_by=is_multi_by,
            axis=axis,
            agg_func=agg_func,
            groupby_args=groupby_kwargs,
            agg_args=agg_kwargs,
            drop=drop,
        )

    # END Manual Partitioning methods

    @add_refer_to("DataFrame.unstack")
    def unstack(self, level, fill_value):
        """
        Pivot a level of the (necessarily hierarchical) index labels.

        Parameters
        ----------
        level : int or label
        fill_value : scalar or dict

        Returns
        -------
        BaseQueryCompiler
        """
        return DataFrameDefault.register(pandas.DataFrame.unstack)(
            self, level=level, fill_value=fill_value
        )

    @add_refer_to("DataFrame.pivot")
    def pivot(self, index, columns, values):
        """
        Produce pivot table based on column values.

        Parameters
        ----------
        index : label or list of such, pandas.Index, optional
        columns : label or list of such
        values : label or list of such, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing pivot table.
        """
        return DataFrameDefault.register(pandas.DataFrame.pivot)(
            self, index=index, columns=columns, values=values
        )

    @add_refer_to("DataFrame.pivot_table")
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
    ):
        """
        Create a spreadsheet-style pivot table from underlying data.

        Parameters
        ----------
        index : label, pandas.Grouper, array or list of such
        values : label, optional
        columns : column, pandas.Grouper, array or list of such
        aggfunc : callable(pandas.Series) -> scalar, dict of list of such
        fill_value : scalar, optional
        margins : bool
        dropna: bool
        margins_name : str
        observed : bool

        Returns
        -------
        BaseQueryCompiler
        """
        return DataFrameDefault.register(pandas.DataFrame.pivot_table)(
            self,
            index=index,
            values=values,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
        )

    @add_refer_to("get_dummies")
    def get_dummies(self, columns, **kwargs):
        """
        Convert categorical variables to dummy variables for certain columns.

        Parameters
        ----------
        columns : label or list of such
            Columns to convert.
        prefix : str or list of such
        prefix_sep : str
        dummy_na : bool
        drop_first : bool
        dtype : dtype

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with categorical variables converted to dummy.
        """

        def get_dummies(df, columns, **kwargs):
            return pandas.get_dummies(df, columns=columns, **kwargs)

        return DataFrameDefault.register(get_dummies)(self, columns=columns, **kwargs)

    @_add_one_column_warning
    def repeat(self, repeats):
        """
        Repeat each element of one-column `QueryCompiler` given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            `QueryCompiler`.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with repeated elements.
        """
        return SeriesDefault.register(pandas.Series.repeat)(self, repeats=repeats)

    # Indexing

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    def get_axis(self, axis):
        """
        Return index labels of the specified axis.

        Parameters
        ----------
        axis : int
            Axis to return labels on.

        Returns
        -------
        pandas.Index
        """
        return self.index if axis == 0 else self.columns

    def view(self, index=None, columns=None):
        """
        Mask `QueryCompiler` with passed keys.

        Parameters
        ----------
        index : list of ints
            Positional indices of rows to grab.
        columns : list of ints
            Positional indices of columns to grab.

        Returns
        -------
        BaseQueryCompiler
            New masked `QueryCompiler`.
        """
        index = [] if index is None else index
        columns = [] if columns is None else columns

        def applyier(df):
            return df.iloc[index, columns]

        return DataFrameDefault.register(applyier)(self)

    def insert_item(self, axis, loc, value, how="inner", replace=False):
        """
        Insert rows/columns defined by `value` at the specified position.

        If frames are not aligned along specified axis, first perform frames allignment.

        Parameters
        ----------
        axis : int
            Axis to insert along. 0 means insert rows, when 1 means insert columns.
        loc : int
            Position to insert `value`.
        value : BaseQueryCompiler
            Rows/columns to insert.
        how : {"inner", "outer", "left", "right"}
            Type of join that will be used if frames are not aligned.
        replace : bool, default: False
            Whether to insert item after column/row at `loc-th` position or to replace
            it by `value`.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with inserted values.
        """
        assert isinstance(value, type(self))

        def mask(idx):
            if len(idx) == len(self.get_axis(axis)):
                return self
            return (
                self.getitem_column_array(idx, numeric=True)
                if axis
                else self.getitem_row_array(idx)
            )

        if 0 <= loc < len(self.get_axis(axis)):
            first_mask = mask(list(range(loc)))
            second_mask_loc = loc + 1 if replace else loc
            second_mask = mask(list(range(second_mask_loc, len(self.get_axis(axis)))))
            return first_mask.concat(axis, [value, second_mask], join=how, sort=False)
        else:
            return self.concat(axis, [value], join=how, sort=False)

    def setitem(self, axis, key, value):
        """
        Set the row/column defined by `key` to the `value` provided.

        Parameters
        ----------
        axis : int
            Axis to set `value` along. 0 means set row, 1 means set column.
        key : label
            Row/column label to set `value` in.
        value : BaseQueryCompiler, list-like or scalar
            Define new row/column value.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with updated `key` value.
        """

        def setitem(df, axis, key, value):
            if is_scalar(key) and isinstance(value, pandas.DataFrame):
                value = value.squeeze()
            if not axis:
                df[key] = value
            else:
                df.loc[key] = value
            return df

        return DataFrameDefault.register(setitem)(self, axis=axis, key=key, value=value)

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        """
        Update `QueryCompiler` elements at the specified positions by passed values.

        Parameters
        ----------
        row_numeric_index : list of ints
            Row positions to write value.
        col_numeric_index : list of ints
            Column positions to write value
        broadcasted_items : 2D-array
            Values to write. Have to be same size as defined by `row_numeric_index`
            and `col_numeric_index`.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` with updated values.
        """

        def write_items(df, broadcasted_items):
            if isinstance(df.iloc[row_numeric_index, col_numeric_index], pandas.Series):
                broadcasted_items = broadcasted_items.squeeze()
            df.iloc[
                list(row_numeric_index), list(col_numeric_index)
            ] = broadcasted_items
            return df

        return DataFrameDefault.register(write_items)(
            self, broadcasted_items=broadcasted_items
        )

    # END Abstract methods for QueryCompiler

    @property
    def __constructor__(self):
        """By default, constructor method will invoke an init."""
        return type(self)

    # __delitem__
    # This will change the shape of the resulting data.
    def delitem(self, key):
        """
        Drop `key` column.

        Parameters
        ----------
        key : label
            Column name to drop.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` without `key` column.
        """
        return self.drop(columns=[key])

    # END __delitem__

    def has_multiindex(self, axis=0):
        """
        Check if specified axis is indexed by MultiIndex.

        Parameters
        ----------
        axis : int, default: 0
            The axis to check (0 - index, 1 - columns).

        Returns
        -------
        bool
            True if index at specified axis is MultiIndex and False otherwise.
        """
        if axis == 0:
            return isinstance(self.index, pandas.MultiIndex)
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    def get_index_name(self, axis=0):
        """
        Get index name of specified axis.

        Parameters
        ----------
        axis: int, default: 0
            Axis to get index name on.

        Returns
        -------
        hashable
            Index name, None for MultiIndex.
        """
        return self.get_axis(axis).name

    def set_index_name(self, name, axis=0):
        """
        Set index name for the specified axis.

        Parameters
        ----------
        name : hashable
            New index name.
        axis : int, default: 0
            Axis to set name along.
        """
        self.get_axis(axis).name = name

    def get_index_names(self, axis=0):
        """
        Get index names of specified axis.

        Parameters
        ----------
        axis : int, default: 0
            Axis to get index names on.

        Returns
        -------
        list
            Index names.
        """
        return self.get_axis(axis).names

    def set_index_names(self, names, axis=0):
        """
        Set index names for the specified axis.

        Parameters
        ----------
        names : list
            New index names.
        axis : int, default: 0
            Axis to set names along.
        """
        self.get_axis(axis).names = names

    # DateTime methods

    def dt_ceil(self, freq, ambiguous, nonexistent):
        return DateTimeDefault.register(pandas.Series.dt.ceil)(
            self, freq, ambiguous, nonexistent
        )

    @_add_one_column_warning
    @add_refer_to("Series.dt.components")
    def dt_components(self):
        """
        Spread each date-time value into its components (days, hours, minutes...).

        Returns
        -------
        BaseQueryCompiler
        """
        return DateTimeDefault.register(pandas.Series.dt.components)(self)

    @_doc_dt_timestamp(property="the date without timezone information", method="date")
    def dt_date(self):
        return DateTimeDefault.register(pandas.Series.dt.date)(self)

    @_doc_dt_timestamp(property="day component", method="day")
    def dt_day(self):
        return DateTimeDefault.register(pandas.Series.dt.day)(self)

    @_doc_dt_timestamp(property="day name", method="day_name")
    def dt_day_name(self, locale):
        return DateTimeDefault.register(pandas.Series.dt.day_name)(self, locale)

    @_doc_dt_timestamp(property="integer day of week", method="dayofweek")
    # FIXME: `dt_dayofweek` is an alias for `dt_weekday`, one of them should
    # be removed.
    def dt_dayofweek(self):
        return DateTimeDefault.register(pandas.Series.dt.dayofweek)(self)

    @_doc_dt_timestamp(property="day of year", method="dayofyear")
    def dt_dayofyear(self):
        return DateTimeDefault.register(pandas.Series.dt.dayofyear)(self)

    @_doc_dt_interval(property="days", method="days")
    def dt_days(self):
        return DateTimeDefault.register(pandas.Series.dt.days)(self)

    @_doc_dt_timestamp(property="number of days in month", method="days_in_month")
    # FIXME: `dt_days_in_month` is an alias for `dt_daysinmonth`, one of them should
    # be removed.
    def dt_days_in_month(self):
        return DateTimeDefault.register(pandas.Series.dt.days_in_month)(self)

    @_doc_dt_timestamp(property="number of days in month", method="daysinmonth")
    def dt_daysinmonth(self):
        return DateTimeDefault.register(pandas.Series.dt.daysinmonth)(self)

    def dt_end_time(self):
        return DateTimeDefault.register(pandas.Series.dt.end_time)(self)

    def dt_floor(self, freq, ambiguous, nonexistent):
        return DateTimeDefault.register(pandas.Series.dt.floor)(
            self, freq, ambiguous, nonexistent
        )

    @_add_one_column_warning
    @add_refer_to("Series.dt.freq")
    def dt_freq(self):
        """
        Get the time frequency of the underlying time-series data.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` containing a single value, the frequency of the data.
        """
        return DateTimeDefault.register(pandas.Series.dt.freq)(self)

    @_doc_dt_timestamp(property="hour", method="hour")
    def dt_hour(self):
        return DateTimeDefault.register(pandas.Series.dt.hour)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether corresponding year is leap",
        method="is_leap_year",
    )
    def dt_is_leap_year(self):
        return DateTimeDefault.register(pandas.Series.dt.is_leap_year)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the last day of the month",
        method="is_month_end",
    )
    def dt_is_month_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_month_end)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the first day of the month",
        method="is_month_start",
    )
    def dt_is_month_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_month_start)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the last day of the quarter",
        method="is_quarter_end",
    )
    def dt_is_quarter_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_quarter_end)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the first day of the quarter",
        method="is_quarter_start",
    )
    def dt_is_quarter_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_quarter_start)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the last day of the year",
        method="is_year_end",
    )
    def dt_is_year_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_year_end)(self)

    @_doc_dt_timestamp(
        property="the boolean of whether the date is the first day of the year",
        method="is_year_start",
    )
    def dt_is_year_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_year_start)(self)

    @_doc_dt_timestamp(property="microseconds component", method="microsecond")
    def dt_microsecond(self):
        return DateTimeDefault.register(pandas.Series.dt.microsecond)(self)

    @_doc_dt_interval(property="microseconds component", method="microseconds")
    def dt_microseconds(self):
        return DateTimeDefault.register(pandas.Series.dt.microseconds)(self)

    @_doc_dt_timestamp(property="minute component", method="minute")
    def dt_minute(self):
        return DateTimeDefault.register(pandas.Series.dt.minute)(self)

    @_doc_dt_timestamp(property="month component", month="month")
    def dt_month(self):
        return DateTimeDefault.register(pandas.Series.dt.month)(self)

    @_doc_dt_timestamp(property="the month name", method="month name")
    def dt_month_name(self, locale):
        return DateTimeDefault.register(pandas.Series.dt.month_name)(self, locale)

    @_doc_dt_timestamp(property="nanoseconds component", method="nanosecond")
    def dt_nanosecond(self):
        return DateTimeDefault.register(pandas.Series.dt.nanosecond)(self)

    @_doc_dt_interval(property="nanoseconds component", method="nanoseconds")
    def dt_nanoseconds(self):
        return DateTimeDefault.register(pandas.Series.dt.nanoseconds)(self)

    @_add_one_column_warning
    @add_refer_to("Series.dt.normalize")
    def dt_normalize(self):
        """
        Set the time component of each date-time value to midnight.

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing date-time values with midnight time.
        """
        return DateTimeDefault.register(pandas.Series.dt.normalize)(self)

    @_doc_dt_timestamp(property="quarter component", method="quarter")
    def dt_quarter(self):
        return DateTimeDefault.register(pandas.Series.dt.quarter)(self)

    def dt_qyear(self):
        return DateTimeDefault.register(pandas.Series.dt.qyear)(self)

    def dt_round(self, freq, ambiguous, nonexistent):
        return DateTimeDefault.register(pandas.Series.dt.round)(
            self, freq, ambiguous, nonexistent
        )

    @_doc_dt_timestamp(property="seconds component", method="second")
    def dt_second(self):
        return DateTimeDefault.register(pandas.Series.dt.second)(self)

    @_doc_dt_interval(property="seconds component", method="seconds")
    def dt_seconds(self):
        return DateTimeDefault.register(pandas.Series.dt.seconds)(self)

    def dt_start_time(self):
        return DateTimeDefault.register(pandas.Series.dt.start_time)(self)

    def dt_strftime(self, date_format):
        return DateTimeDefault.register(pandas.Series.dt.strftime)(self, date_format)

    @_doc_dt_timestamp(property="time component", method="time")
    def dt_time(self):
        return DateTimeDefault.register(pandas.Series.dt.time)(self)

    @_doc_dt_timestamp(
        property="time component with timezone information", method="timetz"
    )
    def dt_timetz(self):
        return DateTimeDefault.register(pandas.Series.dt.timetz)(self)

    def dt_to_period(self, freq):
        return DateTimeDefault.register(pandas.Series.dt.to_period)(self, freq)

    def dt_to_pydatetime(self):
        """
        Convert underlying data to array of python native `datetime`.

        BaseQueryCompiler
            New `QueryCompiler` containing 1D array of `datetime` objects.
        """
        return DateTimeDefault.register(pandas.Series.dt.to_pydatetime)(self)

    # FIXME: there are no references to this method, we should either remove it
    # or add a call reference at the DataFrame level.
    def dt_to_pytimedelta(self):
        """
        Convert underlying data to array of python native `datetime.timedelta`.

        BaseQueryCompiler
            New `QueryCompiler` containing 1D array of `datetime.timedelta`.
        """
        return DateTimeDefault.register(pandas.Series.dt.dt_to_pytimedelta)(self)

    def dt_to_timestamp(self):
        return DateTimeDefault.register(pandas.Series.dt.to_timestamp)(self)

    @_doc_dt_interval(property="duration in seconds", method="total_seconds")
    def dt_total_seconds(self):
        return DateTimeDefault.register(pandas.Series.dt.dt_total_seconds)(self)

    @_add_one_column_warning
    @add_refer_to("Series.dt.tz")
    def dt_tz(self):
        """
        Get the time-zone of the underlying time-series data.

        Returns
        -------
        BaseQueryCompiler
            `QueryCompiler` containing a single value, time-zone of the data.
        """
        return DateTimeDefault.register(pandas.Series.dt.dt_tz)(self)

    @_add_one_column_warning
    @add_refer_to("Series.dt.tz_convert")
    def dt_tz_convert(self, tz):
        """
        Convert time-series data to the specified time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, optional

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing values with converted time zone.
        """
        return DateTimeDefault.register(pandas.Series.dt.tz_convert)(self, tz)

    @_add_one_column_warning
    @add_refer_to("Series.dt.tz_localize")
    def dt_tz_localize(self, tz, ambiguous, nonexistent):
        """
        Localize tz-naive to tz-aware.

        Parameters
        ----------
        tz : str, pytz.timezone, optional
        ambiguous : "inner", "NaT", bool mask
        nonexistent : "shift_forward", "shift_backward, "NaT", pandas.timedelta, default: "raise"

        Returns
        -------
        BaseQueryCompiler
            New `QueryCompiler` containing values with localized time zone.
        """
        return DateTimeDefault.register(pandas.Series.dt.tz_localize)(
            self, tz, ambiguous, nonexistent
        )

    @_doc_dt_timestamp(property="week component", method="week")
    def dt_week(self):
        return DateTimeDefault.register(pandas.Series.dt.week)(self)

    @_doc_dt_timestamp(property="integer day of week", method="weekday")
    def dt_weekday(self):
        return DateTimeDefault.register(pandas.Series.dt.weekday)(self)

    @_doc_dt_timestamp(property="week of year", method="weekofyear")
    def dt_weekofyear(self):
        return DateTimeDefault.register(pandas.Series.dt.weekofyear)(self)

    @_doc_dt_timestamp(property="year component", method="year")
    def dt_year(self):
        return DateTimeDefault.register(pandas.Series.dt.year)(self)

    # End of DateTime methods

    # Resample methods

    resample_agg_df = ResampleDefault.register(pandas.core.resample.Resampler.aggregate)
    resample_agg_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.aggregate, squeeze_self=True
    )
    resample_app_df = ResampleDefault.register(pandas.core.resample.Resampler.apply)
    resample_app_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.apply, squeeze_self=True
    )
    resample_asfreq = ResampleDefault.register(pandas.core.resample.Resampler.asfreq)
    resample_backfill = ResampleDefault.register(
        pandas.core.resample.Resampler.backfill
    )
    resample_bfill = ResampleDefault.register(pandas.core.resample.Resampler.bfill)
    resample_count = ResampleDefault.register(pandas.core.resample.Resampler.count)
    resample_ffill = ResampleDefault.register(pandas.core.resample.Resampler.ffill)
    resample_fillna = ResampleDefault.register(pandas.core.resample.Resampler.fillna)
    resample_first = ResampleDefault.register(pandas.core.resample.Resampler.first)
    resample_get_group = ResampleDefault.register(
        pandas.core.resample.Resampler.get_group
    )
    resample_interpolate = ResampleDefault.register(
        pandas.core.resample.Resampler.interpolate
    )
    resample_last = ResampleDefault.register(pandas.core.resample.Resampler.last)
    resample_max = ResampleDefault.register(pandas.core.resample.Resampler.max)
    resample_mean = ResampleDefault.register(pandas.core.resample.Resampler.mean)
    resample_median = ResampleDefault.register(pandas.core.resample.Resampler.median)
    resample_min = ResampleDefault.register(pandas.core.resample.Resampler.min)
    resample_nearest = ResampleDefault.register(pandas.core.resample.Resampler.nearest)
    resample_nunique = ResampleDefault.register(pandas.core.resample.Resampler.nunique)
    resample_ohlc_df = ResampleDefault.register(pandas.core.resample.Resampler.ohlc)
    resample_ohlc_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.ohlc, squeeze_self=True
    )
    resample_pad = ResampleDefault.register(pandas.core.resample.Resampler.pad)
    resample_pipe = ResampleDefault.register(pandas.core.resample.Resampler.pipe)
    resample_prod = ResampleDefault.register(pandas.core.resample.Resampler.prod)
    resample_quantile = ResampleDefault.register(
        pandas.core.resample.Resampler.quantile
    )
    resample_sem = ResampleDefault.register(pandas.core.resample.Resampler.sem)
    resample_size = ResampleDefault.register(pandas.core.resample.Resampler.size)
    resample_std = ResampleDefault.register(pandas.core.resample.Resampler.std)
    resample_sum = ResampleDefault.register(pandas.core.resample.Resampler.sum)
    resample_transform = ResampleDefault.register(
        pandas.core.resample.Resampler.transform
    )
    resample_var = ResampleDefault.register(pandas.core.resample.Resampler.var)

    # End of Resample methods

    # Str methods

    str_capitalize = StrDefault.register(pandas.Series.str.capitalize)
    str_center = StrDefault.register(pandas.Series.str.center)
    str_contains = StrDefault.register(pandas.Series.str.contains)
    str_count = StrDefault.register(pandas.Series.str.count)
    str_endswith = StrDefault.register(pandas.Series.str.endswith)
    str_find = StrDefault.register(pandas.Series.str.find)
    str_findall = StrDefault.register(pandas.Series.str.findall)
    str_get = StrDefault.register(pandas.Series.str.get)
    str_index = StrDefault.register(pandas.Series.str.index)
    str_isalnum = StrDefault.register(pandas.Series.str.isalnum)
    str_isalpha = StrDefault.register(pandas.Series.str.isalpha)
    str_isdecimal = StrDefault.register(pandas.Series.str.isdecimal)
    str_isdigit = StrDefault.register(pandas.Series.str.isdigit)
    str_islower = StrDefault.register(pandas.Series.str.islower)
    str_isnumeric = StrDefault.register(pandas.Series.str.isnumeric)
    str_isspace = StrDefault.register(pandas.Series.str.isspace)
    str_istitle = StrDefault.register(pandas.Series.str.istitle)
    str_isupper = StrDefault.register(pandas.Series.str.isupper)
    str_join = StrDefault.register(pandas.Series.str.join)
    str_len = StrDefault.register(pandas.Series.str.len)
    str_ljust = StrDefault.register(pandas.Series.str.ljust)
    str_lower = StrDefault.register(pandas.Series.str.lower)
    str_lstrip = StrDefault.register(pandas.Series.str.lstrip)
    str_match = StrDefault.register(pandas.Series.str.match)
    str_normalize = StrDefault.register(pandas.Series.str.normalize)
    str_pad = StrDefault.register(pandas.Series.str.pad)
    str_partition = StrDefault.register(pandas.Series.str.partition)
    str_repeat = StrDefault.register(pandas.Series.str.repeat)
    str_replace = StrDefault.register(pandas.Series.str.replace)
    str_rfind = StrDefault.register(pandas.Series.str.rfind)
    str_rindex = StrDefault.register(pandas.Series.str.rindex)
    str_rjust = StrDefault.register(pandas.Series.str.rjust)
    str_rpartition = StrDefault.register(pandas.Series.str.rpartition)
    str_rsplit = StrDefault.register(pandas.Series.str.rsplit)
    str_rstrip = StrDefault.register(pandas.Series.str.rstrip)
    str_slice = StrDefault.register(pandas.Series.str.slice)
    str_slice_replace = StrDefault.register(pandas.Series.str.slice_replace)
    str_split = StrDefault.register(pandas.Series.str.split)
    str_startswith = StrDefault.register(pandas.Series.str.startswith)
    str_strip = StrDefault.register(pandas.Series.str.strip)
    str_swapcase = StrDefault.register(pandas.Series.str.swapcase)
    str_title = StrDefault.register(pandas.Series.str.title)
    str_translate = StrDefault.register(pandas.Series.str.translate)
    str_upper = StrDefault.register(pandas.Series.str.upper)
    str_wrap = StrDefault.register(pandas.Series.str.wrap)
    str_zfill = StrDefault.register(pandas.Series.str.zfill)

    # End of Str methods

    # Rolling methods

    rolling_aggregate = RollingDefault.register(
        pandas.core.window.rolling.Rolling.aggregate
    )
    rolling_apply = RollingDefault.register(pandas.core.window.rolling.Rolling.apply)
    rolling_corr = RollingDefault.register(pandas.core.window.rolling.Rolling.corr)
    rolling_count = RollingDefault.register(pandas.core.window.rolling.Rolling.count)
    rolling_cov = RollingDefault.register(pandas.core.window.rolling.Rolling.cov)
    rolling_kurt = RollingDefault.register(pandas.core.window.rolling.Rolling.kurt)
    rolling_max = RollingDefault.register(pandas.core.window.rolling.Rolling.max)
    rolling_mean = RollingDefault.register(pandas.core.window.rolling.Rolling.mean)
    rolling_median = RollingDefault.register(pandas.core.window.rolling.Rolling.median)
    rolling_min = RollingDefault.register(pandas.core.window.rolling.Rolling.min)
    rolling_quantile = RollingDefault.register(
        pandas.core.window.rolling.Rolling.quantile
    )
    rolling_skew = RollingDefault.register(pandas.core.window.rolling.Rolling.skew)
    rolling_std = RollingDefault.register(pandas.core.window.rolling.Rolling.std)
    rolling_sum = RollingDefault.register(pandas.core.window.rolling.Rolling.sum)
    rolling_var = RollingDefault.register(pandas.core.window.rolling.Rolling.var)

    # End of Rolling methods

    # Window methods

    window_mean = RollingDefault.register(pandas.core.window.Window.mean)
    window_std = RollingDefault.register(pandas.core.window.Window.std)
    window_sum = RollingDefault.register(pandas.core.window.Window.sum)
    window_var = RollingDefault.register(pandas.core.window.Window.var)

    # End of Window methods

    # Categories methods

    cat_codes = CatDefault.register(pandas.Series.cat.codes)

    # End of Categories methods

    # DataFrame methods

    invert = DataFrameDefault.register(pandas.DataFrame.__invert__)
    mad = DataFrameDefault.register(pandas.DataFrame.mad)
    kurt = DataFrameDefault.register(pandas.DataFrame.kurt)
    sum_min_count = sum
    prod_min_count = prod
    compare = DataFrameDefault.register(pandas.DataFrame.compare)

    # End of DataFrame methods
