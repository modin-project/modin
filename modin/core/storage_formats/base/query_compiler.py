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
Module contains class ``BaseQueryCompiler``.

``BaseQueryCompiler`` is a parent abstract class for any other query compiler class.
"""

import abc

from modin.core.dataframe.algebra.default2pandas import (
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
import modin.core.storage_formats.base.doc_utils as doc_utils
from modin.logging import LoggerMetaClass

from pandas.core.dtypes.common import is_scalar
import pandas.core.resample
import pandas
import numpy as np
from typing import List, Hashable


def _get_axis(axis):
    """
    Build index labels getter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to get labels from.

    Returns
    -------
    callable(BaseQueryCompiler) -> pandas.Index
    """

    def axis_getter(self):
        ErrorMessage.default_to_pandas(f"DataFrame.get_axis({axis})")
        return self.to_pandas().axes[axis]

    return axis_getter


def _set_axis(axis):
    """
    Build index labels setter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to set labels on.

    Returns
    -------
    callable(BaseQueryCompiler)
    """

    def axis_setter(self, labels):
        new_qc = DataFrameDefault.register(pandas.DataFrame.set_axis)(
            self, axis=axis, labels=labels
        )
        self.__dict__.update(new_qc.__dict__)

    return axis_setter


# FIXME: many of the BaseQueryCompiler methods are hiding actual arguments
# by using *args and **kwargs. They should be spread into actual parameters.
# Currently actual arguments are placed in the methods docstrings, but since they're
# not presented in the function's signature it makes linter to raise `PR02: unknown parameters`
# warning. For now, they're silenced by using `noqa` (Modin issue #3108).
class BaseQueryCompiler(
    abc.ABC, metaclass=type("", (abc.ABCMeta, LoggerMetaClass), {})
):
    """
    Abstract class that handles the queries to Modin dataframes.

    This class defines common query compilers API, most of the methods
    are already implemented and defaulting to pandas.

    Attributes
    ----------
    lazy_execution : bool
        Whether underlying execution engine is designed to be executed in a lazy mode only.
        If True, such QueryCompiler will be handled differently at the front-end in order
        to reduce execution triggering as much as possible.

    Notes
    -----
    See the Abstract Methods and Fields section immediately below this
    for a list of requirements for subclassing this object.
    """

    @abc.abstractmethod
    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """
        Do fallback to pandas for the passed function.

        Parameters
        ----------
        pandas_op : callable(pandas.DataFrame) -> object
            Function to apply to the casted to pandas frame.
        *args : iterable
            Positional arguments to pass to `pandas_op`.
        **kwargs : dict
            Key-value arguments to pass to `pandas_op`.

        Returns
        -------
        BaseQueryCompiler
            The result of the `pandas_op`, converted back to ``BaseQueryCompiler``.
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
        suffix : str
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

        Notes
        -----
        For copy, we don't want a situation where we modify the metadata of the
        copies if we end up modifying something here. We copy all of the metadata
        to prevent that.
        """
        return DataFrameDefault.register(pandas.DataFrame.copy)(self)

    # END Abstract copy

    # Abstract join and append helper functions

    def concat(self, axis, other, **kwargs):  # noqa: PR02
        """
        Concatenate `self` with passed query compilers along specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to concatenate along. 0 is for index and 1 is for columns.
        other : BaseQueryCompiler or list of such
            Objects to concatenate with `self`.
        join : {'outer', 'inner', 'right', 'left'}, default: 'outer'
            Type of join that will be used if indices on the other axis are different.
            (note: if specified, has to be passed as ``join=value``).
        ignore_index : bool, default: False
            If True, do not use the index values along the concatenation axis.
            The resulting axis will be labeled 0, …, n - 1.
            (note: if specified, has to be passed as ``ignore_index=value``).
        sort : bool, default: False
            Whether or not to sort non-concatenation axis.
            (note: if specified, has to be passed as ``sort=value``).
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

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
        Convert underlying query compilers data to ``pandas.DataFrame``.

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
        Build QueryCompiler from pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas DataFrame to convert from.
        data_cls : type
            :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` class
            (or its descendant) to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the pandas DataFrame.
        """
        pass

    # END To/From Pandas

    # From Arrow
    @classmethod
    @abc.abstractmethod
    def from_arrow(cls, at, data_cls):
        """
        Build QueryCompiler from Arrow Table.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.
        data_cls : type
            :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` class
            (or its descendant) to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the pandas DataFrame.
        """
        pass

    # END From Arrow

    # To NumPy

    def to_numpy(self, **kwargs):  # noqa: PR02
        """
        Convert underlying query compilers data to NumPy array.

        Parameters
        ----------
        dtype : dtype
            The dtype of the resulted array.
        copy : bool
            Whether to ensure that the returned value is not a view on another array.
        na_value : object
            The value to replace missing values with.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        np.ndarray
            The QueryCompiler converted to NumPy array.
        """
        return DataFrameDefault.register(pandas.DataFrame.to_numpy)(self, **kwargs)

    # END To NumPy

    # Dataframe exchange protocol

    @abc.abstractmethod
    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        """
        Get a DataFrame exchange protocol object representing data of the Modin DataFrame.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

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
            A dataframe object following the DataFrame protocol specification.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_dataframe(cls, df, data_cls):
        """
        Build QueryCompiler from a DataFrame object supporting the dataframe exchange protocol `__dataframe__()`.

        Parameters
        ----------
        df : DataFrame
            The DataFrame object supporting the dataframe exchange protocol.
        data_cls : type
            :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe` class
            (or its descendant) to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the DataFrame.
        """
        pass

    # END Dataframe exchange protocol

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    @doc_utils.doc_binary_method(operation="addition", sign="+")
    def add(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.add)(self, other=other, **kwargs)

    @doc_utils.add_refer_to("DataFrame.combine")
    def combine(self, other, **kwargs):  # noqa: PR02
        """
        Perform column-wise combine with another QueryCompiler with passed `func`.

        If axes are not equal, perform frames alignment first.

        Parameters
        ----------
        other : BaseQueryCompiler
            Left operand of the binary operation.
        func : callable(pandas.Series, pandas.Series) -> pandas.Series
            Function that takes two ``pandas.Series`` with aligned axes
            and returns one ``pandas.Series`` as resulting combination.
        fill_value : float or None
            Value to fill missing values with after frame alignment occurred.
        overwrite : bool
            If True, columns in `self` that do not exist in `other`
            will be overwritten with NaNs.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            Result of combine.
        """
        return BinaryDefault.register(pandas.DataFrame.combine)(
            self, other=other, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.combine_first")
    def combine_first(self, other, **kwargs):  # noqa: PR02
        """
        Fill null elements of `self` with value in the same location in `other`.

        If axes are not equal, perform frames alignment first.

        Parameters
        ----------
        other : BaseQueryCompiler
            Provided frame to use to fill null values from.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
        """
        return BinaryDefault.register(pandas.DataFrame.combine_first)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="equality comparison", sign="==")
    def eq(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.eq)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(operation="integer division", sign="//")
    def floordiv(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.floordiv)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(
        operation="greater than or equal comparison", sign=">=", op_type="comparison"
    )
    def ge(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.ge)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(
        operation="greater than comparison", sign=">", op_type="comparison"
    )
    def gt(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.gt)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(
        operation="less than or equal comparison", sign="<=", op_type="comparison"
    )
    def le(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.le)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(
        operation="less than comparison", sign="<", op_type="comparison"
    )
    def lt(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.lt)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(operation="modulo", sign="%")
    def mod(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.mod)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(operation="multiplication", sign="*")
    def mul(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.mul)(self, other=other, **kwargs)

    @doc_utils.add_refer_to("DataFrame.corr")
    def corr(self, **kwargs):  # noqa: PR02
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable(pandas.Series, pandas.Series) -> pandas.Series
            Correlation method.
        min_periods : int
            Minimum number of observations required per pair of columns
            to have a valid result. If fewer than `min_periods` non-NA values
            are present the result will be NA.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            Correlation matrix.
        """
        return DataFrameDefault.register(pandas.DataFrame.corr)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.cov")
    def cov(self, **kwargs):  # noqa: PR02
        """
        Compute pairwise covariance of columns, excluding NA/null values.

        Parameters
        ----------
        min_periods : int
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            Covariance matrix.
        """
        return DataFrameDefault.register(pandas.DataFrame.cov)(self, **kwargs)

    def dot(self, other, **kwargs):  # noqa: PR02
        """
        Compute the matrix multiplication of `self` and `other`.

        Parameters
        ----------
        other : BaseQueryCompiler or NumPy array
            The other query compiler or NumPy array to matrix multiply with `self`.
        squeeze_self : boolean
            If `self` is a one-column query compiler, indicates whether it represents Series object.
        squeeze_other : boolean
            If `other` is a one-column query compiler, indicates whether it represents Series object.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

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

    @doc_utils.doc_binary_method(
        operation="not equal comparison", sign="!=", op_type="comparison"
    )
    def ne(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.ne)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(operation="exponential power", sign="**")
    def pow(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.pow)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(
        operation="integer division", sign="//", self_on_right=True
    )
    def rfloordiv(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.rfloordiv)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="modulo", sign="%", self_on_right=True)
    def rmod(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.rmod)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(
        operation="exponential power", sign="**", self_on_right=True
    )
    def rpow(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.rpow)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="subtraction", sign="-", self_on_right=True)
    def rsub(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.rsub)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="division", sign="/", self_on_right=True)
    def rtruediv(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.rtruediv)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="subtraction", sign="-")
    def sub(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.sub)(self, other=other, **kwargs)

    @doc_utils.doc_binary_method(operation="division", sign="/")
    def truediv(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.truediv)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="conjunction", sign="&", op_type="logical")
    def __and__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__and__)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="disjunction", sign="|", op_type="logical")
    def __or__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__or__)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(
        operation="conjunction", sign="&", op_type="logical", self_on_right=True
    )
    def __rand__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__rand__)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(
        operation="disjunction", sign="|", op_type="logical", self_on_right=True
    )
    def __ror__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__ror__)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(
        operation="exclusive or", sign="^", op_type="logical", self_on_right=True
    )
    def __rxor__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__rxor__)(
            self, other=other, **kwargs
        )

    @doc_utils.doc_binary_method(operation="exclusive or", sign="^", op_type="logical")
    def __xor__(self, other, **kwargs):  # noqa: PR02
        return BinaryDefault.register(pandas.DataFrame.__xor__)(
            self, other=other, **kwargs
        )

    # FIXME: query compiler shoudln't care about differences between Frame and Series.
    # We should combine `df_update` and `series_update` into one method (Modin issue #3101).
    @doc_utils.add_refer_to("DataFrame.update")
    def df_update(self, other, **kwargs):  # noqa: PR02
        """
        Update values of `self` using non-NA values of `other` at the corresponding positions.

        If axes are not equal, perform frames alignment first.

        Parameters
        ----------
        other : BaseQueryCompiler
            Frame to grab replacement values from.
        join : {"left"}
            Specify type of join to align frames if axes are not equal
            (note: currently only one type of join is implemented).
        overwrite : bool
            Whether to overwrite every corresponding value of self, or only if it's NAN.
        filter_func : callable(pandas.Series, pandas.Series) -> numpy.ndarray<bool>
            Function that takes column of the self and return bool mask for values, that
            should be overwritten in the self frame.
        errors : {"raise", "ignore"}
            If "raise", will raise a ``ValueError`` if `self` and `other` both contain
            non-NA data in the same place.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated values.
        """
        return BinaryDefault.register(pandas.DataFrame.update, inplace=True)(
            self, other=other, **kwargs
        )

    @doc_utils.add_refer_to("Series.update")
    def series_update(self, other, **kwargs):  # noqa: PR02
        """
        Update values of `self` using values of `other` at the corresponding indices.

        Parameters
        ----------
        other : BaseQueryCompiler
            One-column query compiler with updated values.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated values.
        """
        return BinaryDefault.register(pandas.Series.update, inplace=True)(
            self,
            other=other,
            squeeze_self=True,
            squeeze_other=True,
            **kwargs,
        )

    @doc_utils.add_refer_to("DataFrame.clip")
    def clip(self, lower, upper, **kwargs):  # noqa: PR02
        """
        Trim values at input threshold.

        Parameters
        ----------
        lower : float or list-like
        upper : float or list-like
        axis : {0, 1}
        inplace : {False}
            This parameter serves the compatibility purpose. Always has to be False.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with values limited by the specified thresholds.
        """
        if isinstance(lower, BaseQueryCompiler):
            lower = lower.to_pandas().squeeze(1)
        if isinstance(upper, BaseQueryCompiler):
            upper = upper.to_pandas().squeeze(1)
        return DataFrameDefault.register(pandas.DataFrame.clip)(
            self, lower=lower, upper=upper, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.where")
    def where(self, cond, other, **kwargs):  # noqa: PR02
        """
        Update values of `self` using values from `other` at positions where `cond` is False.

        Parameters
        ----------
        cond : BaseQueryCompiler
            Boolean mask. True - keep the self value, False - replace by `other` value.
        other : BaseQueryCompiler or pandas.Series
            Object to grab replacement values from.
        axis : {0, 1}
            Axis to align frames along if axes of self, `cond` and `other` are not equal.
            0 is for index, when 1 is for columns.
        level : int or label, optional
            Level of MultiIndex to align frames along if axes of self, `cond`
            and `other` are not equal. Currently `level` parameter is not implemented,
            so only None value is acceptable.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with updated data.
        """
        return DataFrameDefault.register(pandas.DataFrame.where)(
            self, cond=cond, other=other, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.merge")
    def merge(self, right, **kwargs):  # noqa: PR02
        """
        Merge QueryCompiler objects using a database-style join.

        Parameters
        ----------
        right : BaseQueryCompiler
            QueryCompiler of the right frame to merge with.
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
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler that contains result of the merge.
        """
        return DataFrameDefault.register(pandas.DataFrame.merge)(
            self, right=right, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.join")
    def join(self, right, **kwargs):  # noqa: PR02
        """
        Join columns of another QueryCompiler.

        Parameters
        ----------
        right : BaseQueryCompiler
            QueryCompiler of the right frame to join with.
        on : label or list of such
        how : {"left", "right", "outer", "inner"}
        lsuffix : str
        rsuffix : str
        sort : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler that contains result of the join.
        """
        return DataFrameDefault.register(pandas.DataFrame.join)(self, right, **kwargs)

    # END Abstract inter-data operations

    # Abstract Transpose
    def transpose(self, *args, **kwargs):  # noqa: PR02
        """
        Transpose this QueryCompiler.

        Parameters
        ----------
        copy : bool
            Whether to copy the data after transposing.
        *args : iterable
            Serves the compatibility purpose. Does not affect the result.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            Transposed new QueryCompiler.
        """
        return DataFrameDefault.register(pandas.DataFrame.transpose)(
            self, *args, **kwargs
        )

    def columnarize(self):
        """
        Transpose this QueryCompiler if it has a single row but multiple columns.

        This method should be called for QueryCompilers representing a Series object,
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
        """
        Check whether this QueryCompiler can represent ``modin.pandas.Series`` object.

        Returns
        -------
        bool
            Return True if QueryCompiler has a single column or row, False otherwise.
        """
        return len(self.columns) == 1 or len(self.index) == 1

    # END Abstract Transpose

    # Abstract reindex/reset_index (may shuffle data)
    @doc_utils.add_refer_to("DataFrame.reindex")
    def reindex(self, axis, labels, **kwargs):  # noqa: PR02
        """
        Align QueryCompiler data with a new index along specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to align labels along. 0 is for index, 1 is for columns.
        labels : list-like
            Index-labels to align with.
        method : {None, "backfill"/"bfill", "pad"/"ffill", "nearest"}
            Method to use for filling holes in reindexed frame.
        fill_value : scalar
            Value to use for missing values in the resulted frame.
        limit : int
        tolerance : int
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with aligned axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.reindex)(
            self, axis=axis, labels=labels, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.reset_index")
    def reset_index(self, **kwargs):  # noqa: PR02
        """
        Reset the index, or a level of it.

        Parameters
        ----------
        drop : bool
            Whether to drop the reset index or insert it at the beginning of the frame.
        level : int or label, optional
            Level to remove from index. Removes all levels by default.
        col_level : int or label
            If the columns have multiple levels, determines which level the labels
            are inserted into.
        col_fill : label
            If the columns have multiple levels, determines how the other levels
            are named.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with reset index.
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
        drop : bool, default: True
            Whether or not to drop the columns provided in the `keys` argument.
        append : bool, default: True
            Whether or not to add the columns in `keys` as new levels appended to the
            existing index.

        Returns
        -------
        BaseQueryCompiler
            A new QueryCompiler with updated index.
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
        Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_increasing)(self)

    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_decreasing)(self)

    @doc_utils.doc_reduce_agg(
        method="number of non-NaN values", refer_to="count", extra_params=["**kwargs"]
    )
    def count(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.count)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="maximum value", refer_to="max", extra_params=["skipna", "**kwargs"]
    )
    def max(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.max)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="mean value", refer_to="mean", extra_params=["skipna", "**kwargs"]
    )
    def mean(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.mean)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="minimum value", refer_to="min", extra_params=["skipna", "**kwargs"]
    )
    def min(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.min)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="production",
        refer_to="prod",
        extra_params=["**kwargs"],
        params="axis : {0, 1}",
    )
    def prod(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.prod)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="sum",
        refer_to="sum",
        extra_params=["**kwargs"],
        params="axis : {0, 1}",
    )
    def sum(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.sum)(self, **kwargs)

    @doc_utils.add_refer_to("to_datetime")
    def to_datetime(self, *args, **kwargs):
        """
        Convert columns of the QueryCompiler to the datetime dtype.

        Parameters
        ----------
        *args : iterable
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with all columns converted to datetime dtype.
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
            QueryCompiler with absolute numeric value of each element.
        """
        return DataFrameDefault.register(pandas.DataFrame.abs)(self)

    def applymap(self, func, *args, **kwargs):
        """
        Apply passed function elementwise.

        Parameters
        ----------
        func : callable(scalar) -> scalar
            Function to apply to each element of the QueryCompiler.
        *args : iterable
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            Transformed QueryCompiler.
        """
        return DataFrameDefault.register(pandas.DataFrame.applymap)(
            self, func, *args, **kwargs
        )

    # FIXME: `**kwargs` which follows `numpy.conj` signature was inherited
    # from ``PandasQueryCompiler``, we should get rid of this dependency.
    # (Modin issue #3108)
    def conj(self, **kwargs):
        """
        Get the complex conjugate for every element of self.

        Parameters
        ----------
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with conjugate applied element-wise.

        Notes
        -----
        Please refer to ``numpy.conj`` for parameters description.
        """

        def conj(df, *args, **kwargs):
            return pandas.DataFrame(np.conj(df))

        return DataFrameDefault.register(conj)(self, **kwargs)

    # FIXME:
    #   1. This function takes Modin Series and DataFrames via `values` parameter,
    #      we should avoid leaking of the high-level objects to the query compiler level.
    #      (Modin issue #3106)
    #   2. Spread **kwargs into actual arguments (Modin issue #3108).
    def isin(self, **kwargs):  # noqa: PR02
        """
        Check for each element of `self` whether it's contained in passed `values`.

        Parameters
        ----------
        values : list-like, modin.pandas.Series, modin.pandas.DataFrame or dict
            Values to check elements of self in.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

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

    # FIXME: this method is not supposed to take any parameters (Modin issue #3108).
    def negative(self, **kwargs):
        """
        Change the sign for every value of self.

        Parameters
        ----------
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler

        Notes
        -----
        Be aware, that all QueryCompiler values have to be numeric.
        """
        return DataFrameDefault.register(pandas.DataFrame.__neg__)(self, **kwargs)

    def notna(self):
        """
        Check for each element of `self` whether it's existing (non-missing) value.

        Returns
        -------
        BaseQueryCompiler
            Boolean mask for `self` of whether an element at the corresponding
            position is not NaN.
        """
        return DataFrameDefault.register(pandas.DataFrame.notna)(self)

    @doc_utils.add_refer_to("DataFrame.round")
    def round(self, **kwargs):  # noqa: PR02
        """
        Round every numeric value up to specified number of decimals.

        Parameters
        ----------
        decimals : int or list-like
            Number of decimals to round each column to.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with rounded values.
        """
        return DataFrameDefault.register(pandas.DataFrame.round)(self, **kwargs)

    # FIXME:
    #   1. high-level objects leaks to the query compiler (Modin issue #3106).
    #   2. remove `inplace` parameter.
    @doc_utils.add_refer_to("DataFrame.replace")
    def replace(self, **kwargs):  # noqa: PR02
        """
        Replace values given in `to_replace` by `value`.

        Parameters
        ----------
        to_replace : scalar, list-like, regex, modin.pandas.Series, or None
        value : scalar, list-like, regex or dict
        inplace : {False}
            This parameter serves the compatibility purpose. Always has to be False.
        limit : int or None
        regex : bool or same types as `to_replace`
        method : {"pad", "ffill", "bfill", None}
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with all `to_replace` values replaced by `value`.
        """
        return DataFrameDefault.register(pandas.DataFrame.replace)(self, **kwargs)

    @doc_utils.add_one_column_warning
    # FIXME: adding refer-to note will create two instances of the "Notes" section,
    # this breaks numpydoc style rules and also crashes the doc-style checker script.
    # For now manually added the refer-to message.
    # @doc_utils.add_refer_to("Series.view")
    def series_view(self, **kwargs):  # noqa: PR02
        """
        Reinterpret underlying data with new dtype.

        Parameters
        ----------
        dtype : dtype
            Data type to reinterpret underlying data with.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler of the same data in memory, with reinterpreted values.

        Notes
        -----
            - Be aware, that if this method do fallback to pandas, then newly created
              QueryCompiler will be the copy of the original data.
            - Please refer to ``modin.pandas.Series.view`` for more information
              about parameters and output format.
        """
        return SeriesDefault.register(pandas.Series.view)(self, **kwargs)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("to_numeric")
    def to_numeric(self, *args, **kwargs):  # noqa: PR02
        """
        Convert underlying data to numeric dtype.

        Parameters
        ----------
        errors : {"ignore", "raise", "coerce"}
        downcast : {"integer", "signed", "unsigned", "float", None}
        *args : iterable
            Serves the compatibility purpose. Does not affect the result.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with converted to numeric values.
        """
        return SeriesDefault.register(pandas.to_numeric)(self, *args, **kwargs)

    # FIXME: get rid of `**kwargs` parameter (Modin issue #3108).
    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.unique")
    def unique(self, **kwargs):
        """
        Get unique values of `self`.

        Parameters
        ----------
        **kwargs : dict
            Serves compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with unique values.
        """
        return SeriesDefault.register(pandas.Series.unique)(self, **kwargs)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.searchsorted")
    def searchsorted(self, **kwargs):  # noqa: PR02
        """
        Find positions in a sorted `self` where `value` should be inserted to maintain order.

        Parameters
        ----------
        value : list-like
        side : {"left", "right"}
        sorter : list-like, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            One-column QueryCompiler which contains indices to insert.
        """
        return SeriesDefault.register(pandas.Series.searchsorted)(self, **kwargs)

    # END Abstract map partitions operations

    @doc_utils.add_refer_to("DataFrame.stack")
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
    def astype(self, col_dtypes, **kwargs):  # noqa: PR02
        """
        Convert columns dtypes to given dtypes.

        Parameters
        ----------
        col_dtypes : dict
            Map for column names and new dtypes.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated dtypes.
        """
        return DataFrameDefault.register(pandas.DataFrame.astype)(
            self, dtype=col_dtypes, **kwargs
        )

    @property
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

    # FIXME: we're handling level parameter at front-end, it shouldn't
    # propagate to the query compiler (Modin issue #3102)
    @doc_utils.add_refer_to("DataFrame.all")
    def all(self, **kwargs):  # noqa: PR02
        """
        Return whether all the elements are true, potentially over an axis.

        Parameters
        ----------
        axis : {0, 1}, optional
        bool_only : bool, optional
        skipna : bool
        level : int or label
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            If axis was specified return one-column QueryCompiler with index labels
            of the specified axis, where each row contains boolean of whether all elements
            at the corresponding row or column are True. Otherwise return QueryCompiler
            with a single bool of whether all elements are True.
        """
        return DataFrameDefault.register(pandas.DataFrame.all)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.any")
    def any(self, **kwargs):  # noqa: PR02
        """
        Return whether any element is true, potentially over an axis.

        Parameters
        ----------
        axis : {0, 1}, optional
        bool_only : bool, optional
        skipna : bool
        level : int or label
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            If axis was specified return one-column QueryCompiler with index labels
            of the specified axis, where each row contains boolean of whether any element
            at the corresponding row or column is True. Otherwise return QueryCompiler
            with a single bool of whether any element is True.
        """
        return DataFrameDefault.register(pandas.DataFrame.any)(self, **kwargs)

    def first_valid_index(self):
        """
        Return index label of first non-NaN/NULL value.

        Returns
        -------
        scalar
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.first_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    @doc_utils.add_refer_to("DataFrame.idxmax")
    def idxmax(self, **kwargs):  # noqa: PR02
        """
        Get position of the first occurrence of the maximum for each row or column.

        Parameters
        ----------
        axis : {0, 1}
        skipna : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            One-column QueryCompiler with index labels of the specified axis,
            where each row contains position of the maximum element for the
            corresponding row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmax)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.idxmin")
    def idxmin(self, **kwargs):  # noqa: PR02
        """
        Get position of the first occurrence of the minimum for each row or column.

        Parameters
        ----------
        axis : {0, 1}
        skipna : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            One-column QueryCompiler with index labels of the specified axis,
            where each row contains position of the minimum element for the
            corresponding row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmin)(self, **kwargs)

    def last_valid_index(self):
        """
        Return index label of last non-NaN/NULL value.

        Returns
        -------
        scalar
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.last_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    @doc_utils.doc_reduce_agg(
        method="median value", refer_to="median", extra_params=["skipna", "**kwargs"]
    )
    def median(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.median)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.memory_usage")
    def memory_usage(self, **kwargs):  # noqa: PR02
        """
        Return the memory usage of each column in bytes.

        Parameters
        ----------
        index : bool
        deep : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            One-column QueryCompiler with index labels of `self`, where each row
            contains the memory usage for the corresponding column.
        """
        return DataFrameDefault.register(pandas.DataFrame.memory_usage)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="number of unique values",
        refer_to="nunique",
        params="""
        axis : {0, 1}
        dropna : bool""",
        extra_params=["**kwargs"],
    )
    def nunique(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.nunique)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="value at the given quantile",
        refer_to="quantile",
        params="""
        q : float
        axis : {0, 1}
        numeric_only : bool
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}""",
        extra_params=["**kwargs"],
    )
    def quantile_for_single_value(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="unbiased skew", refer_to="skew", extra_params=["skipna", "**kwargs"]
    )
    def skew(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.skew)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="standard deviation of the mean",
        refer_to="sem",
        extra_params=["skipna", "ddof", "**kwargs"],
    )
    def sem(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.sem)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="standard deviation",
        refer_to="std",
        extra_params=["skipna", "ddof", "**kwargs"],
    )
    def std(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.std)(self, **kwargs)

    @doc_utils.doc_reduce_agg(
        method="variance", refer_to="var", extra_params=["skipna", "ddof", "**kwargs"]
    )
    def var(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.var)(self, **kwargs)

    # END Abstract column/row partitions reduce operations

    # Abstract column/row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    @doc_utils.add_refer_to("DataFrame.describe")
    def describe(self, **kwargs):  # noqa: PR02
        """
        Generate descriptive statistics.

        Parameters
        ----------
        percentiles : list-like
        include : "all" or list of dtypes, optional
        exclude : list of dtypes, optional
        datetime_is_numeric : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler object containing the descriptive statistics
            of the underlying data.
        """
        return DataFrameDefault.register(pandas.DataFrame.describe)(self, **kwargs)

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    @doc_utils.doc_cum_agg(method="sum", refer_to="cumsum")
    def cumsum(self, fold_axis, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.cumsum)(self, **kwargs)

    @doc_utils.doc_cum_agg(method="maximum", refer_to="cummax")
    def cummax(self, fold_axis, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.cummax)(self, **kwargs)

    @doc_utils.doc_cum_agg(method="minimum", refer_to="cummin")
    def cummin(self, fold_axis, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.cummin)(self, **kwargs)

    @doc_utils.doc_cum_agg(method="product", refer_to="cumprod")
    def cumprod(self, fold_axis, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.cumprod)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.diff")
    def diff(self, fold_axis, **kwargs):  # noqa: PR02
        """
        First discrete difference of element.

        Parameters
        ----------
        periods : int
        fold_axis : {0, 1}
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler of the same shape as `self`, where each element is the difference
            between the corresponding value and the previous value in this row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.diff)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.dropna")
    def dropna(self, **kwargs):  # noqa: PR02
        """
        Remove missing values.

        Parameters
        ----------
        axis : {0, 1}
        how : {"any", "all"}
        thresh : int, optional
        subset : list of labels
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with null values dropped along given axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.dropna)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.nlargest")
    def nlargest(self, n=5, columns=None, keep="first"):
        """
        Return the first `n` rows ordered by `columns` in descending order.

        Parameters
        ----------
        n : int, default: 5
        columns : list of labels, optional
            Column labels to order by.
            (note: this parameter can be omitted only for a single-column query compilers
            representing Series object, otherwise `columns` has to be specified).
        keep : {"first", "last", "all"}, default: "first"

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

    @doc_utils.add_refer_to("DataFrame.nsmallest")
    def nsmallest(self, n=5, columns=None, keep="first"):
        """
        Return the first `n` rows ordered by `columns` in ascending order.

        Parameters
        ----------
        n : int, default: 5
        columns : list of labels, optional
            Column labels to order by.
            (note: this parameter can be omitted only for a single-column query compilers
            representing Series object, otherwise `columns` has to be specified).
        keep : {"first", "last", "all"}, default: "first"

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

    @doc_utils.add_refer_to("DataFrame.eval")
    def eval(self, expr, **kwargs):
        """
        Evaluate string expression on QueryCompiler columns.

        Parameters
        ----------
        expr : str
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing the result of evaluation.
        """
        return DataFrameDefault.register(pandas.DataFrame.eval)(
            self, expr=expr, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.mode")
    def mode(self, **kwargs):  # noqa: PR02
        """
        Get the modes for every column or row.

        Parameters
        ----------
        axis : {0, 1}
        numeric_only : bool
        dropna : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with modes calculated along given axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.mode)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.fillna")
    def fillna(self, **kwargs):  # noqa: PR02
        """
        Replace NaN values using provided method.

        Parameters
        ----------
        value : scalar or dict
        method : {"backfill", "bfill", "pad", "ffill", None}
        axis : {0, 1}
        inplace : {False}
            This parameter serves the compatibility purpose. Always has to be False.
        limit : int, optional
        downcast : dict, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with all null values filled.
        """
        squeeze_self = kwargs.pop("squeeze_self", False)
        squeeze_value = kwargs.pop("squeeze_value", False)

        def fillna(df, value, **kwargs):
            if squeeze_self:
                df = df.squeeze(axis=1)
            if squeeze_value:
                value = value.squeeze(axis=1)
            return df.fillna(value, **kwargs)

        return DataFrameDefault.register(fillna)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.query")
    def query(self, expr, **kwargs):
        """
        Query columns of the QueryCompiler with a boolean expression.

        Parameters
        ----------
        expr : str
        **kwargs : dict

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the rows where the boolean expression is satisfied.
        """
        return DataFrameDefault.register(pandas.DataFrame.query)(
            self, expr=expr, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.rank")
    def rank(self, **kwargs):  # noqa: PR02
        """
        Compute numerical rank along the specified axis.

        By default, equal values are assigned a rank that is the average of the ranks
        of those values, this behavior can be changed via `method` parameter.

        Parameters
        ----------
        axis : {0, 1}
        method : {"average", "min", "max", "first", "dense"}
        numeric_only : bool
        na_option : {"keep", "top", "bottom"}
        ascending : bool
        pct : bool
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler of the same shape as `self`, where each element is the
            numerical rank of the corresponding value along row or column.
        """
        return DataFrameDefault.register(pandas.DataFrame.rank)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.sort_index")
    def sort_index(self, **kwargs):  # noqa: PR02
        """
        Sort data by index or column labels.

        Parameters
        ----------
        axis : {0, 1}
        level : int, label or list of such
        ascending : bool
        inplace : bool
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        sort_remaining : bool
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the data sorted by columns or indices.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_index)(self, **kwargs)

    @doc_utils.add_refer_to("DataFrame.melt")
    def melt(self, *args, **kwargs):  # noqa: PR02
        """
        Unpivot QueryCompiler data from wide to long format.

        Parameters
        ----------
        id_vars : list of labels, optional
        value_vars : list of labels, optional
        var_name : label
        value_name : label
        col_level : int or label
        ignore_index : bool
        *args : iterable
            Serves the compatibility purpose. Does not affect the result.
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with unpivoted data.
        """
        return DataFrameDefault.register(pandas.DataFrame.melt)(self, *args, **kwargs)

    @doc_utils.add_refer_to("DataFrame.sort_values")
    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):  # noqa: PR02
        """
        Reorder the columns based on the lexicographic order of the given rows.

        Parameters
        ----------
        rows : label or list of labels
            The row or rows to sort by.
        ascending : bool, default: True
            Sort in ascending order (True) or descending order (False).
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler that contains result of the sort.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
            self, by=rows, axis=1, ascending=ascending, **kwargs
        )

    @doc_utils.add_refer_to("DataFrame.sort_values")
    def sort_rows_by_column_values(
        self, columns, ascending=True, **kwargs
    ):  # noqa: PR02
        """
        Reorder the rows based on the lexicographic order of the given columns.

        Parameters
        ----------
        columns : label or list of labels
            The column or columns to sort by.
        ascending : bool, default: True
            Sort in ascending order (True) or descending order (False).
        kind : {"quicksort", "mergesort", "heapsort"}
        na_position : {"first", "last"}
        ignore_index : bool
        key : callable(pandas.Index) -> pandas.Index, optional
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler that contains result of the sort.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
            self, by=columns, axis=0, ascending=ascending, **kwargs
        )

    # END Abstract map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    @doc_utils.doc_reduce_agg(
        method="value at the given quantile",
        refer_to="quantile",
        params="""
        q : list-like
        axis : {0, 1}
        numeric_only : bool
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}""",
        extra_params=["**kwargs"],
    )
    def quantile_for_list_of_values(self, **kwargs):  # noqa: PR02
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    # END Abstract map across rows/columns

    # Abstract __getitem__ methods
    def getitem_array(self, key):
        """
        Mask QueryCompiler with `key`.

        Parameters
        ----------
        key : BaseQueryCompiler, np.ndarray or list of column labels
            Boolean mask represented by QueryCompiler or ``np.ndarray`` of the same
            shape as `self`, or enumerable of columns to pick.

        Returns
        -------
        BaseQueryCompiler
            New masked QueryCompiler.
        """
        if isinstance(key, type(self)):
            key = key.to_pandas().squeeze(axis=1)

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
        numeric : bool, default: False
            Whether or not the key passed in represents the numeric index
            or the named index.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler that contains specified columns.
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
            New QueryCompiler that contains specified rows.
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
            QueryCompiler with new column inserted.
        """

        def inserter(df, loc, column, value):
            if isinstance(value, pandas.DataFrame):
                value = value.squeeze(axis=1)
            df.insert(loc, column, value)
            return df

        return DataFrameDefault.register(inserter, inplace=True)(
            self, loc=loc, column=column, value=value
        )

    # END Abstract insert

    # Abstract drop
    def drop(self, index=None, columns=None):
        """
        Drop specified rows or columns.

        Parameters
        ----------
        index : list of labels, optional
            Labels of rows to drop.
        columns : list of labels, optional
            Labels of columns to drop.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with removed data.
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
    def apply(self, func, axis, raw=False, result_type=None, *args, **kwargs):
        """
        Apply passed function across given axis.

        Parameters
        ----------
        func : callable(pandas.Series) -> scalar, str, list or dict of such
            The function to apply to each column or row.
        axis : {0, 1}
            Target axis to apply the function along.
            0 is for index, 1 is for columns.
        raw : bool, default: False
            Whether to pass a high-level Series object (False) or a raw representation
            of the data (True).
        result_type : {"expand", "reduce", "broadcast", None}, default: None
            Determines how to treat list-like return type of the `func` (works only if
            a single function was passed):

            - "expand": expand list-like result into columns.
            - "reduce": keep result into a single cell (opposite of "expand").
            - "broadcast": broadcast result to original data shape (overwrite the existing column/row with the function result).
            - None: use "expand" strategy if Series is returned, "reduce" otherwise.
        *args : iterable
            Positional arguments to pass to `func`.
        **kwargs : dict
            Keyword arguments to pass to `func`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler that contains the results of execution and is built by
            the following rules:

            - Index of the specified axis contains: the names of the passed functions if multiple
              functions are passed, otherwise: indices of the `func` result if "expand" strategy
              is used, indices of the original frame if "broadcast" strategy is used, a single
              label "__reduced__" if "reduce" strategy is used.
            - Labels of the opposite axis are preserved.
            - Each element is the result of execution of `func` against
              corresponding row/column.
        """
        return DataFrameDefault.register(pandas.DataFrame.apply)(
            self,
            func=func,
            axis=axis,
            raw=raw,
            result_type=result_type,
            *args,
            **kwargs,
        )

    def explode(self, column):
        """
        Explode the given columns.

        Parameters
        ----------
        column : Union[Hashable, Sequence[Hashable]]
            The columns to explode.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler that contains the results of execution. For each row
            in the input QueryCompiler, if the selected columns each contain M
            items, there will be M rows created by exploding the columns.
        """
        return DataFrameDefault.register(pandas.DataFrame.explode)(self, column)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    # FIXME: `map_args` and `reduce_args` leaked there from `PandasQueryCompiler.groupby_*`,
    # pandas storage format implements groupby via TreeReduce approach, but for other storage formats these
    # parameters make no sense, they shouldn't be present in a base class.

    @doc_utils.doc_groupby_method(
        action="count non-null values",
        result="number of non-null values",
        refer_to="count",
    )
    def groupby_count(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.count)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="check whether any element is True",
        result="boolean of whether there is any element which is True",
        refer_to="any",
    )
    def groupby_any(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.any)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get the minimum value", result="minimum value", refer_to="min"
    )
    def groupby_min(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.min)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(result="product", refer_to="prod")
    def groupby_prod(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.prod)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get the maximum value", result="maximum value", refer_to="max"
    )
    def groupby_max(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.max)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="check whether all elements are True",
        result="boolean of whether all elements are True",
        refer_to="all",
    )
    def groupby_all(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.all)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(result="sum", refer_to="sum")
    def groupby_sum(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.sum)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get the number of elements",
        result="number of elements",
        refer_to="size",
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.size)(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
            method="size",
        )

    @doc_utils.add_refer_to("GroupBy.aggregate")
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
        """
        Group QueryCompiler data and apply passed aggregation function.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        agg_func : str, dict or callable(Series | DataFrame) -> scalar | Series | DataFrame
            Function to apply to the GroupBy object.
        axis : {0, 1}
            Axis to group and apply aggregation function along.
            0 is for index, when 1 is for columns.
        groupby_kwargs : dict
            GroupBy parameters as expected by ``modin.pandas.DataFrame.groupby`` signature.
        agg_args : list-like
            Positional arguments to pass to the `agg_func`.
        agg_kwargs : dict
            Key arguments to pass to the `agg_func`.
        how : {'axis_wise', 'group_wise', 'transform'}, default: 'axis_wise'
            How to apply passed `agg_func`:
                - 'axis_wise': apply the function against each row/column.
                - 'group_wise': apply the function against every group.
                - 'transform': apply the function against every group and broadcast
                  the result to the original Query Compiler shape.
        drop : bool, default: False
            If `by` is a QueryCompiler indicates whether or not by-data came
            from the `self`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing the result of groupby aggregation.
        """
        if isinstance(by, type(self)) and len(by.columns) == 1:
            by = by.columns[0] if drop else by.to_pandas().squeeze()
        # converting QC 'by' to a list of column labels only if this 'by' comes from the self (if drop is True)
        elif drop and isinstance(by, type(self)):
            by = list(by.columns)

        return GroupByDefault.register(GroupByDefault.get_aggregation_method(how))(
            self,
            by=by,
            agg_func=agg_func,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute the mean value", result="mean value", refer_to="mean"
    )
    def groupby_mean(
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
            agg_func="mean",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute unbiased skew", result="unbiased skew", refer_to="skew"
    )
    def groupby_skew(
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
            agg_func="skew",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute cumulative sum",
        result="sum of all the previous values",
        refer_to="cumsum",
    )
    def groupby_cumsum(
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
            agg_func="cumsum",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get cumulative maximum",
        result="maximum of all the previous values",
        refer_to="cummax",
    )
    def groupby_cummax(
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
            agg_func="cummax",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get cumulative minimum",
        result="minimum of all the previous values",
        refer_to="cummin",
    )
    def groupby_cummin(
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
            agg_func="cummin",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get cumulative production",
        result="production of all the previous values",
        refer_to="cumprod",
    )
    def groupby_cumprod(
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
            agg_func="cumprod",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute standard deviation", result="standard deviation", refer_to="std"
    )
    def groupby_std(
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
            agg_func="std",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute numerical rank", result="numerical rank", refer_to="rank"
    )
    def groupby_rank(
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
            agg_func="rank",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute variance", result="variance", refer_to="var"
    )
    def groupby_var(
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
            agg_func="var",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get the number of unique values",
        result="number of unique values",
        refer_to="nunique",
    )
    def groupby_nunique(
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
            agg_func="nunique",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get the median value", result="median value", refer_to="median"
    )
    def groupby_median(
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
            agg_func="median",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="compute specified quantile",
        result="quantile value",
        refer_to="quantile",
    )
    def groupby_quantile(
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
            agg_func="quantile",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="fill NaN values",
        result="`fill_value` if it was NaN, original value otherwise",
        refer_to="fillna",
    )
    def groupby_fillna(
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
            agg_func="fillna",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="get data types", result="data type", refer_to="dtypes"
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
            agg_func="dtypes",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    @doc_utils.doc_groupby_method(
        action="shift data with the specified settings",
        result="shifted value",
        refer_to="shift",
    )
    def groupby_shift(
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
            agg_func="shift",
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
        )

    # END Manual Partitioning methods

    @doc_utils.add_refer_to("DataFrame.unstack")
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

    @doc_utils.add_refer_to("DataFrame.pivot")
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
            New QueryCompiler containing pivot table.
        """
        return DataFrameDefault.register(pandas.DataFrame.pivot)(
            self, index=index, columns=columns, values=values
        )

    @doc_utils.add_refer_to("DataFrame.pivot_table")
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
        dropna : bool
        margins_name : str
        observed : bool
        sort : bool

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
            sort=sort,
        )

    @doc_utils.add_refer_to("get_dummies")
    def get_dummies(self, columns, **kwargs):  # noqa: PR02
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
        **kwargs : dict
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with categorical variables converted to dummy.
        """

        def get_dummies(df, columns, **kwargs):
            return pandas.get_dummies(df, columns=columns, **kwargs)

        return DataFrameDefault.register(get_dummies)(self, columns=columns, **kwargs)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.repeat")
    def repeat(self, repeats):
        """
        Repeat each element of one-column QueryCompiler given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            QueryCompiler.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with repeated elements.
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
        axis : {0, 1}
            Axis to return labels on.
            0 is for index, when 1 is for columns.

        Returns
        -------
        pandas.Index
        """
        return self.index if axis == 0 else self.columns

    def view(self, index=None, columns=None):
        """
        Mask QueryCompiler with passed keys.

        Parameters
        ----------
        index : list-like of ints, optional
            Positional indices of rows to grab.
        columns : list-like of ints, optional
            Positional indices of columns to grab.

        Returns
        -------
        BaseQueryCompiler
            New masked QueryCompiler.
        """
        index = slice(None) if index is None else index
        columns = slice(None) if columns is None else columns

        def applyier(df):
            return df.iloc[index, columns]

        return DataFrameDefault.register(applyier)(self)

    def insert_item(self, axis, loc, value, how="inner", replace=False):
        """
        Insert rows/columns defined by `value` at the specified position.

        If frames are not aligned along specified axis, perform frames alignment first.

        Parameters
        ----------
        axis : {0, 1}
            Axis to insert along. 0 means insert rows, when 1 means insert columns.
        loc : int
            Position to insert `value`.
        value : BaseQueryCompiler
            Rows/columns to insert.
        how : {"inner", "outer", "left", "right"}, default: "inner"
            Type of join that will be used if frames are not aligned.
        replace : bool, default: False
            Whether to insert item after column/row at `loc-th` position or to replace
            it by `value`.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with inserted values.
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
        axis : {0, 1}
            Axis to set `value` along. 0 means set row, 1 means set column.
        key : label
            Row/column label to set `value` in.
        value : BaseQueryCompiler, list-like or scalar
            Define new row/column value.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated `key` value.
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
        Update QueryCompiler elements at the specified positions by passed values.

        In contrast to ``setitem`` this method allows to do 2D assignments.

        Parameters
        ----------
        row_numeric_index : list of ints
            Row positions to write value.
        col_numeric_index : list of ints
            Column positions to write value.
        broadcasted_items : 2D-array
            Values to write. Have to be same size as defined by `row_numeric_index`
            and `col_numeric_index`.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with updated values.
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
        """
        Get query compiler constructor.

        By default, constructor method will invoke an init.

        Returns
        -------
        callable
        """
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
            New QueryCompiler without `key` column.
        """
        return self.drop(columns=[key])

    # END __delitem__

    def has_multiindex(self, axis=0):
        """
        Check if specified axis is indexed by MultiIndex.

        Parameters
        ----------
        axis : {0, 1}, default: 0
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
        axis : {0, 1}, default: 0
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
        axis : {0, 1}, default: 0
            Axis to set name along.
        """
        self.get_axis(axis).name = name

    def get_index_names(self, axis=0):
        """
        Get index names of specified axis.

        Parameters
        ----------
        axis : {0, 1}, default: 0
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
        axis : {0, 1}, default: 0
            Axis to set names along.
        """
        self.get_axis(axis).names = names

    # DateTime methods

    @doc_utils.doc_dt_round(refer_to="ceil")
    def dt_ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        return DateTimeDefault.register(pandas.Series.dt.ceil)(
            self, freq, ambiguous, nonexistent
        )

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.components")
    def dt_components(self):
        """
        Spread each date-time value into its components (days, hours, minutes...).

        Returns
        -------
        BaseQueryCompiler
        """
        return DateTimeDefault.register(pandas.Series.dt.components)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the date without timezone information", refer_to="date"
    )
    def dt_date(self):
        return DateTimeDefault.register(pandas.Series.dt.date)(self)

    @doc_utils.doc_dt_timestamp(prop="day component", refer_to="day")
    def dt_day(self):
        return DateTimeDefault.register(pandas.Series.dt.day)(self)

    @doc_utils.doc_dt_timestamp(
        prop="day name", refer_to="day_name", params="locale : str, optional"
    )
    def dt_day_name(self, locale=None):
        return DateTimeDefault.register(pandas.Series.dt.day_name)(self, locale)

    @doc_utils.doc_dt_timestamp(prop="integer day of week", refer_to="dayofweek")
    # FIXME: `dt_dayofweek` is an alias for `dt_weekday`, one of them should
    # be removed (Modin issue #3107).
    def dt_dayofweek(self):
        return DateTimeDefault.register(pandas.Series.dt.dayofweek)(self)

    @doc_utils.doc_dt_timestamp(prop="day of year", refer_to="dayofyear")
    def dt_dayofyear(self):
        return DateTimeDefault.register(pandas.Series.dt.dayofyear)(self)

    @doc_utils.doc_dt_interval(prop="days", refer_to="days")
    def dt_days(self):
        return DateTimeDefault.register(pandas.Series.dt.days)(self)

    @doc_utils.doc_dt_timestamp(
        prop="number of days in month", refer_to="days_in_month"
    )
    # FIXME: `dt_days_in_month` is an alias for `dt_daysinmonth`, one of them should
    # be removed (Modin issue #3107).
    def dt_days_in_month(self):
        return DateTimeDefault.register(pandas.Series.dt.days_in_month)(self)

    @doc_utils.doc_dt_timestamp(prop="number of days in month", refer_to="daysinmonth")
    def dt_daysinmonth(self):
        return DateTimeDefault.register(pandas.Series.dt.daysinmonth)(self)

    @doc_utils.doc_dt_period(prop="the timestamp of end time", refer_to="end_time")
    def dt_end_time(self):
        return DateTimeDefault.register(pandas.Series.dt.end_time)(self)

    @doc_utils.doc_dt_round(refer_to="floor")
    def dt_floor(self, freq, ambiguous="raise", nonexistent="raise"):
        return DateTimeDefault.register(pandas.Series.dt.floor)(
            self, freq, ambiguous, nonexistent
        )

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.freq")
    def dt_freq(self):
        """
        Get the time frequency of the underlying time-series data.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing a single value, the frequency of the data.
        """
        return DateTimeDefault.register(pandas.Series.dt.freq)(self)

    @doc_utils.doc_dt_timestamp(prop="hour", refer_to="hour")
    def dt_hour(self):
        return DateTimeDefault.register(pandas.Series.dt.hour)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether corresponding year is leap",
        refer_to="is_leap_year",
    )
    def dt_is_leap_year(self):
        return DateTimeDefault.register(pandas.Series.dt.is_leap_year)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the last day of the month",
        refer_to="is_month_end",
    )
    def dt_is_month_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_month_end)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the first day of the month",
        refer_to="is_month_start",
    )
    def dt_is_month_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_month_start)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the last day of the quarter",
        refer_to="is_quarter_end",
    )
    def dt_is_quarter_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_quarter_end)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the first day of the quarter",
        refer_to="is_quarter_start",
    )
    def dt_is_quarter_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_quarter_start)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the last day of the year",
        refer_to="is_year_end",
    )
    def dt_is_year_end(self):
        return DateTimeDefault.register(pandas.Series.dt.is_year_end)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the boolean of whether the date is the first day of the year",
        refer_to="is_year_start",
    )
    def dt_is_year_start(self):
        return DateTimeDefault.register(pandas.Series.dt.is_year_start)(self)

    @doc_utils.doc_dt_timestamp(prop="microseconds component", refer_to="microsecond")
    def dt_microsecond(self):
        return DateTimeDefault.register(pandas.Series.dt.microsecond)(self)

    @doc_utils.doc_dt_interval(prop="microseconds component", refer_to="microseconds")
    def dt_microseconds(self):
        return DateTimeDefault.register(pandas.Series.dt.microseconds)(self)

    @doc_utils.doc_dt_timestamp(prop="minute component", refer_to="minute")
    def dt_minute(self):
        return DateTimeDefault.register(pandas.Series.dt.minute)(self)

    @doc_utils.doc_dt_timestamp(prop="month component", refer_to="month")
    def dt_month(self):
        return DateTimeDefault.register(pandas.Series.dt.month)(self)

    @doc_utils.doc_dt_timestamp(
        prop="the month name", refer_to="month name", params="locale : str, optional"
    )
    def dt_month_name(self, locale=None):
        return DateTimeDefault.register(pandas.Series.dt.month_name)(self, locale)

    @doc_utils.doc_dt_timestamp(prop="nanoseconds component", refer_to="nanosecond")
    def dt_nanosecond(self):
        return DateTimeDefault.register(pandas.Series.dt.nanosecond)(self)

    @doc_utils.doc_dt_interval(prop="nanoseconds component", refer_to="nanoseconds")
    def dt_nanoseconds(self):
        return DateTimeDefault.register(pandas.Series.dt.nanoseconds)(self)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.normalize")
    def dt_normalize(self):
        """
        Set the time component of each date-time value to midnight.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing date-time values with midnight time.
        """
        return DateTimeDefault.register(pandas.Series.dt.normalize)(self)

    @doc_utils.doc_dt_timestamp(prop="quarter component", refer_to="quarter")
    def dt_quarter(self):
        return DateTimeDefault.register(pandas.Series.dt.quarter)(self)

    @doc_utils.doc_dt_period(prop="the fiscal year", refer_to="qyear")
    def dt_qyear(self):
        return DateTimeDefault.register(pandas.Series.dt.qyear)(self)

    @doc_utils.doc_dt_round(refer_to="round")
    def dt_round(self, freq, ambiguous="raise", nonexistent="raise"):
        return DateTimeDefault.register(pandas.Series.dt.round)(
            self, freq, ambiguous, nonexistent
        )

    @doc_utils.doc_dt_timestamp(prop="seconds component", refer_to="second")
    def dt_second(self):
        return DateTimeDefault.register(pandas.Series.dt.second)(self)

    @doc_utils.doc_dt_interval(prop="seconds component", refer_to="seconds")
    def dt_seconds(self):
        return DateTimeDefault.register(pandas.Series.dt.seconds)(self)

    @doc_utils.doc_dt_period(prop="the timestamp of start time", refer_to="start_time")
    def dt_start_time(self):
        return DateTimeDefault.register(pandas.Series.dt.start_time)(self)

    @doc_utils.add_refer_to("Series.dt.strftime")
    def dt_strftime(self, date_format):
        """
        Format underlying date-time data using specified format.

        Parameters
        ----------
        date_format : str

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing formatted date-time values.
        """
        return DateTimeDefault.register(pandas.Series.dt.strftime)(self, date_format)

    @doc_utils.doc_dt_timestamp(prop="time component", refer_to="time")
    def dt_time(self):
        return DateTimeDefault.register(pandas.Series.dt.time)(self)

    @doc_utils.doc_dt_timestamp(
        prop="time component with timezone information", refer_to="timetz"
    )
    def dt_timetz(self):
        return DateTimeDefault.register(pandas.Series.dt.timetz)(self)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.to_period")
    def dt_to_period(self, freq=None):
        """
        Convert underlying data to the period at a particular frequency.

        Parameters
        ----------
        freq : str, optional

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing period data.
        """
        return DateTimeDefault.register(pandas.Series.dt.to_period)(self, freq)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.to_pydatetime")
    def dt_to_pydatetime(self):
        """
        Convert underlying data to array of python native ``datetime``.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing 1D array of ``datetime`` objects.
        """
        return DateTimeDefault.register(pandas.Series.dt.to_pydatetime)(self)

    # FIXME: there are no references to this method, we should either remove it
    # or add a call reference at the DataFrame level (Modin issue #3103).
    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.to_pytimedelta")
    def dt_to_pytimedelta(self):
        """
        Convert underlying data to array of python native ``datetime.timedelta``.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing 1D array of ``datetime.timedelta``.
        """
        return DateTimeDefault.register(pandas.Series.dt.to_pytimedelta)(self)

    @doc_utils.doc_dt_period(
        prop="the timestamp representation", refer_to="to_timestamp"
    )
    def dt_to_timestamp(self):
        return DateTimeDefault.register(pandas.Series.dt.to_timestamp)(self)

    @doc_utils.doc_dt_interval(prop="duration in seconds", refer_to="total_seconds")
    def dt_total_seconds(self):
        return DateTimeDefault.register(pandas.Series.dt.total_seconds)(self)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.tz")
    def dt_tz(self):
        """
        Get the time-zone of the underlying time-series data.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing a single value, time-zone of the data.
        """
        return DateTimeDefault.register(pandas.Series.dt.tz)(self)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.tz_convert")
    def dt_tz_convert(self, tz):
        """
        Convert time-series data to the specified time zone.

        Parameters
        ----------
        tz : str, pytz.timezone

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing values with converted time zone.
        """
        return DateTimeDefault.register(pandas.Series.dt.tz_convert)(self, tz)

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.dt.tz_localize")
    def dt_tz_localize(self, tz, ambiguous="raise", nonexistent="raise"):
        """
        Localize tz-naive to tz-aware.

        Parameters
        ----------
        tz : str, pytz.timezone, optional
        ambiguous : {"raise", "inner", "NaT"} or bool mask, default: "raise"
        nonexistent : {"raise", "shift_forward", "shift_backward, "NaT"} or pandas.timedelta, default: "raise"

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing values with localized time zone.
        """
        return DateTimeDefault.register(pandas.Series.dt.tz_localize)(
            self, tz, ambiguous, nonexistent
        )

    @doc_utils.doc_dt_timestamp(prop="week component", refer_to="week")
    def dt_week(self):
        return DateTimeDefault.register(pandas.Series.dt.week)(self)

    @doc_utils.doc_dt_timestamp(prop="integer day of week", refer_to="weekday")
    def dt_weekday(self):
        return DateTimeDefault.register(pandas.Series.dt.weekday)(self)

    @doc_utils.doc_dt_timestamp(prop="week of year", refer_to="weekofyear")
    def dt_weekofyear(self):
        return DateTimeDefault.register(pandas.Series.dt.weekofyear)(self)

    @doc_utils.doc_dt_timestamp(prop="year component", refer_to="year")
    def dt_year(self):
        return DateTimeDefault.register(pandas.Series.dt.year)(self)

    # End of DateTime methods

    # Resample methods

    # FIXME:
    #   1. Query Compiler shouldn't care about differences between Series and DataFrame
    #      so `resample_agg_df` and `resample_agg_ser` should be combined (Modin issue #3104).
    #   2. In DataFrame API `Resampler.aggregate` is an alias for `Resampler.apply`
    #      we should remove one of these methods: `resample_agg_*` or `resample_app_*` (Modin issue #3107).
    @doc_utils.doc_resample_agg(
        action="apply passed aggregation function",
        params="func : str, dict, callable(pandas.Series) -> scalar, or list of such",
        output="function names",
        refer_to="agg",
    )
    def resample_agg_df(self, resample_kwargs, func, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.aggregate)(
            self, resample_kwargs, func, *args, **kwargs
        )

    @doc_utils.add_deprecation_warning(replacement_method="resample_agg_df")
    @doc_utils.doc_resample_agg(
        action="apply passed aggregation function in a one-column query compiler",
        params="func : str, dict, callable(pandas.Series) -> scalar, or list of such",
        output="function names",
        refer_to="agg",
    )
    def resample_agg_ser(self, resample_kwargs, func, *args, **kwargs):
        return ResampleDefault.register(
            pandas.core.resample.Resampler.aggregate, squeeze_self=True
        )(self, resample_kwargs, func, *args, **kwargs)

    @doc_utils.add_deprecation_warning(replacement_method="resample_agg_df")
    @doc_utils.doc_resample_agg(
        action="apply passed aggregation function",
        params="func : str, dict, callable(pandas.Series) -> scalar, or list of such",
        output="function names",
        refer_to="apply",
    )
    def resample_app_df(self, resample_kwargs, func, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.apply)(
            self, resample_kwargs, func, *args, **kwargs
        )

    @doc_utils.add_deprecation_warning(replacement_method="resample_agg_df")
    @doc_utils.doc_resample_agg(
        action="apply passed aggregation function in a one-column query compiler",
        params="func : str, dict, callable(pandas.Series) -> scalar, or list of such",
        output="function names",
        refer_to="apply",
    )
    def resample_app_ser(self, resample_kwargs, func, *args, **kwargs):
        return ResampleDefault.register(
            pandas.core.resample.Resampler.apply, squeeze_self=True
        )(self, resample_kwargs, func, *args, **kwargs)

    def resample_asfreq(self, resample_kwargs, fill_value):
        """
        Resample time-series data and get the values at the new frequency.

        Group data into intervals by time-series row/column with
        a specified frequency and get values at the new frequency.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        fill_value : scalar

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing values at the specified frequency.
        """
        return ResampleDefault.register(pandas.core.resample.Resampler.asfreq)(
            self, resample_kwargs, fill_value
        )

    # FIXME: `resample_backfill` is an alias for `resample_bfill`, one of these method
    # should be removed (Modin issue #3107).
    @doc_utils.doc_resample_fillna(method="back-fill", refer_to="backfill")
    def resample_backfill(self, resample_kwargs, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.backfill)(
            self, resample_kwargs, limit
        )

    @doc_utils.doc_resample_fillna(method="back-fill", refer_to="bfill")
    def resample_bfill(self, resample_kwargs, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.bfill)(
            self, resample_kwargs, limit
        )

    @doc_utils.doc_resample_reduce(
        result="number of non-NA values", refer_to="count", compatibility_params=False
    )
    def resample_count(self, resample_kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.count)(
            self, resample_kwargs
        )

    # FIXME: `resample_ffill` is an alias for `resample_pad`, one of these method
    # should be removed (Modin issue #3107).
    @doc_utils.doc_resample_fillna(method="forward-fill", refer_to="ffill")
    def resample_ffill(self, resample_kwargs, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.ffill)(
            self, resample_kwargs, limit
        )

    # FIXME: we should combine all resample fillna methods into `resample_fillna`
    # (Modin issue #3107)
    @doc_utils.doc_resample_fillna(
        method="specified", refer_to="fillna", params="method : str"
    )
    def resample_fillna(self, resample_kwargs, method, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.fillna)(
            self, resample_kwargs, method, limit
        )

    @doc_utils.doc_resample_reduce(
        result="first element", refer_to="first", params="_method : str"
    )
    def resample_first(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.first)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    # FIXME: This function takes Modin DataFrame via `obj` parameter,
    # we should avoid leaking of the high-level objects to the query compiler level.
    # (Modin issue #3106)
    def resample_get_group(self, resample_kwargs, name, obj):
        """
        Resample time-series data and get the specified group.

        Group data into intervals by time-series row/column with
        a specified frequency and get the values of the specified group.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        name : object
        obj : modin.pandas.DataFrame, optional

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the values from the specified group.
        """
        return ResampleDefault.register(pandas.core.resample.Resampler.get_group)(
            self, resample_kwargs, name, obj
        )

    @doc_utils.doc_resample_fillna(
        method="specified interpolation",
        refer_to="interpolate",
        params="""
        method : str
        axis : {0, 1}
        limit : int
        inplace : {False}
            This parameter serves the compatibility purpose. Always has to be False.
        limit_direction : {"forward", "backward", "both"}
        limit_area : {None, "inside", "outside"}
        downcast : str, optional
        **kwargs : dict
        """,
        overwrite_template_params=True,
    )
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
        return ResampleDefault.register(pandas.core.resample.Resampler.interpolate)(
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
        )

    @doc_utils.doc_resample_reduce(
        result="last element", params="_method : str", refer_to="last"
    )
    def resample_last(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.last)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="maximum value", params="_method : str", refer_to="max"
    )
    def resample_max(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.max)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="mean value", params="_method : str", refer_to="mean"
    )
    def resample_mean(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.mean)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="median value", params="_method : str", refer_to="median"
    )
    def resample_median(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.median)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="minimum value", params="_method : str", refer_to="min"
    )
    def resample_min(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.min)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_fillna(method="'nearest'", refer_to="nearest")
    def resample_nearest(self, resample_kwargs, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.nearest)(
            self, resample_kwargs, limit
        )

    @doc_utils.doc_resample_reduce(
        result="number of unique values", params="_method : str", refer_to="nunique"
    )
    def resample_nunique(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.nunique)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    # FIXME: Query Compiler shouldn't care about differences between Series and DataFrame
    # so `resample_ohlc_df` and `resample_ohlc_ser` should be combined (Modin issue #3104).
    @doc_utils.doc_resample_agg(
        action="compute open, high, low and close values",
        params="_method : str",
        output="labels of columns containing computed values",
        refer_to="ohlc",
    )
    def resample_ohlc_df(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.ohlc)(
            self, resample_kwargs, _method, *args, **kwargs
        )

    @doc_utils.doc_resample_agg(
        action="compute open, high, low and close values",
        params="_method : str",
        output="labels of columns containing computed values",
        refer_to="ohlc",
    )
    def resample_ohlc_ser(self, resample_kwargs, _method, *args, **kwargs):
        return ResampleDefault.register(
            pandas.core.resample.Resampler.ohlc, squeeze_self=True
        )(self, resample_kwargs, _method, *args, **kwargs)

    @doc_utils.doc_resample_fillna(method="'pad'", refer_to="pad")
    def resample_pad(self, resample_kwargs, limit):
        return ResampleDefault.register(pandas.core.resample.Resampler.pad)(
            self, resample_kwargs, limit
        )

    # FIXME: This method require us to build high-level resampler object
    # which we shouldn't do at the query compiler. We need to move this at the front.
    # (Modin issue #3105)
    @doc_utils.add_refer_to("Resampler.pipe")
    def resample_pipe(self, resample_kwargs, func, *args, **kwargs):
        """
        Resample time-series data and apply aggregation on it.

        Group data into intervals by time-series row/column with
        a specified frequency, build equivalent ``pandas.Resampler`` object
        and apply passed function to it.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        func : callable(pandas.Resampler) -> object or tuple(callable, str)
        *args : iterable
            Positional arguments to pass to function.
        **kwargs : dict
            Keyword arguments to pass to function.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the result of passed function.
        """
        return ResampleDefault.register(pandas.core.resample.Resampler.pipe)(
            self, resample_kwargs, func, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="product",
        params="""
        _method : str
        min_count : int""",
        refer_to="prod",
    )
    def resample_prod(self, resample_kwargs, _method, min_count, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.prod)(
            self, resample_kwargs, _method, min_count, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="quantile", params="q : float", refer_to="quantile"
    )
    def resample_quantile(self, resample_kwargs, q, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.quantile)(
            self, resample_kwargs, q, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="standard error of the mean",
        params="ddof : int, default: 1",
        refer_to="sem",
    )
    def resample_sem(self, resample_kwargs, ddof=1, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.sem)(
            self, resample_kwargs, ddof, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="number of elements in a group", refer_to="size"
    )
    def resample_size(self, resample_kwargs, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.size)(
            self, resample_kwargs, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="standard deviation", params="ddof : int", refer_to="std"
    )
    def resample_std(self, resample_kwargs, ddof, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.std)(
            self, resample_kwargs, ddof, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="sum",
        params="""
        _method : str
        min_count : int""",
        refer_to="sum",
    )
    def resample_sum(self, resample_kwargs, _method, min_count, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.sum)(
            self, resample_kwargs, _method, min_count, *args, **kwargs
        )

    def resample_transform(self, resample_kwargs, arg, *args, **kwargs):
        """
        Resample time-series data and apply aggregation on it.

        Group data into intervals by time-series row/column with
        a specified frequency and call passed function on each group.
        In contrast to ``resample_app_df`` apply function to the whole group,
        instead of a single axis.

        Parameters
        ----------
        resample_kwargs : dict
            Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.
        arg : callable(pandas.DataFrame) -> pandas.Series
        *args : iterable
            Positional arguments to pass to function.
        **kwargs : dict
            Keyword arguments to pass to function.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the result of passed function.
        """
        return ResampleDefault.register(pandas.core.resample.Resampler.transform)(
            self, resample_kwargs, arg, *args, **kwargs
        )

    @doc_utils.doc_resample_reduce(
        result="variance", params="ddof : int", refer_to="var"
    )
    def resample_var(self, resample_kwargs, ddof, *args, **kwargs):
        return ResampleDefault.register(pandas.core.resample.Resampler.var)(
            self, resample_kwargs, ddof, *args, **kwargs
        )

    # End of Resample methods

    # Str methods

    @doc_utils.doc_str_method(refer_to="capitalize", params="")
    def str_capitalize(self):
        return StrDefault.register(pandas.Series.str.capitalize)(self)

    @doc_utils.doc_str_method(
        refer_to="center",
        params="""
        width : int
        fillchar : str, default: ' '""",
    )
    def str_center(self, width, fillchar=" "):
        return StrDefault.register(pandas.Series.str.center)(self, width, fillchar)

    @doc_utils.doc_str_method(
        refer_to="contains",
        params="""
        pat : str
        case : bool, default: True
        flags : int, default: 0
        na : object, default: np.NaN
        regex : bool, default: True""",
    )
    def str_contains(self, pat, case=True, flags=0, na=np.NaN, regex=True):
        return StrDefault.register(pandas.Series.str.contains)(
            self, pat, case, flags, na, regex
        )

    @doc_utils.doc_str_method(
        refer_to="count",
        params="""
        pat : str
        flags : int, default: 0
        **kwargs : dict""",
    )
    def str_count(self, pat, flags=0, **kwargs):
        return StrDefault.register(pandas.Series.str.count)(self, pat, flags, **kwargs)

    @doc_utils.doc_str_method(
        refer_to="endswith",
        params="""
        pat : str
        na : object, default: np.NaN""",
    )
    def str_endswith(self, pat, na=np.NaN):
        return StrDefault.register(pandas.Series.str.endswith)(self, pat, na)

    @doc_utils.doc_str_method(
        refer_to="find",
        params="""
        sub : str
        start : int, default: 0
        end : int, optional""",
    )
    def str_find(self, sub, start=0, end=None):
        return StrDefault.register(pandas.Series.str.find)(self, sub, start, end)

    @doc_utils.doc_str_method(
        refer_to="findall",
        params="""
        pat : str
        flags : int, default: 0
        **kwargs : dict""",
    )
    def str_findall(self, pat, flags=0, **kwargs):
        return StrDefault.register(pandas.Series.str.findall)(
            self, pat, flags, **kwargs
        )

    @doc_utils.doc_str_method(refer_to="get", params="i : int")
    def str_get(self, i):
        return StrDefault.register(pandas.Series.str.get)(self, i)

    @doc_utils.doc_str_method(
        refer_to="index",
        params="""
        sub : str
        start : int, default: 0
        end : int, optional""",
    )
    def str_index(self, sub, start=0, end=None):
        return StrDefault.register(pandas.Series.str.index)(self, sub, start, end)

    @doc_utils.doc_str_method(refer_to="isalnum", params="")
    def str_isalnum(self):
        return StrDefault.register(pandas.Series.str.isalnum)(self)

    @doc_utils.doc_str_method(refer_to="isalpha", params="")
    def str_isalpha(self):
        return StrDefault.register(pandas.Series.str.isalpha)(self)

    @doc_utils.doc_str_method(refer_to="isdecimal", params="")
    def str_isdecimal(self):
        return StrDefault.register(pandas.Series.str.isdecimal)(self)

    @doc_utils.doc_str_method(refer_to="isdigit", params="")
    def str_isdigit(self):
        return StrDefault.register(pandas.Series.str.isdigit)(self)

    @doc_utils.doc_str_method(refer_to="islower", params="")
    def str_islower(self):
        return StrDefault.register(pandas.Series.str.islower)(self)

    @doc_utils.doc_str_method(refer_to="isnumeric", params="")
    def str_isnumeric(self):
        return StrDefault.register(pandas.Series.str.isnumeric)(self)

    @doc_utils.doc_str_method(refer_to="isspace", params="")
    def str_isspace(self):
        return StrDefault.register(pandas.Series.str.isspace)(self)

    @doc_utils.doc_str_method(refer_to="istitle", params="")
    def str_istitle(self):
        return StrDefault.register(pandas.Series.str.istitle)(self)

    @doc_utils.doc_str_method(refer_to="isupper", params="")
    def str_isupper(self):
        return StrDefault.register(pandas.Series.str.isupper)(self)

    @doc_utils.doc_str_method(refer_to="join", params="sep : str")
    def str_join(self, sep):
        return StrDefault.register(pandas.Series.str.join)(self, sep)

    @doc_utils.doc_str_method(refer_to="len", params="")
    def str_len(self):
        return StrDefault.register(pandas.Series.str.len)(self)

    @doc_utils.doc_str_method(
        refer_to="ljust",
        params="""
        width : int
        fillchar : str, default: ' '""",
    )
    def str_ljust(self, width, fillchar=" "):
        return StrDefault.register(pandas.Series.str.ljust)(self, width, fillchar)

    @doc_utils.doc_str_method(refer_to="lower", params="")
    def str_lower(self):
        return StrDefault.register(pandas.Series.str.lower)(self)

    @doc_utils.doc_str_method(refer_to="lstrip", params="to_strip : str, optional")
    def str_lstrip(self, to_strip=None):
        return StrDefault.register(pandas.Series.str.lstrip)(self, to_strip)

    @doc_utils.doc_str_method(
        refer_to="match",
        params="""
        pat : str
        case : bool, default: True
        flags : int, default: 0
        na : object, default: np.NaN""",
    )
    def str_match(self, pat, case=True, flags=0, na=np.NaN):
        return StrDefault.register(pandas.Series.str.match)(self, pat, case, flags, na)

    @doc_utils.doc_str_method(
        refer_to="normalize", params="form : {'NFC', 'NFKC', 'NFD', 'NFKD'}"
    )
    def str_normalize(self, form):
        return StrDefault.register(pandas.Series.str.normalize)(self, form)

    @doc_utils.doc_str_method(
        refer_to="pad",
        params="""
        width : int
        side : {'left', 'right', 'both'}, default: 'left'
        fillchar : str, default: ' '""",
    )
    def str_pad(self, width, side="left", fillchar=" "):
        return StrDefault.register(pandas.Series.str.pad)(self, width, side, fillchar)

    @doc_utils.doc_str_method(
        refer_to="partition",
        params="""
        sep : str, default: ' '
        expand : bool, default: True""",
    )
    def str_partition(self, sep=" ", expand=True):
        return StrDefault.register(pandas.Series.str.partition)(self, sep, expand)

    @doc_utils.doc_str_method(refer_to="repeat", params="repeats : int")
    def str_repeat(self, repeats):
        return StrDefault.register(pandas.Series.str.repeat)(self, repeats)

    @doc_utils.doc_str_method(
        refer_to="replace",
        params="""
        pat : str
        repl : str or callable
        n : int, default: -1
        case : bool, optional
        flags : int, default: 0
        regex : bool, default: True""",
    )
    def str_replace(self, pat, repl, n=-1, case=None, flags=0, regex=True):
        return StrDefault.register(pandas.Series.str.replace)(
            self, pat, repl, n, case, flags, regex
        )

    @doc_utils.doc_str_method(
        refer_to="rfind",
        params="""
        sub : str
        start : int, default: 0
        end : int, optional""",
    )
    def str_rfind(self, sub, start=0, end=None):
        return StrDefault.register(pandas.Series.str.rfind)(self, sub, start, end)

    @doc_utils.doc_str_method(
        refer_to="rindex",
        params="""
        sub : str
        start : int, default: 0
        end : int, optional""",
    )
    def str_rindex(self, sub, start=0, end=None):
        return StrDefault.register(pandas.Series.str.rindex)(self, sub, start, end)

    @doc_utils.doc_str_method(
        refer_to="rjust",
        params="""
        width : int
        fillchar : str, default: ' '""",
    )
    def str_rjust(self, width, fillchar=" "):
        return StrDefault.register(pandas.Series.str.rjust)(self, width, fillchar)

    @doc_utils.doc_str_method(
        refer_to="rpartition",
        params="""
        sep : str, default: ' '
        expand : bool, default: True""",
    )
    def str_rpartition(self, sep=" ", expand=True):
        return StrDefault.register(pandas.Series.str.rpartition)(self, sep, expand)

    @doc_utils.doc_str_method(
        refer_to="rsplit",
        params="""
        pat : str, optional
        n : int, default: -1
        expand : bool, default: False""",
    )
    def str_rsplit(self, pat=None, n=-1, expand=False):
        return StrDefault.register(pandas.Series.str.rsplit)(self, pat, n, expand)

    @doc_utils.doc_str_method(refer_to="rstrip", params="to_strip : str, optional")
    def str_rstrip(self, to_strip=None):
        return StrDefault.register(pandas.Series.str.rstrip)(self, to_strip)

    @doc_utils.doc_str_method(
        refer_to="slice",
        params="""
        start : int, optional
        stop : int, optional
        step : int, optional""",
    )
    def str_slice(self, start=None, stop=None, step=None):
        return StrDefault.register(pandas.Series.str.slice)(self, start, stop, step)

    @doc_utils.doc_str_method(
        refer_to="slice_replace",
        params="""
        start : int, optional
        stop : int, optional
        repl : str or callable, optional""",
    )
    def str_slice_replace(self, start=None, stop=None, repl=None):
        return StrDefault.register(pandas.Series.str.slice_replace)(
            self, start, stop, repl
        )

    @doc_utils.doc_str_method(
        refer_to="split",
        params="""
        pat : str, optional
        n : int, default: -1
        expand : bool, default: False""",
    )
    def str_split(self, pat=None, n=-1, expand=False):
        return StrDefault.register(pandas.Series.str.split)(self, pat, n, expand)

    @doc_utils.doc_str_method(
        refer_to="startswith",
        params="""
        pat : str
        na : object, default: np.NaN""",
    )
    def str_startswith(self, pat, na=np.NaN):
        return StrDefault.register(pandas.Series.str.startswith)(self, pat, na)

    @doc_utils.doc_str_method(refer_to="strip", params="to_strip : str, optional")
    def str_strip(self, to_strip=None):
        return StrDefault.register(pandas.Series.str.strip)(self, to_strip)

    @doc_utils.doc_str_method(refer_to="swapcase", params="")
    def str_swapcase(self):
        return StrDefault.register(pandas.Series.str.swapcase)(self)

    @doc_utils.doc_str_method(refer_to="title", params="")
    def str_title(self):
        return StrDefault.register(pandas.Series.str.title)(self)

    @doc_utils.doc_str_method(refer_to="translate", params="table : dict")
    def str_translate(self, table):
        return StrDefault.register(pandas.Series.str.translate)(self, table)

    @doc_utils.doc_str_method(refer_to="upper", params="")
    def str_upper(self):
        return StrDefault.register(pandas.Series.str.upper)(self)

    @doc_utils.doc_str_method(
        refer_to="wrap",
        params="""
        width : int
        **kwargs : dict""",
    )
    def str_wrap(self, width, **kwargs):
        return StrDefault.register(pandas.Series.str.wrap)(self, width, **kwargs)

    @doc_utils.doc_str_method(refer_to="zfill", params="width : int")
    def str_zfill(self, width):
        return StrDefault.register(pandas.Series.str.zfill)(self, width)

    @doc_utils.doc_str_method(refer_to="__getitem__", params="key : object")
    def str___getitem__(self, key):
        return StrDefault.register(pandas.Series.str.__getitem__)(self, key)

    # End of Str methods

    # Rolling methods

    # FIXME: most of the rolling/window methods take *args and **kwargs parameters
    # which are only needed for the compatibility with numpy, this behavior is inherited
    # from the API level, we should get rid of it (Modin issue #3108).

    @doc_utils.doc_window_method(
        result="the result of passed functions",
        action="apply specified functions",
        refer_to="aggregate",
        params="""
        func : str, dict, callable(pandas.Series) -> scalar, or list of such
        *args : iterable
        **kwargs : dict""",
        build_rules="udf_aggregation",
    )
    def rolling_aggregate(self, fold_axis, rolling_args, func, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.aggregate)(
            self, rolling_args, func, *args, **kwargs
        )

    # FIXME: at the query compiler method `rolling_apply` is an alias for `rolling_aggregate`,
    # one of these should be removed (Modin issue #3107).
    @doc_utils.add_deprecation_warning(replacement_method="rolling_aggregate")
    @doc_utils.doc_window_method(
        result="the result of passed function",
        action="apply specified function",
        refer_to="apply",
        params="""
        func : callable(pandas.Series) -> scalar
        raw : bool, default: False
        engine : None, default: None
            This parameters serves the compatibility purpose. Always has to be None.
        engine_kwargs : None, default: None
            This parameters serves the compatibility purpose. Always has to be None.
        args : tuple, optional
        kwargs : dict, optional""",
        build_rules="udf_aggregation",
    )
    def rolling_apply(
        self,
        fold_axis,
        rolling_args,
        func,
        raw=False,
        engine=None,
        engine_kwargs=None,
        args=None,
        kwargs=None,
    ):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.apply)(
            self, rolling_args, func, raw, engine, engine_kwargs, args, kwargs
        )

    @doc_utils.doc_window_method(
        result="correlation",
        refer_to="corr",
        params="""
        other : modin.pandas.Series, modin.pandas.DataFrame, list-like, optional
        pairwise : bool, optional
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_corr(
        self, fold_axis, rolling_args, other=None, pairwise=None, *args, **kwargs
    ):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.corr)(
            self, rolling_args, other, pairwise, *args, **kwargs
        )

    @doc_utils.doc_window_method(result="number of non-NA values", refer_to="count")
    def rolling_count(self, fold_axis, rolling_args):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.count)(
            self, rolling_args
        )

    @doc_utils.doc_window_method(
        result="covariance",
        refer_to="cov",
        params="""
        other : modin.pandas.Series, modin.pandas.DataFrame, list-like, optional
        pairwise : bool, optional
        ddof : int, default:  1
        **kwargs : dict""",
    )
    def rolling_cov(
        self, fold_axis, rolling_args, other=None, pairwise=None, ddof=1, **kwargs
    ):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.cov)(
            self, rolling_args, other, pairwise, ddof, **kwargs
        )

    @doc_utils.doc_window_method(
        result="unbiased kurtosis", refer_to="kurt", params="**kwargs : dict"
    )
    def rolling_kurt(self, fold_axis, rolling_args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.kurt)(
            self, rolling_args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="maximum value",
        refer_to="max",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_max(self, fold_axis, rolling_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.max)(
            self, rolling_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="mean value",
        refer_to="mean",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_mean(self, fold_axis, rolling_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.mean)(
            self, rolling_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="median value", refer_to="median", params="**kwargs : dict"
    )
    def rolling_median(self, fold_axis, rolling_args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.median)(
            self, rolling_args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="minimum value",
        refer_to="min",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_min(self, fold_axis, rolling_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.min)(
            self, rolling_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="quantile",
        refer_to="quantile",
        params="""
        quantile : float
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default: 'linear'
        **kwargs : dict""",
    )
    def rolling_quantile(
        self, fold_axis, rolling_args, quantile, interpolation="linear", **kwargs
    ):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.quantile)(
            self, rolling_args, quantile, interpolation, **kwargs
        )

    @doc_utils.doc_window_method(
        result="unbiased skewness", refer_to="skew", params="**kwargs : dict"
    )
    def rolling_skew(self, fold_axis, rolling_args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.skew)(
            self, rolling_args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="standard deviation",
        refer_to="std",
        params="""
        ddof : int, default: 1
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_std(self, fold_axis, rolling_args, ddof=1, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.std)(
            self, rolling_args, ddof, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="sum",
        refer_to="sum",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_sum(self, fold_axis, rolling_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.sum)(
            self, rolling_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        result="variance",
        refer_to="var",
        params="""
        ddof : int, default: 1
        *args : iterable
        **kwargs : dict""",
    )
    def rolling_var(self, fold_axis, rolling_args, ddof=1, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.rolling.Rolling.var)(
            self, rolling_args, ddof, *args, **kwargs
        )

    # End of Rolling methods

    # Window methods

    @doc_utils.doc_window_method(
        win_type="window of the specified type",
        result="mean",
        refer_to="mean",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def window_mean(self, fold_axis, window_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.Window.mean)(
            self, window_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        win_type="window of the specified type",
        result="standard deviation",
        refer_to="std",
        params="""
        ddof : int, default: 1
        *args : iterable
        **kwargs : dict""",
    )
    def window_std(self, fold_axis, window_args, ddof=1, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.Window.std)(
            self, window_args, ddof, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        win_type="window of the specified type",
        result="sum",
        refer_to="sum",
        params="""
        *args : iterable
        **kwargs : dict""",
    )
    def window_sum(self, fold_axis, window_args, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.Window.sum)(
            self, window_args, *args, **kwargs
        )

    @doc_utils.doc_window_method(
        win_type="window of the specified type",
        result="variance",
        refer_to="var",
        params="""
        ddof : int, default: 1
        *args : iterable
        **kwargs : dict""",
    )
    def window_var(self, fold_axis, window_args, ddof=1, *args, **kwargs):
        return RollingDefault.register(pandas.core.window.Window.var)(
            self, window_args, ddof, *args, **kwargs
        )

    # End of Window methods

    # Categories methods

    @doc_utils.add_one_column_warning
    @doc_utils.add_refer_to("Series.cat.codes")
    def cat_codes(self):
        """
        Convert underlying categories data into its codes.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the integer codes of the underlying
            categories.
        """
        return CatDefault.register(pandas.Series.cat.codes)(self)

    # End of Categories methods

    # DataFrame methods

    def invert(self):
        """
        Apply bitwise inversion for each element of the QueryCompiler.

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing bitwise inversion for each value.
        """
        return DataFrameDefault.register(pandas.DataFrame.__invert__)(self)

    @doc_utils.doc_reduce_agg(
        method="mean absolute deviation",
        params="""
        axis : {0, 1}
        skipna : bool
        level : None, default: None
            Serves the compatibility purpose. Always has to be None.""",
        refer_to="mad",
    )
    def mad(self, axis, skipna, level=None):
        return DataFrameDefault.register(pandas.DataFrame.mad)(
            self, axis=axis, skipna=skipna, level=level
        )

    @doc_utils.doc_reduce_agg(
        method="unbiased kurtosis", refer_to="kurt", extra_params=["skipna", "**kwargs"]
    )
    def kurt(self, axis, level=None, numeric_only=None, skipna=True, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.kurt)(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    sum_min_count = sum
    prod_min_count = prod

    @doc_utils.add_refer_to("DataFrame.compare")
    def compare(self, other, align_axis, keep_shape, keep_equal):
        """
        Compare data of two QueryCompilers and highlight the difference.

        Parameters
        ----------
        other : BaseQueryCompiler
            Query compiler to compare with. Have to be the same shape and the same
            labeling as `self`.
        align_axis : {0, 1}
        keep_shape : bool
        keep_equal : bool

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler containing the differences between `self` and passed
            query compiler.
        """
        return DataFrameDefault.register(pandas.DataFrame.compare)(
            self,
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
        )

    # End of DataFrame methods
