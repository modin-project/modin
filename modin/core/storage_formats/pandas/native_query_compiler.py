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
Module contains ``NativeQueryCompiler`` class.

``NativeQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for small data and empty ``PandasDataFrame``.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas
from pandas.core.dtypes.common import is_scalar

from modin.config.envvars import NativePandasMaxRows, NativePandasTransferThreshold
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
)
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import _inherit_docstrings, try_cast_to_pandas

if TYPE_CHECKING:
    from modin.pandas import DataFrame, Series
    from modin.pandas.base import BasePandasDataset

_NO_REPARTITION_ON_NATIVE_EXECUTION_EXCEPTION_MESSAGE = (
    "Modin dataframes and series using native execution do not have partitions."
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
    callable(NativeQueryCompiler) -> pandas.Index
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
    callable(NativeQueryCompiler)
    """
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


@_inherit_docstrings(BaseQueryCompiler)
class NativeQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for executing operations with native pandas.

    Parameters
    ----------
    pandas_frame : pandas.DataFrame
        The pandas frame to query with the compiled queries.
    """

    _OPERATION_INITIALIZATION_OVERHEAD = 0
    _OPERATION_PER_ROW_OVERHEAD = 0

    _modin_frame: pandas.DataFrame
    _should_warn_on_default_to_pandas: bool = False

    def __init__(self, pandas_frame):
        if hasattr(pandas_frame, "_to_pandas"):
            pandas_frame = pandas_frame._to_pandas()
        if is_scalar(pandas_frame):
            pandas_frame = pandas.DataFrame([pandas_frame])
        elif isinstance(pandas_frame, pandas.DataFrame):
            # NOTE we have to make a deep copy of the input pandas dataframe
            # so that we don't modify it.
            # TODO(https://github.com/modin-project/modin/issues/7435): Look
            # into avoiding this copy.
            pandas_frame = pandas_frame.copy()
        else:
            pandas_frame = pandas.DataFrame(pandas_frame)

        self._modin_frame = pandas_frame

    storage_format = property(
        lambda self: "Native", doc=BaseQueryCompiler.storage_format.__doc__
    )
    engine = property(lambda self: "Native", doc=BaseQueryCompiler.engine.__doc__)

    def execute(self):
        pass

    @property
    def frame_has_materialized_dtypes(self) -> bool:
        """
        Check if the undelying dataframe has materialized dtypes.

        Returns
        -------
        bool
        """
        return True

    def set_frame_dtypes_cache(self, dtypes):
        """
        Set dtypes cache for the underlying dataframe frame.

        Parameters
        ----------
        dtypes : pandas.Series, ModinDtypes, callable or None

        Notes
        -----
        This function is for consistency with other QCs,
        dtypes should be assigned directly on the frame.
        """
        pass

    def set_frame_index_cache(self, index):
        """
        Set index cache for underlying dataframe.

        Parameters
        ----------
        index : sequence, callable or None

        Notes
        -----
        This function is for consistency with other QCs,
        index should be assigned directly on the frame.
        """
        pass

    @property
    def frame_has_index_cache(self):
        """
        Check if the index cache exists for underlying dataframe.

        Returns
        -------
        bool
        """
        return True

    @property
    def frame_has_dtypes_cache(self) -> bool:
        """
        Check if the dtypes cache exists for the underlying dataframe.

        Returns
        -------
        bool
        """
        return True

    def copy(self):
        return self.__constructor__(self._modin_frame)

    def to_pandas(self):
        # NOTE we have to make a deep copy of the input pandas dataframe
        # so that the user doesn't modify it.
        # TODO(https://github.com/modin-project/modin/issues/7435): Look
        # into avoiding this copy when we default to pandas to execute each
        # method.
        return self._modin_frame.copy()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(df)

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(at.to_pandas())

    def free(self):
        return

    def finalize(self):
        return

    @classmethod
    def _engine_max_size(cls):
        # do not return the custom configuration for sub-classes
        if cls == NativeQueryCompiler:
            return NativePandasMaxRows.get()
        return cls._MAX_SIZE_THIS_ENGINE_CAN_HANDLE

    @classmethod
    def _transfer_threshold(cls):
        # do not return the custom configuration for sub-classes
        if cls == NativeQueryCompiler:
            return NativePandasTransferThreshold.get()
        return cls._TRANSFER_THRESHOLD

    def do_array_ufunc_implementation(
        self,
        frame: "BasePandasDataset",
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any
    ) -> Union["DataFrame", "Series", Any]:
        assert (
            self is frame._query_compiler
        ), "array ufunc called with mismatched query compiler and input frame"
        pandas_frame = self._modin_frame
        if not frame._is_dataframe:
            pandas_frame = pandas_frame.iloc[:, 0]
        pandas_result = pandas_frame.__array_ufunc__(
            ufunc,
            method,
            *(
                pandas_frame if each_input is frame else try_cast_to_pandas(each_input)
                for each_input in inputs
            ),
            **try_cast_to_pandas(kwargs),
        )
        if isinstance(pandas_result, pandas.DataFrame):
            from modin.pandas import DataFrame

            return DataFrame(pandas_result)
        elif isinstance(pandas_result, pandas.Series):
            from modin.pandas import Series

            return Series(pandas_result)
        # ufuncs are required to be one-to-one mappings, so this branch should never be hit
        return pandas_result  # pragma: no cover

    # Dataframe interchange protocol
    def to_interchange_dataframe(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ):
        return self._modin_frame.__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_interchange_dataframe(cls, df: ProtocolDataframe, data_cls):
        return cls(pandas.api.interchange.from_dataframe(df))

    # END Dataframe interchange protocol

    def support_materialization_in_worker_process(self) -> bool:
        """
        Whether it's possible to call function `to_pandas` during the pickling process, at the moment of recreating the object.

        Returns
        -------
        bool
        """
        return False

    def get_pandas_backend(self) -> Optional[str]:
        """
        Get backend stored in `_modin_frame`.

        Returns
        -------
        str | None
            Backend name.
        """
        return None

    # NOTE that because this query compiler provides the index of its underlying
    # pandas dataframe, updating the index affects this frame, and vice versa.
    # Consequently, native execution does not suffer from the issue
    # https://github.com/modin-project/modin/issues/1618
    index: pandas.Index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @_inherit_docstrings(BaseQueryCompiler.repartition)
    def repartition(self, axis=None):
        raise Exception(_NO_REPARTITION_ON_NATIVE_EXECUTION_EXCEPTION_MESSAGE)
