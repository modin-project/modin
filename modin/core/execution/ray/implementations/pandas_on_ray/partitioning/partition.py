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

"""Module houses class that wraps data (block partition) and its metadata."""

from typing import Callable, Union

import pandas
import ray

from modin.config import LazyExecution, RayTaskCustomResources
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
    DeferredExecution,
    MetaList,
    MetaListHook,
)
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import disable_logging, get_logger
from modin.pandas.indexing import compute_sliced_len
from modin.utils import _inherit_docstrings


class PandasOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    data : ObjectIDType or DeferredExecution
        A reference to ``pandas.DataFrame`` that needs to be wrapped with this class
        or a reference to DeferredExecution that needs to be executed on demand.
    length : ObjectIDType or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : ObjectIDType or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    ip : ObjectIDType or str, optional
        Node IP address or reference to it that holds wrapped ``pandas.DataFrame``.
    meta : MetaList
        Meta information, containing the lengths and the worker address (the last value).
    meta_offset : int
        The lengths offset in the meta list.
    """

    execution_wrapper = RayWrapper

    def __init__(
        self,
        data: Union[ray.ObjectRef, DeferredExecution],
        length: int = None,
        width: int = None,
        ip: str = None,
        meta: MetaList = None,
        meta_offset: int = 0,
    ):
        super().__init__()
        if isinstance(data, DeferredExecution):
            data.subscribe()
        self._data_ref = data
        # The metadata is stored in the MetaList at 0 offset. If the data is
        # a DeferredExecution, the _meta will be replaced with the list, returned
        # by the remote function. The returned list may contain data for multiple
        # results and, in this case, _meta_offset corresponds to the meta related to
        # this partition.
        if meta is None:
            self._meta = MetaList([length, width, ip])
            self._meta_offset = 0
        else:
            self._meta = meta
            self._meta_offset = meta_offset

        log = get_logger()
        self._is_debug(log) and log.debug(
            "Partition ID: {}, Height: {}, Width: {}, Node IP: {}".format(
                self._identity,
                str(self._length_cache),
                str(self._width_cache),
                str(self._ip_cache),
            )
        )

    @disable_logging
    def __del__(self):
        """Unsubscribe from DeferredExecution."""
        if isinstance(self._data_ref, DeferredExecution):
            self._data_ref.unsubscribe()

    def apply(self, func: Union[Callable, ray.ObjectRef], *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.

        Notes
        -----
        It does not matter if `func` is callable or an ``ray.ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.apply::{self._identity}")
        de = DeferredExecution(self._data_ref, func, args, kwargs)
        data, meta, meta_offset = de.exec()
        self._is_debug(log) and log.debug(f"EXIT::Partition.apply::{self._identity}")
        return self.__constructor__(data, meta=meta, meta_offset=meta_offset)

    @_inherit_docstrings(PandasDataframePartition.add_to_apply_calls)
    def add_to_apply_calls(
        self,
        func: Union[Callable, ray.ObjectRef],
        *args,
        length=None,
        width=None,
        **kwargs,
    ):
        return self.__constructor__(
            data=DeferredExecution(self._data_ref, func, args, kwargs),
            length=length,
            width=width,
        )

    @_inherit_docstrings(PandasDataframePartition.drain_call_queue)
    def drain_call_queue(self):
        data = self._data_ref
        if not isinstance(data, DeferredExecution):
            return data

        log = get_logger()
        self._is_debug(log) and log.debug(
            f"ENTER::Partition.drain_call_queue::{self._identity}"
        )
        self._data_ref, self._meta, self._meta_offset = data.exec()
        self._is_debug(log) and log.debug(
            f"EXIT::Partition.drain_call_queue::{self._identity}"
        )

    @_inherit_docstrings(PandasDataframePartition.wait)
    def wait(self):
        self.drain_call_queue()
        RayWrapper.wait(self._data_ref)

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnRayDataframePartition
            A copy of this partition.
        """
        return self.__constructor__(
            self._data_ref,
            meta=self._meta,
            meta_offset=self._meta_offset,
        )

    def mask(self, row_labels, col_labels):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_labels : list-like, slice or label
            The row labels for the rows to extract.
        col_labels : list-like, slice or label
            The column labels for the columns to extract.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.mask::{self._identity}")
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and isinstance(
            (len_cache := self._length_cache), ObjectIDType
        ):
            if row_labels == slice(None):
                # fast path - full axis take
                new_obj._length_cache = len_cache
            else:
                new_obj._length_cache = SlicerHook(len_cache, row_labels)
        if isinstance(col_labels, slice) and isinstance(
            (width_cache := self._width_cache), ObjectIDType
        ):
            if col_labels == slice(None):
                # fast path - full axis take
                new_obj._width_cache = width_cache
            else:
                new_obj._width_cache = SlicerHook(width_cache, col_labels)
        self._is_debug(log) and log.debug(f"EXIT::Partition.mask::{self._identity}")
        return new_obj

    @classmethod
    def put(cls, obj: pandas.DataFrame):
        """
        Put the data frame into Plasma store and wrap it with partition object.

        Parameters
        ----------
        obj : pandas.DataFrame
            A data frame to be put.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        return cls(cls.execution_wrapper.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """
        Put a function into the Plasma store to use in ``apply``.

        Parameters
        ----------
        func : callable
            A function to preprocess.

        Returns
        -------
        ray.ObjectRef
            A reference to `func`.
        """
        return cls.execution_wrapper.put(func)

    def length(self, materialize=True):
        """
        Get the length of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The length of the object.
        """
        if (length := self._length_cache) is None:
            self.drain_call_queue()
            if (length := self._length_cache) is None:
                length, self._width_cache = _get_index_and_columns.options(
                    resources=RayTaskCustomResources.get()
                ).remote(self._data_ref)
                self._length_cache = length
        if materialize and isinstance(length, ObjectIDType):
            self._length_cache = length = RayWrapper.materialize(length)
        return length

    def width(self, materialize=True):
        """
        Get the width of the object wrapped by the partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The width of the object.
        """
        if (width := self._width_cache) is None:
            self.drain_call_queue()
            if (width := self._width_cache) is None:
                self._length_cache, width = _get_index_and_columns.options(
                    resources=RayTaskCustomResources.get()
                ).remote(self._data_ref)
                self._width_cache = width
        if materialize and isinstance(width, ObjectIDType):
            self._width_cache = width = RayWrapper.materialize(width)
        return width

    def ip(self, materialize=True):
        """
        Get the node IP address of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if (ip := self._ip_cache) is None:
            self.drain_call_queue()
        if materialize and isinstance(ip, ObjectIDType):
            self._ip_cache = ip = RayWrapper.materialize(ip)
        return ip

    @property
    def _data(self) -> ray.ObjectRef:  # noqa: GL08
        self.drain_call_queue()
        return self._data_ref

    @property
    def _length_cache(self):  # noqa: GL08
        return self._meta[self._meta_offset]

    @_length_cache.setter
    def _length_cache(self, value):  # noqa: GL08
        self._meta[self._meta_offset] = value

    @property
    def _width_cache(self):  # noqa: GL08
        return self._meta[self._meta_offset + 1]

    @_width_cache.setter
    def _width_cache(self, value):  # noqa: GL08
        self._meta[self._meta_offset + 1] = value

    @property
    def _ip_cache(self):  # noqa: GL08
        return self._meta[-1]

    @_ip_cache.setter
    def _ip_cache(self, value):  # noqa: GL08
        self._meta[-1] = value


@ray.remote(num_returns=2)
def _get_index_and_columns(df):  # pragma: no cover
    """
    Get the number of rows and columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame which dimensions are needed.

    Returns
    -------
    int
        The number of rows.
    int
        The number of columns.
    """
    return len(df.index), len(df.columns)


PandasOnRayDataframePartition._eager_exec_func = PandasOnRayDataframePartition.apply
PandasOnRayDataframePartition._lazy_exec_func = (
    PandasOnRayDataframePartition.add_to_apply_calls
)


def _configure_lazy_exec(cls: LazyExecution):
    """Configure lazy execution mode for PandasOnRayDataframePartition."""
    mode = cls.get()
    get_logger().debug(f"Ray lazy execution mode: {mode}")
    if mode == "Auto":
        PandasOnRayDataframePartition.apply = (
            PandasOnRayDataframePartition._eager_exec_func
        )
        PandasOnRayDataframePartition.add_to_apply_calls = (
            PandasOnRayDataframePartition._lazy_exec_func
        )
    elif mode == "On":

        def lazy_exec(self, func, *args, **kwargs):
            return self._lazy_exec_func(func, *args, length=None, width=None, **kwargs)

        PandasOnRayDataframePartition.apply = lazy_exec
        PandasOnRayDataframePartition.add_to_apply_calls = (
            PandasOnRayDataframePartition._lazy_exec_func
        )
    elif mode == "Off":

        def eager_exec(self, func, *args, length=None, width=None, **kwargs):
            return self._eager_exec_func(func, *args, **kwargs)

        PandasOnRayDataframePartition.apply = (
            PandasOnRayDataframePartition._eager_exec_func
        )
        PandasOnRayDataframePartition.add_to_apply_calls = eager_exec
    else:
        raise ValueError(f"Invalid lazy execution mode: {mode}")


LazyExecution.subscribe(_configure_lazy_exec)


class SlicerHook(MaterializationHook):
    """
    Used by mask() for the slilced length computation.

    Parameters
    ----------
    ref : ObjectIDType
        Non-materialized length to be sliced.
    slc : slice
        The slice to be applied.
    """

    def __init__(self, ref: ObjectIDType, slc: slice):
        self.ref = ref
        self.slc = slc

    def pre_materialize(self):
        """
        Get the sliced length or object ref if not materialized.

        Returns
        -------
        int or ObjectIDType
        """
        if isinstance(self.ref, MetaListHook):
            len_or_ref = self.ref.pre_materialize()
            return (
                compute_sliced_len(self.slc, len_or_ref)
                if isinstance(len_or_ref, int)
                else len_or_ref
            )
        return self.ref

    def post_materialize(self, materialized):
        """
        Get the sliced length.

        Parameters
        ----------
        materialized : list or int

        Returns
        -------
        int
        """
        if isinstance(self.ref, MetaListHook):
            materialized = self.ref.post_materialize(materialized)
        return compute_sliced_len(self.slc, materialized)
