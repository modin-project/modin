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

"""The module defines base interface for a partition of a Modin DataFrame."""

from __future__ import annotations

import logging
import uuid
from abc import ABC
from copy import copy
from functools import cached_property

import pandas
from pandas.api.types import is_scalar

from modin.core.storage_formats.pandas.utils import length_fn_pandas, width_fn_pandas
from modin.logging import ClassLogger, get_logger
from modin.logging.config import LogLevel
from modin.pandas.indexing import compute_sliced_len


class PandasDataframePartition(
    ABC, ClassLogger, modin_layer="BLOCK-PARTITION", log_level=LogLevel.DEBUG
):  # pragma: no cover
    """
    An abstract class that is base for any partition class of ``pandas`` storage format.

    The class providing an API that has to be overridden by child classes.
    """

    _length_cache = None
    _width_cache = None
    _identity_cache = None
    _data = None
    execution_wrapper = None

    # these variables are intentionally initialized at runtime
    # so as not to initialize the engine during import
    _iloc_func = None

    def __init__(self):
        if type(self)._iloc_func is None:
            # Places `_iloc` function into the storage to speed up
            # remote function calls and caches the result.
            # It also postpones engine initialization, which happens
            # implicitly when `execution_wrapper.put` is called.
            if self.execution_wrapper is not None:
                type(self)._iloc_func = staticmethod(
                    self.execution_wrapper.put(self._iloc)
                )
            else:
                type(self)._iloc_func = staticmethod(self._iloc)

    @cached_property
    def __constructor__(self) -> type[PandasDataframePartition]:
        """
        Create a new instance of this object.

        Returns
        -------
        PandasDataframePartition
            New instance of pandas partition.
        """
        return type(self)

    def get(self):
        """
        Get the object wrapped by this partition.

        Returns
        -------
        object
            The object that was wrapped by this partition.

        Notes
        -----
        This is the opposite of the classmethod `put`.
        E.g. if you assign `x = PandasDataframePartition.put(1)`, `x.get()` should
        always return 1.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.get::{self._identity}")
        self.drain_call_queue()
        result = self.execution_wrapper.materialize(self._data)
        self._is_debug(log) and log.debug(f"EXIT::Partition.get::{self._identity}")
        return result

    @property
    def list_of_blocks(self):
        """
        Get the list of physical partition objects that compose this partition.

        Returns
        -------
        list
            A list of physical partition objects (``ray.ObjectRef``, ``distributed.Future`` e.g.).
        """
        # Defer draining call queue until we get the partitions.
        # TODO Look into draining call queue at the same time as the task
        self.drain_call_queue()
        return [self._data]

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable
            Function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasDataframePartition
            New `PandasDataframePartition` object.

        Notes
        -----
        It is up to the implementation how `kwargs` are handled. They are
        an important part of many implementations. As of right now, they
        are not serialized.
        """
        pass

    def add_to_apply_calls(self, func, *args, length=None, width=None, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        length : reference or int, optional
            Length, or reference to length, of wrapped ``pandas.DataFrame``.
        width : reference or int, optional
            Width, or reference to width, of wrapped ``pandas.DataFrame``.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasDataframePartition
            New `PandasDataframePartition` object with the function added to the call queue.

        Notes
        -----
        This function will be executed when `apply` is called. It will be executed
        in the order inserted; apply's func operates the last and return.
        """
        return self.__constructor__(
            self._data,
            call_queue=self.call_queue + [[func, args, kwargs]],
            length=length,
            width=width,
        )

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        pass

    def wait(self):
        """Wait for completion of computations on the object wrapped by the partition."""
        pass

    def to_pandas(self):
        """
        Convert the object wrapped by this partition to a ``pandas.DataFrame``.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        If the underlying object is a pandas DataFrame, this will likely
        only need to call `get`.
        """
        dataframe = self.get()
        assert isinstance(dataframe, (pandas.DataFrame, pandas.Series))
        return dataframe

    def to_numpy(self, **kwargs):
        """
        Convert the object wrapped by this partition to a NumPy array.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed in ``to_numpy``.

        Returns
        -------
        np.ndarray

        Notes
        -----
        If the underlying object is a pandas DataFrame, this will return
        a 2D NumPy array.
        """
        return self.apply(lambda df: df.to_numpy(**kwargs)).get()

    @staticmethod
    def _iloc(df, row_labels, col_labels):  # noqa: RT01, PR01
        """Perform `iloc` on dataframes wrapped in partitions (helper function)."""
        return df.iloc[row_labels, col_labels]

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
        PandasDataframePartition
            New `PandasDataframePartition` object.
        """

        def is_full_axis_mask(index, axis_length):
            """Check whether `index` mask grabs `axis_length` amount of elements."""
            if isinstance(index, slice):
                return index == slice(None) or (
                    isinstance(axis_length, int)
                    and compute_sliced_len(index, axis_length) == axis_length
                )
            return (
                hasattr(index, "__len__")
                and isinstance(axis_length, int)
                and len(index) == axis_length
            )

        row_labels = [row_labels] if is_scalar(row_labels) else row_labels
        col_labels = [col_labels] if is_scalar(col_labels) else col_labels

        if is_full_axis_mask(row_labels, self._length_cache) and is_full_axis_mask(
            col_labels, self._width_cache
        ):
            return copy(self)

        new_obj = self.add_to_apply_calls(self._iloc_func, row_labels, col_labels)

        def try_recompute_cache(indices, previous_cache):
            """Compute new axis-length cache for the masked frame based on its previous cache."""
            if not isinstance(indices, slice):
                return len(indices)
            if not isinstance(previous_cache, int):
                return None
            return compute_sliced_len(indices, previous_cache)

        new_obj._length_cache = try_recompute_cache(row_labels, self._length_cache)
        new_obj._width_cache = try_recompute_cache(col_labels, self._width_cache)
        return new_obj

    @classmethod
    def put(cls, obj):
        """
        Put an object into a store and wrap it with partition object.

        Parameters
        ----------
        obj : object
            An object to be put.

        Returns
        -------
        PandasDataframePartition
            New `PandasDataframePartition` object.
        """
        pass

    @classmethod
    def preprocess_func(cls, func):
        """
        Preprocess a function before an `apply` call.

        Parameters
        ----------
        func : callable
            Function to preprocess.

        Returns
        -------
        callable
            An object that can be accepted by `apply`.

        Notes
        -----
        This is a classmethod because the definition of how to preprocess
        should be class-wide. Also, we may want to use this before we
        deploy a preprocessed function to multiple `PandasDataframePartition`
        objects.
        """
        pass

    @classmethod
    def _length_extraction_fn(cls):
        """
        Return the function that computes the length of the object wrapped by this partition.

        Returns
        -------
        callable
            The function that computes the length of the object wrapped by this partition.
        """
        return length_fn_pandas

    @classmethod
    def _width_extraction_fn(cls):
        """
        Return the function that computes the width of the object wrapped by this partition.

        Returns
        -------
        callable
            The function that computes the width of the object wrapped by this partition.
        """
        return width_fn_pandas

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
        int or its Future
            The length of the object.
        """
        if self._length_cache is None:
            self._length_cache = self.apply(self._length_extraction_fn()).get()
        return self._length_cache

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
        int or its Future
            The width of the object.
        """
        if self._width_cache is None:
            self._width_cache = self.apply(self._width_extraction_fn()).get()
        return self._width_cache

    @property
    def _identity(self):
        """
        Calculate identifier on request for debug logging mode.

        Returns
        -------
        str
        """
        if self._identity_cache is None:
            self._identity_cache = uuid.uuid4().hex
        return self._identity_cache

    def split(self, split_func, num_splits, *args):
        """
        Split the object wrapped by the partition into multiple partitions.

        Parameters
        ----------
        split_func : Callable[pandas.DataFrame, List[Any]] -> List[pandas.DataFrame]
            The function that will split this partition into multiple partitions. The list contains
            pivots to split by, and will have the same dtype as the major column we are shuffling on.
        num_splits : int
            The number of resulting partitions (may be empty).
        *args : List[Any]
            Arguments to pass to ``split_func``.

        Returns
        -------
        list
            A list of partitions.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.split::{self._identity}")

        self._is_debug(log) and log.debug(f"SUBMIT::_split_df::{self._identity}")
        outputs = self.execution_wrapper.deploy(
            split_func, [self._data] + list(args), num_returns=num_splits
        )
        self._is_debug(log) and log.debug(f"EXIT::Partition.split::{self._identity}")
        return [self.__constructor__(output) for output in outputs]

    @classmethod
    def empty(cls):
        """
        Create a new partition that wraps an empty pandas DataFrame.

        Returns
        -------
        PandasDataframePartition
            New `PandasDataframePartition` object.
        """
        return cls.put(pandas.DataFrame(), 0, 0)

    def _is_debug(self, logger=None):
        """
        Check that the logger is set to debug mode.

        Parameters
        ----------
        logger : logging.logger, optional
            Logger obtained from Modin's `get_logger` utility.
            Explicit transmission of this parameter can be used in the case
            when within the context of `_is_debug` call there was already
            `get_logger` call. This is an optimization.

        Returns
        -------
        bool
        """
        if logger is None:
            logger = get_logger()
        return logger.isEnabledFor(logging.DEBUG)
