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

from abc import ABC
from modin.pandas.indexing import compute_sliced_len
from copy import copy

from pandas.api.types import is_scalar


class PandasDataframePartition(ABC):  # pragma: no cover
    """
    An abstract class that is base for any partition class of ``pandas`` storage format.

    The class providing an API that has to be overridden by child classes.
    """

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
        pass

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

    def add_to_apply_calls(self, func, *args, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
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
        pass

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        pass

    def wait(self):
        """Wait for completion of computations on the object wrapped by the partition."""
        pass

    def to_pandas(self):
        """
        Convert the object wrapped by this partition to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        If the underlying object is a pandas DataFrame, this will likely
        only need to call `get`.
        """
        pass

    def to_numpy(self, **kwargs):
        """
        Convert the object wrapped by this partition to a NumPy array.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed in `to_numpy`.

        Returns
        -------
        np.ndarray

        Notes
        -----
        If the underlying object is a pandas DataFrame, this will return
        a 2D NumPy array.
        """
        pass

    def mask(self, row_indices, col_indices):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_indices : list-like, slice or label
            The indices for the rows to extract.
        col_indices : list-like, slice or label
            The indices for the columns to extract.

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

        row_indices = [row_indices] if is_scalar(row_indices) else row_indices
        col_indices = [col_indices] if is_scalar(col_indices) else col_indices

        if is_full_axis_mask(row_indices, self._length_cache) and is_full_axis_mask(
            col_indices, self._width_cache
        ):
            return copy(self)

        new_obj = self.add_to_apply_calls(lambda df: df.iloc[row_indices, col_indices])

        def try_recompute_cache(indices, previous_cache):
            """Compute new axis-length cache for the masked frame based on its previous cache."""
            if not isinstance(indices, slice):
                return len(indices)
            if not isinstance(previous_cache, int):
                return None
            return compute_sliced_len(indices, previous_cache)

        new_obj._length_cache = try_recompute_cache(row_indices, self._length_cache)
        new_obj._width_cache = try_recompute_cache(col_indices, self._width_cache)
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
        pass

    @classmethod
    def _width_extraction_fn(cls):
        """
        Return the function that computes the width of the object wrapped by this partition.

        Returns
        -------
        callable
            The function that computes the width of the object wrapped by this partition.
        """
        pass

    _length_cache = None
    _width_cache = None

    def length(self):
        """
        Get the length of the object wrapped by this partition.

        Returns
        -------
        int
            The length of the object.
        """
        if self._length_cache is None:
            cls = type(self)
            func = cls._length_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)
            self._length_cache = self.apply(preprocessed_func)
        return self._length_cache

    def width(self):
        """
        Get the width of the object wrapped by the partition.

        Returns
        -------
        int
            The width of the object.
        """
        if self._width_cache is None:
            cls = type(self)
            func = cls._width_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)
            self._width_cache = self.apply(preprocessed_func)
        return self._width_cache

    @classmethod
    def empty(cls):
        """
        Create a new partition that wraps an empty pandas DataFrame.

        Returns
        -------
        PandasDataframePartition
            New `PandasDataframePartition` object.
        """
        pass
