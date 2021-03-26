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

from abc import ABC


class BaseFramePartition(ABC):  # pragma: no cover
    """An abstract class that holds the data and metadata for a single partition.

    The public API exposed by the children of this object is used in `BaseFrameManager`.

    Note: These objects are treated as immutable by `BaseFrameManager`
    subclasses. There is no logic for updating inplace.
    """

    def get(self):
        """Return the object wrapped by this one to the original format.

        Note: This is the opposite of the classmethod `put`.
            E.g. if you assign `x = BaseFramePartition.put(1)`, `x.get()` should
            always return 1.

        Returns
        -------
            The object that was `put`.
        """
        pass

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args
        ----
        func : callable
            The function to apply.

        Returns
        -------
             A new `BaseFramePartition` containing the object that has had `func`
             applied to it.
        """
        pass

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        Note: This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return

        Args
        ----
        func : callable
            The function to apply.

        Returns
        -------
            A new `BaseFramePartition` with the function added to the call queue.
        """
        pass

    def drain_call_queue(self):
        """Execute all functionality stored in the call queue."""
        pass

    def wait(self):
        """Wait for partition' computations result."""
        pass

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns
        -------
            A Pandas DataFrame.
        """
        pass

    def to_numpy(self, **kwargs):
        """Convert the object stored in this partition to a NumPy array.

        Note: If the underlying object is a Pandas DataFrame, this will return
            a 2D NumPy array.

        Returns
        -------
            A NumPy array.
        """
        pass

    def mask(self, row_indices, col_indices):
        """Lazily create a mask that extracts the indices provided.

        Args:
            row_indices: The indices for the rows to extract.
            col_indices: The indices for the columns to extract.

        Returns
        -------
            A `BaseFramePartition` object.
        """
        pass

    @classmethod
    def put(cls, obj):
        """Format a given object.

        Parameters
        ----------
        obj: object
            An object.

        Returns
        -------
            A `BaseFramePartition` object.
        """
        pass

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `BaseFramePartition`
            objects.

        Args
        ----
        func : callable
            The function to preprocess.

        Returns
        -------
            An object that can be accepted by `apply`.
        """
        pass

    @classmethod
    def length_extraction_fn(cls):
        """Compute the length of the object in this partition.

        Returns
        -------
            A callable function.
        """
        pass

    @classmethod
    def width_extraction_fn(cls):
        """Compute the width of the object in this partition.

        Returns
        -------
            A callable function.
        """
        pass

    _length_cache = None
    _width_cache = None

    def length(self):
        """Return the length of partition."""
        if self._length_cache is None:
            cls = type(self)
            func = cls.length_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)
            self._length_cache = self.apply(preprocessed_func)
        return self._length_cache

    def width(self):
        """Return the width of partition."""
        if self._width_cache is None:
            cls = type(self)
            func = cls.width_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)
            self._width_cache = self.apply(preprocessed_func)
        return self._width_cache

    @classmethod
    def empty(cls):
        """Create an empty partition.

        Returns
        -------
            An empty partition
        """
        pass
