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

import pandas

from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.base.frame.partition import BaseFramePartition


class PandasOnPythonFramePartition(BaseFramePartition):
    """This abstract class holds the data and metadata for a single partition.
    The methods required for implementing this abstract class are listed in
    the section immediately following this.

    The API exposed by the children of this object is used in
    `BaseFrameManager`.

    Note: These objects are treated as immutable by `BaseFrameManager`
    subclasses. There is no logic for updating inplace.
    """

    def __init__(self, data, length=None, width=None, call_queue=None):
        self.data = data
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Flushes the call_queue and returns the data.

        Note: Since this object is a simple wrapper, just return the data.

        Returns:
            The object that was `put`.
        """
        self.drain_call_queue()
        return self.data.copy()

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args:
            func: The lambda to apply (may already be correctly formatted)

        Returns:
             A new `BaseFramePartition` containing the object that has had `func`
             applied to it.
        """

        def call_queue_closure(data, call_queues):
            result = data.copy()
            for func, kwargs in call_queues:
                try:
                    result = func(result, **kwargs)
                except Exception as e:
                    self.call_queue = []
                    raise e
            return result

        self.data = call_queue_closure(self.data, self.call_queue)
        self.call_queue = []
        return PandasOnPythonFramePartition(func(self.data.copy(), **kwargs))

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnPythonFramePartition(
            self.data.copy(), call_queue=self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        self.apply(lambda x: x)

    def mask(self, row_indices=None, col_indices=None):
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        if not isinstance(row_indices, slice):
            new_obj._length_cache = len(row_indices)
        if not isinstance(col_indices, slice):
            new_obj._width_cache = len(col_indices)
        return new_obj

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    def to_numpy(self, **kwargs):
        """
        Convert the object stored in this partition to a NumPy array.

        Returns
        -------
            A NumPy array.
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        return cls(obj)

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `BaseFramePartition`
            objects.

        Args:
            func: The function to preprocess.

        Returns:
            An object that can be accepted by `apply`.
        """
        return func

    @classmethod
    def length_extraction_fn(cls):
        """The function to compute the length of the object in this partition.

        Returns:
            A callable function.
        """
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        """The function to compute the width of the object in this partition.

        Returns:
            A callable function.
        """
        return width_fn_pandas

    _length_cache = None
    _width_cache = None

    def length(self):
        if self._length_cache is None:
            self._length_cache = self.apply(self.length_extraction_fn()).data
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            self._width_cache = self.apply(self.width_extraction_fn()).data
        return self._width_cache

    @classmethod
    def empty(cls):
        return cls(pandas.DataFrame())
