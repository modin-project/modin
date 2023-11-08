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

"""The module defines interface for a partition with pandas storage format and Python engine."""

import warnings

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.python.common import PythonWrapper


class PandasOnPythonDataframePartition(PandasDataframePartition):
    """
    Partition class with interface for pandas storage format and Python engine.

    Class holds the data and metadata for a single partition and implements
    methods of parent abstract class ``PandasDataframePartition``.

    Parameters
    ----------
    data : pandas.DataFrame
        ``pandas.DataFrame`` that should be wrapped with this class.
    length : int, optional
        Length of `data` (number of rows in the input dataframe).
    width : int, optional
        Width of `data` (number of columns in the input dataframe).
    call_queue : list, optional
        Call queue of the partition (list with entities that should be called
        before partition materialization).

    Notes
    -----
    Objects of this class are treated as immutable by partition manager
    subclasses. There is no logic for updating in-place.
    """

    execution_wrapper = PythonWrapper

    def __init__(self, data, length=None, width=None, call_queue=None):
        super().__init__()
        if hasattr(data, "copy"):
            data = data.copy()
        self._data = data
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """
        Flush the `call_queue` and return copy of the data.

        Returns
        -------
        pandas.DataFrame
            Copy of DataFrame that was wrapped by this partition.

        Notes
        -----
        Since this object is a simple wrapper, just return the copy of data.
        """
        self.drain_call_queue()
        return self._data.copy() if hasattr(self._data, "copy") else self._data

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
        PandasOnPythonDataframePartition
            New ``PandasOnPythonDataframePartition`` object.
        """

        def call_queue_closure(data, call_queue):
            """
            Apply callables from `call_queue` on copy of the `data` and return the result.

            Parameters
            ----------
            data : pandas.DataFrame or pandas.Series
                Data to use for computations.
            call_queue : array-like
                Array with callables and it's kwargs to be applied to the `data`.

            Returns
            -------
            pandas.DataFrame or pandas.Series
            """
            result = data.copy()
            for func, f_args, f_kwargs in call_queue:
                try:
                    result = func(result, *f_args, **f_kwargs)
                except Exception as err:
                    self.call_queue = []
                    raise err
            return result

        self._data = call_queue_closure(self._data, self.call_queue)
        self.call_queue = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return self.__constructor__(func(self._data.copy(), *args, **kwargs))

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        if len(self.call_queue) == 0:
            return
        self.apply(lambda x: x)

    def wait(self):
        """
        Wait for completion of computations on the object wrapped by the partition.

        Internally will be done by flushing the call queue.
        """
        self.drain_call_queue()

    @classmethod
    def put(cls, obj):
        """
        Create partition containing `obj`.

        Parameters
        ----------
        obj : pandas.DataFrame
            DataFrame to be put into the new partition.

        Returns
        -------
        PandasOnPythonDataframePartition
            New ``PandasOnPythonDataframePartition`` object.
        """
        return cls(obj.copy(), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """
        Preprocess a function before an ``apply`` call.

        Parameters
        ----------
        func : callable
            Function to preprocess.

        Returns
        -------
        callable
            An object that can be accepted by ``apply``.

        Notes
        -----
        No special preprocessing action is required, so unmodified
        `func` will be returned.
        """
        return func
