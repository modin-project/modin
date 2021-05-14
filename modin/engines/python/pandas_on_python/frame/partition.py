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

"""The module defines interface for a partition with pandas backend and Python engine."""

import pandas

from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.base.frame.partition import PandasFramePartition


class PandasOnPythonFramePartition(PandasFramePartition):
    """
    Partition class with interface for pandas backend and Python engine.

    Class holds the data and metadata for a single partition and implements
    methods of parent abstract class ``BaseFramePartition``.

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

    def __init__(self, data, length=None, width=None, call_queue=None):
        self.data = data
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
        return self.data.copy()

    def apply(self, func, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable
            Function to apply.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnPythonFramePartition
            New ``PandasOnPythonFramePartition`` object.
        """

        def call_queue_closure(data, call_queues):
            """
            Apply callables from `call_queues` on copy of the `data` and return the result.

            Parameters
            ----------
            data : pandas.DataFrame or pandas.Series
                Data to use for computations.
            call_queues : array-like
                Array with callables and it's kwargs to be applied to the `data`.

            Returns
            -------
            pandas.DataFrame or pandas.Series
            """
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
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnPythonFramePartition
            New ``PandasOnPythonFramePartition`` object with extended call queue.
        """
        return PandasOnPythonFramePartition(
            self.data.copy(), call_queue=self.call_queue + [(func, kwargs)]
        )

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

    # FIXME: row_indices and col_indices can't be optional - df.iloc[None, None]
    # will raise ValueError
    def mask(self, row_indices=None, col_indices=None):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_indices : list-like, optional
            The indices for the rows to extract.
        col_indices : list-like, optional
            The indices for the columns to extract.

        Returns
        -------
        PandasOnPythonFramePartition
            New ``PandasOnPythonFramePartition`` object.
        """
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        if not isinstance(row_indices, slice):
            new_obj._length_cache = len(row_indices)
        if not isinstance(col_indices, slice):
            new_obj._width_cache = len(col_indices)
        return new_obj

    def to_pandas(self):
        """
        Return copy of the ``pandas.Dataframe`` stored in this partition.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Equivalent to ``get`` method for this class.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    def to_numpy(self, **kwargs):
        """
        Return NumPy array representation of ``pandas.DataFrame`` stored in this partition.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass into `pandas.DataFrame.to_numpy` function.

        Returns
        -------
        np.ndarray
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

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
        PandasOnPythonFramePartition
            New ``PandasOnPythonFramePartition`` object.
        """
        return cls(obj)

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
            self._length_cache = self.apply(self._length_extraction_fn()).data
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
            self._width_cache = self.apply(self._width_extraction_fn()).data
        return self._width_cache

    @classmethod
    def empty(cls):
        """
        Create a new partition that wraps an empty pandas DataFrame.

        Returns
        -------
        PandasOnPythonFramePartition
            New ``PandasOnPythonFramePartition`` object wrapping empty pandas DataFrame.
        """
        return cls(pandas.DataFrame())
