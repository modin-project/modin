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

import pandas

from modin.core.storage_formats.pandas.utils import length_fn_pandas, width_fn_pandas
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition

from distributed.client import default_client
from distributed import Future
from distributed.utils import get_ip
from dask.distributed import wait

from modin.pandas.indexing import compute_sliced_len


class PandasOnDaskDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    future : distributed.Future
        A reference to pandas DataFrame that need to be wrapped with this class.
    length : distributed.Future or int, optional
        Length or reference to it of wrapped pandas DataFrame.
    width : distributed.Future or int, optional
        Width or reference to it of wrapped pandas DataFrame.
    ip : distributed.Future or str, optional
        Node IP address or reference to it that holds wrapped pandas DataFrame.
    call_queue : list, optional
        Call queue that needs to be executed on wrapped pandas DataFrame.
    """

    def __init__(self, future, length=None, width=None, ip=None, call_queue=None):
        self.future = future
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip

    def get(self):
        """
        Get the object wrapped by this partition out of the distributed memory.

        Returns
        -------
        pandas.DataFrame
            The object from the distributed memory.
        """
        self.drain_call_queue()
        # blocking operation
        if isinstance(self.future, pandas.DataFrame):
            return self.future
        return self.future.result()

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        client = default_client()
        call_queue = self.call_queue + [[func, args, kwargs]]
        if len(call_queue) > 1:
            future = client.submit(
                apply_list_of_funcs, call_queue, self.future, pure=False
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this improves performance a bit.
            func, args, kwargs = call_queue[0]
            future = client.submit(apply_func, self.future, func, *args, **kwargs)
        futures = [
            client.submit(lambda l, i: l[i], future, i, pure=False) for i in range(2)
        ]
        return PandasOnDaskDataframePartition(futures[0], ip=futures[1])

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
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        return PandasOnDaskDataframePartition(
            self.future, call_queue=self.call_queue + [[func, args, kwargs]]
        )

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        if len(self.call_queue) == 0:
            return
        call_queue = self.call_queue
        client = default_client()
        if len(call_queue) > 1:
            future = client.submit(
                apply_list_of_funcs, call_queue, self.future, pure=False
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this improves performance a bit.
            func, args, kwargs = call_queue[0]
            future = client.submit(apply_func, self.future, func, *args, **kwargs)
        futures = [
            client.submit(lambda l, i: l[i], future, i, pure=False) for i in range(2)
        ]
        self.future = futures[0]
        self._ip_cache = futures[1]
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait(self.future)

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
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.
        """
        new_obj = super().mask(row_labels, col_labels)
        client = default_client()
        if isinstance(row_labels, slice) and isinstance(self._length_cache, Future):
            new_obj._length_cache = client.submit(
                compute_sliced_len, row_labels, self._length_cache
            )
        if isinstance(col_labels, slice) and isinstance(self._width_cache, Future):
            new_obj._width_cache = client.submit(
                compute_sliced_len, col_labels, self._width_cache
            )
        return new_obj

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnDaskDataframePartition
            A copy of this partition.
        """
        return PandasOnDaskDataframePartition(
            self.future,
            length=self._length_cache,
            width=self._width_cache,
            ip=self._ip_cache,
            call_queue=self.call_queue,
        )

    def to_pandas(self):
        """
        Convert the object wrapped by this partition to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

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
        np.ndarray.
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

    @classmethod
    def put(cls, obj):
        """
        Put an object into distributed memory and wrap it with partition object.

        Parameters
        ----------
        obj : any
            An object to be put.

        Returns
        -------
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.
        """
        client = default_client()
        return cls(client.scatter(obj, hash=False))

    @classmethod
    def preprocess_func(cls, func):
        """
        Preprocess a function before an ``apply`` call.

        Parameters
        ----------
        func : callable
            The function to preprocess.

        Returns
        -------
        callable
            An object that can be accepted by ``apply``.
        """
        return default_client().scatter(func, hash=False, broadcast=True)

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
            self._length_cache = self.apply(lambda df: len(df)).future
        if isinstance(self._length_cache, Future):
            self._length_cache = self._length_cache.result()
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
            self._width_cache = self.apply(lambda df: len(df.columns)).future
        if isinstance(self._width_cache, Future):
            self._width_cache = self._width_cache.result()
        return self._width_cache

    def ip(self):
        """
        Get the node IP address of the object wrapped by this partition.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if self._ip_cache is None:
            self._ip_cache = self.apply(lambda df: df)._ip_cache
        if isinstance(self._ip_cache, Future):
            self._ip_cache = self._ip_cache.result()
        return self._ip_cache

    @classmethod
    def empty(cls):
        """
        Create a new partition that wraps an empty pandas DataFrame.

        Returns
        -------
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.
        """
        return cls(pandas.DataFrame(), 0, 0)


def apply_func(partition, func, *args, **kwargs):
    """
    Execute a function on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the function needs to be executed on.
    func : callable
        Function that needs to be executed on `partition`.
    *args
        Additional positional arguments to be passed in `func`.
    **kwargs
        Additional keyword arguments to be passed in `func`.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    result = func(partition, *args, **kwargs)
    return result, get_ip()


def apply_list_of_funcs(funcs, partition):
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    funcs : list
        A call queue that needs to be executed on the partition.
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    for func, args, kwargs in funcs:
        partition = func(partition, *args, **kwargs)
    return partition, get_ip()
