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

from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.base.frame.partition import PandasFramePartition

from distributed.client import get_client
from distributed import Future
from distributed.utils import get_ip
import cloudpickle as pkl
from dask.distributed import wait


def apply_list_of_funcs(funcs, df):
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    funcs : list
        A call queue that needs to be executed on the partition.
    df : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    for func, kwargs in funcs:
        if isinstance(func, bytes):
            func = pkl.loads(func)
        df = func(df, **kwargs)
    return df, get_ip()


class PandasOnDaskFramePartition(PandasFramePartition):
    """
    The class implements the interface in ``PandasFramePartition``.

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

    def apply(self, func, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable
            A function to apply.
        **kwargs
            Additional keyword arguments to be passed in ``func``.

        Returns
        -------
        PandasOnDaskFramePartition
            A new ``PandasOnDaskFramePartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        func = pkl.dumps(func)
        call_queue = self.call_queue + [[func, kwargs]]
        future = get_client().submit(
            apply_list_of_funcs, call_queue, self.future, pure=False
        )
        futures = [get_client().submit(lambda l: l[i], future) for i in range(2)]
        return PandasOnDaskFramePartition(futures[0], ip=futures[1])

    def add_to_apply_calls(self, func, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        **kwargs : dict
            Additional keyword arguments to be passed in ``func``.

        Returns
        -------
        PandasOnDaskFramePartition
            A new ``PandasOnDaskFramePartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        return PandasOnDaskFramePartition(
            self.future, call_queue=self.call_queue + [[pkl.dumps(func), kwargs]]
        )

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        if len(self.call_queue) == 0:
            return
        new_partition = self.apply(lambda x: x)
        self.future = new_partition.future
        self._ip_cache = new_partition._ip_cache
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait(self.future)

    def mask(self, row_indices, col_indices):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_indices : list-like
            The indices for the rows to extract.
        col_indices : list-like
            The indices for the columns to extract.

        Returns
        -------
        PandasOnDaskFramePartition
            A new ``PandasOnDaskFramePartition`` object.
        """
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        if not isinstance(row_indices, slice):
            new_obj._length_cache = len(row_indices)
        if not isinstance(col_indices, slice):
            new_obj._width_cache = len(col_indices)
        return new_obj

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnDaskFramePartition
            A copy of this partition.
        """
        return PandasOnDaskFramePartition(
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
        PandasOnDaskFramePartition
            A new ``PandasOnDaskFramePartition`` object.
        """
        client = get_client()
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
        PandasOnDaskFramePartition
            A new ``PandasOnDaskFramePartition`` object.
        """
        return cls(pandas.DataFrame(), 0, 0)
