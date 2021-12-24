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
from modin.pandas.indexing import compute_sliced_len
from modin.core.execution.ray.common.task_wrapper import ObjectRef, RayWrapper
from modin.core.execution.ray.common.utils import get_node_ip_address


class PandasOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    object_id : ObjectRef
        A reference to ``pandas.DataFrame`` that need to be wrapped with this class.
    length : ObjectRef or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : ObjectRef or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    ip : ObjectRef or str, optional
        Node IP address or reference to it that holds wrapped ``pandas.DataFrame``.
    call_queue : list
        Call queue that needs to be executed on wrapped ``pandas.DataFrame``.
    """

    def __init__(self, object_id, length=None, width=None, ip=None, call_queue=None):
        assert isinstance(object_id, ObjectRef)

        self.oid = object_id
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip

    @property
    def future(self):
        """
        Should be moved in abstract class.
        """
        # self.drain_call_queue ?
        return self.oid

    def get(self):
        """
        Get the object wrapped by this partition out of the Plasma store.

        Returns
        -------
        pandas.DataFrame
            The object from the Plasma store.
        """
        if len(self.call_queue):
            self.drain_call_queue()
        return RayWrapper.materialize(self.oid)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable or ObjectRef
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
        It does not matter if `func` is callable or an ``ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        oid = self.oid
        call_queue = self.call_queue + [(func, args, kwargs)]
        if len(call_queue) > 1:
            result, length, width, ip = RayWrapper.deploy(
                _apply_list_of_funcs,
                4,
                call_queue,
                oid,
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, args, kwargs = call_queue[0]
            result, length, width, ip = RayWrapper.deploy(
                _apply_func, 4, oid, func, *args, **kwargs
            )
        return PandasOnRayDataframePartition(result, length, width, ip)

    def add_to_apply_calls(self, func, *args, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable or ObjectRef
            Function to be added to the call queue.
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
        It does not matter if `func` is callable or an ``ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        return PandasOnRayDataframePartition(
            self.oid, call_queue=self.call_queue + [(func, args, kwargs)]
        )

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        if len(call_queue) > 1:
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = RayWrapper.deploy(
                _apply_list_of_funcs,
                4,
                call_queue,
                oid,
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, args, kwargs = call_queue[0]
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = RayWrapper.deploy(_apply_func, 4, oid, func, *args, **kwargs)
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        RayWrapper.wait([self.oid])

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnRayDataframePartition
            A copy of this partition.
        """
        return PandasOnRayDataframePartition(
            self.oid,
            length=self._length_cache,
            width=self._width_cache,
            ip=self._ip_cache,
            call_queue=self.call_queue,
        )

    def to_pandas(self):
        """
        Convert the object wrapped by this partition to a ``pandas.DataFrame``.

        Returns
        -------
        pandas DataFrame.
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
        np.ndarray
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

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
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        new_obj = super().mask(row_indices, col_indices)
        if isinstance(row_indices, slice) and isinstance(self._length_cache, ObjectRef):
            new_obj._length_cache = RayWrapper.deploy(
                compute_sliced_len,
                1,
                row_indices,
                self._length_cache,
            )
        if isinstance(col_indices, slice) and isinstance(self._width_cache, ObjectRef):
            new_obj._width_cache = RayWrapper.deploy(
                compute_sliced_len,
                1,
                col_indices,
                self._width_cache,
            )
        return new_obj

    @classmethod
    def put(cls, obj):
        """
        Put an object into Plasma store and wrap it with partition object.

        Parameters
        ----------
        obj : any
            An object to be put.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        return PandasOnRayDataframePartition(
            RayWrapper.put(obj), len(obj.index), len(obj.columns)
        )

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
        ObjectRef
            A reference to `func`.
        """
        return RayWrapper.put(func)

    def length(self):
        """
        Get the length of the object wrapped by this partition.

        Returns
        -------
        int
            The length of the object.
        """
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = RayWrapper.deploy(
                    _get_index_and_columns,
                    2,
                    self.oid,
                )
        if isinstance(self._length_cache, ObjectRef):
            self._length_cache = RayWrapper.materialize(self._length_cache)
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
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = RayWrapper.deploy(
                    _get_index_and_columns,
                    2,
                    self.oid,
                )
        if isinstance(self._width_cache, ObjectRef):
            self._width_cache = RayWrapper.materialize(self._width_cache)
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
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._ip_cache = self.apply(lambda df: df)._ip_cache
        if isinstance(self._ip_cache, ObjectRef):
            self._ip_cache = RayWrapper.materialize(self._ip_cache)
        return self._ip_cache

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

    @classmethod
    def empty(cls):
        """
        Create a new partition that wraps an empty pandas DataFrame.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        return cls.put(pandas.DataFrame())


def _get_index_and_columns(df):
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


def _apply_func(partition, func, *args, **kwargs):  # pragma: no cover
    """
    Execute a function on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the function needs to be executed on.
    func : callable
        Function that needs to be executed on the partition.
    *args : iterable
        Additional positional arguments to be passed in `func`.
    **kwargs : dict
        Additional keyword arguments to be passed in `func`.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    int
        The number of rows of the resulting pandas DataFrame.
    int
        The number of columns of the resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    try:
        result = func(partition, *args, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), *args, **kwargs)
    return (
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
        get_node_ip_address(),
    )


def _apply_list_of_funcs(funcs, partition):  # pragma: no cover
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
    int
        The number of rows of the resulting pandas DataFrame.
    int
        The number of columns of the resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """

    def deserialize(obj):
        if isinstance(obj, ObjectRef):
            return RayWrapper.materialize(obj)
        elif isinstance(obj, (tuple, list)) and any(
            isinstance(o, ObjectRef) for o in obj
        ):
            return RayWrapper.materialize(list(obj))
        elif isinstance(obj, dict) and any(
            isinstance(val, ObjectRef) for val in obj.values()
        ):
            return dict(zip(obj.keys(), RayWrapper.materialize(list(obj.values()))))
        else:
            return obj

    for func, args, kwargs in funcs:
        func = deserialize(func)
        args = deserialize(args)
        kwargs = deserialize(kwargs)
        try:
            partition = func(partition, *args, **kwargs)
        # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
        # don't want the error to propagate to the user, and we want to avoid copying unless
        # we absolutely have to.
        except ValueError:
            partition = func(partition.copy(), *args, **kwargs)

    return (
        partition,
        len(partition) if hasattr(partition, "__len__") else 0,
        len(partition.columns) if hasattr(partition, "columns") else 0,
        get_node_ip_address(),
    )
