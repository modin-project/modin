from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import pyarrow

from modin.engines.base.frame.partition import BaseFramePartition
import ray


class PyarrowOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id, length=None, width=None, call_queue=[]):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        if len(self.call_queue):
            return self.apply(lambda x: x).get()

        return ray.get(self.oid)

    def apply(self, func, **kwargs):
        """Apply a function to the object stored in this partition.

        Note: It does not matter if func is callable or an ObjectID. Ray will
            handle it correctly either way. The keyword arguments are sent as a
            dictionary.

        Args:
            func: The function to apply.

        Returns:
            A RayRemotePartition object.
        """
        oid = self.oid
        self.call_queue.append((func, kwargs))

        def call_queue_closure(oid_obj, call_queues):
            for func, kwargs in call_queues:
                if isinstance(func, ray.ObjectID):
                    func = ray.get(func)
                if isinstance(kwargs, ray.ObjectID):
                    kwargs = ray.get(kwargs)

                oid_obj = func(oid_obj, **kwargs)

            return oid_obj

        oid = deploy_ray_func.remote(
            call_queue_closure, oid, kwargs={"call_queues": self.call_queue}
        )
        self.call_queue = []

        return PyarrowOnRayFramePartition(oid)

    def add_to_apply_calls(self, func, **kwargs):
        self.call_queue.append((func, kwargs))
        return self

    def __copy__(self):
        return PyarrowOnRayFramePartition(object_id=self.oid)

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get().to_pandas()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PyarrowOnRayFramePartition(ray.put(pyarrow.Table.from_pandas(obj)))

    @classmethod
    def preprocess_func(cls, func):
        """Put a callable function into the plasma store for use in `apply`.

        Args:
            func: The function to preprocess.

        Returns:
            A ray.ObjectID.
        """
        return ray.put(func)

    @classmethod
    def length_extraction_fn(cls):
        return lambda table: table.num_rows

    @classmethod
    def width_extraction_fn(cls):
        return lambda table: table.num_columns - (1 if "index" in table.columns else 0)

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


@ray.remote
def deploy_ray_func(func, partition, kwargs):
    """Deploy a function to a partition in Ray.

    Args:
        func: The function to apply.
        partition: The partition to apply the function to.
        kwargs: A dictionary of keyword arguments for the function.

    Returns:
        The result of the function.
    """
    try:
        result = func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate
    # on it. We don't want the error to propagate to the user, and we want to
    # avoid copying unless we absolutely have to.
    except Exception:
        result = func(partition.to_pandas(), **kwargs)
        if isinstance(result, pandas.Series):
            result = pandas.DataFrame(result).T
        if isinstance(result, pandas.DataFrame):
            return pyarrow.Table.from_pandas(result)
    return result
