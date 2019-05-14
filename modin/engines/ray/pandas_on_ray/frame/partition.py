from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray
from ray.worker import RayTaskError

from modin.engines.base.frame.partition import BaseFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.ray.utils import handle_ray_task_error


class PandasOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id, length=None, width=None):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.call_queue = []
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        if len(self.call_queue):
            return self.apply(lambda x: x).get()
        try:
            return ray.get(self.oid)
        except RayTaskError as e:
            handle_ray_task_error(e)

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
        call_queue = self.call_queue + [(func, kwargs)]
        new_obj, result, length, width = deploy_ray_func.remote(call_queue, oid)
        if len(self.call_queue) > 0:
            self.oid = new_obj
            self.call_queue = []
        return PandasOnRayFramePartition(result, PandasOnRayFramePartition(length), PandasOnRayFramePartition(width))

    def add_to_apply_calls(self, func, **kwargs):
        self.call_queue.append((func, kwargs))

    def __copy__(self):
        return PandasOnRayFramePartition(self.oid, self._length_cache, self._width_cache)

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def mask(self, row_indices, col_indices):
        new_obj = PandasOnRayFramePartition(self.oid, len(row_indices), len(col_indices))
        if len(self.call_queue) > 0:
            [new_obj.add_to_apply_calls(call, **kwargs) for call, kwargs in self.call_queue]
        new_obj.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        return new_obj

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PandasOnRayFramePartition(ray.put(obj), len(obj.index), len(obj.columns))

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
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        return width_fn_pandas

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


@ray.remote(num_return_vals=4)
def deploy_ray_func(call_queue, oid_obj):

    def deserialize(obj):
        if isinstance(obj, ray.ObjectID):
            return ray.get(obj)
        return obj

    if len(call_queue) > 1:
        for func, kwargs in call_queue[:-1]:
            func = deserialize(func)
            kwargs = deserialize(kwargs)
            oid_obj = func(oid_obj, **kwargs)
    func, kwargs = call_queue[-1]
    func = deserialize(func)
    kwargs = deserialize(kwargs)
    result = func(oid_obj, **kwargs)
    return oid_obj if len(call_queue) > 1 else None, result, len(result) if hasattr(result, "__len__") else 0, len(result.columns) if hasattr(result, "columns") else 0


@ray.remote(num_return_vals=3)
def _deploy_ray_func(func, partition, kwargs):  # pragma: no cover
    """Deploy a function to a partition in Ray.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)

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
    except ValueError:
        result = func(partition.copy(), **kwargs)
    return result, len(result) if hasattr(result, "__len__") else 0, len(result.columns) if hasattr(result, "columns") else 0
