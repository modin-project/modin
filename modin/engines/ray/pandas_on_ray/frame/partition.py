import pandas

from modin.engines.base.frame.partition import BaseFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.ray.utils import handle_ray_task_error
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray
    from ray.worker import RayTaskError


class PandasOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id, length=None, width=None, call_queue=None):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        if len(self.call_queue):
            self.drain_call_queue()
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
        result, length, width = deploy_ray_func.remote(call_queue, oid)
        return PandasOnRayFramePartition(result, length, width)

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnRayFramePartition(
            self.oid, call_queue=self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        self.oid, self._length_cache, self._width_cache = deploy_ray_func.remote(
            call_queue, oid
        )
        self.call_queue = []

    def __copy__(self):
        return PandasOnRayFramePartition(
            self.oid, self._length_cache, self._width_cache, self.call_queue
        )

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def to_numpy(self):
        """Convert the object stored in this parition to a Numpy Array.

        Returns:
            A Numpy Array.
        """
        return self.apply(lambda df: df.values).get()

    def mask(self, row_indices, col_indices):
        if (
            isinstance(row_indices, slice)
            or (
                self._length_cache is not None
                and len(row_indices) == self._length_cache
            )
        ) and (
            isinstance(col_indices, slice)
            or (self._width_cache is not None and len(col_indices) == self._width_cache)
        ):
            return self.__copy__()

        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        new_obj._length_cache = (
            len(row_indices)
            if not isinstance(row_indices, slice)
            else self._length_cache
        )
        new_obj._width_cache = (
            len(col_indices)
            if not isinstance(col_indices, slice)
            else self._width_cache
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

    def length(self):
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._length_cache, ray.ObjectID):
            try:
                self._length_cache = ray.get(self._length_cache)
            except RayTaskError as e:
                handle_ray_task_error(e)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._width_cache, ray.ObjectID):
            try:
                self._width_cache = ray.get(self._width_cache)
            except RayTaskError as e:
                handle_ray_task_error(e)
        return self._width_cache

    @classmethod
    def length_extraction_fn(cls):
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        return width_fn_pandas

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


if __execution_engine__ == "Ray":

    @ray.remote(num_return_vals=2)
    def get_index_and_columns(df):
        return len(df.index), len(df.columns)

    @ray.remote(num_return_vals=3)
    def deploy_ray_func(call_queue, partition):  # pragma: no cover
        def deserialize(obj):
            if isinstance(obj, ray.ObjectID):
                return ray.get(obj)
            return obj

        if len(call_queue) > 1:
            for func, kwargs in call_queue[:-1]:
                func = deserialize(func)
                kwargs = deserialize(kwargs)
                try:
                    partition = func(partition, **kwargs)
                except ValueError:
                    partition = func(partition.copy(), **kwargs)
        func, kwargs = call_queue[-1]
        func = deserialize(func)
        kwargs = deserialize(kwargs)
        try:
            result = func(partition, **kwargs)
        # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
        # don't want the error to propagate to the user, and we want to avoid copying unless
        # we absolutely have to.
        except ValueError:
            result = func(partition.copy(), **kwargs)
        return (
            result,
            len(result) if hasattr(result, "__len__") else 0,
            len(result.columns) if hasattr(result, "columns") else 0,
        )
