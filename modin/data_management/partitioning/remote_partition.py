from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray


class RemotePartition(object):
    """This abstract class holds the data and metadata for a single partition.
        The methods required for implementing this abstract class are listed in
        the section immediately following this.

        The API exposed by the children of this object is used in
        `BlockPartitions`.

        Note: These objects are treated as immutable by `BlockPartitions`
        subclasses. There is no logic for updating inplace.
    """

    # Abstract methods and fields. These must be implemented in order to
    # properly subclass this object. There are also some abstract classmethods
    # to implement.
    def get(self):
        """Return the object wrapped by this one to the original format.

        Note: This is the opposite of the classmethod `put`.
            E.g. if you assign `x = RemotePartition.put(1)`, `x.get()` should
            always return 1.

        Returns:
            The object that was `put`.
        """
        raise NotImplementedError("Must be implemented in child class")

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args:
            func: The lambda to apply (may already be correctly formatted)

        Returns:
             A new `RemotePartition` containing the object that has had `func`
             applied to it.
        """
        raise NotImplementedError("Must be implemented in child class")

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return
        """
        raise NotImplementedError("Must be implemented in child class")

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns:
            A Pandas DataFrame.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `RemotePartition`
            objects.

        Args:
            func: The function to preprocess.

        Returns:
            An object that can be accepted by `apply`.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def length_extraction_fn(cls):
        """The function to compute the length of the object in this partition.

        Returns:
            A callable function.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def width_extraction_fn(cls):
        """The function to compute the width of the object in this partition.

        Returns:
            A callable function.
        """
        raise NotImplementedError("Must be implemented in child class")

    _length_cache = None
    _width_cache = None

    def length(self):
        if self._length_cache is None:
            cls = type(self)
            func = cls.length_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)

            self._length_cache = self.apply(preprocessed_func)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            cls = type(self)
            func = cls.width_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)

            self._width_cache = self.apply(preprocessed_func)
        return self._width_cache

    @classmethod
    def empty(cls):
        raise NotImplementedError("To be implemented in the child class!")


class RayRemotePartition(RemotePartition):
    def __init__(self, object_id):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.call_queue = []

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

        return RayRemotePartition(oid)

    def add_to_apply_calls(self, func, **kwargs):
        self.call_queue.append((func, kwargs))
        return self

    def __copy__(self):
        return RayRemotePartition(object_id=self.oid)

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
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
        return RayRemotePartition(ray.put(obj))

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


def length_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series))
    return len(df)


def width_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series))
    if isinstance(df, pandas.DataFrame):
        return len(df.columns)
    else:
        return 1


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
        return func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate
    # on it. We don't want the error to propagate to the user, and we want to
    # avoid copying unless we absolutely have to.
    except ValueError:
        return func(partition.copy(), **kwargs)
