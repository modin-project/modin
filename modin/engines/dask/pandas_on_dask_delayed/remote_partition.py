import pandas

from modin.engines.base.remote_partition import BaseRemotePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas


class DaskRemotePartition(BaseRemotePartition):
    def __init__(self, dask_obj, func=None):
        import dask

        self.dask_obj = dask_obj
        self.delayed_call = (
            dask_obj if func is None else dask.delayed(func[0])(dask_obj, **func[1])
        )

    def get(self):
        """Return the object wrapped by this one to the original format.

        Note: This is the opposite of the classmethod `put`.
            E.g. if you assign `x = BaseRemotePartition.put(1)`, `x.get()` should
            always return 1.

        Returns:
            The object that was `put`.
        """
        self.delayed_call = self.dask_obj
        return self.delayed_call.compute()

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args:
            func: The lambda to apply (may already be correctly formatted)

        Returns:
             A new `BaseRemotePartition` containing the object that has had `func`
             applied to it.
        """
        import dask

        # applies the func lazily
        delayed_call = self.delayed_call
        self.delayed_call = self.dask_obj
        return self.__class__(dask.delayed(func)(delayed_call, **kwargs))

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return
        """
        import dask

        self.delayed_call = dask.delayed(func)(self.delayed_call, **kwargs)
        return self

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Assumes the underlying object is a Pandas DataFrame and simply calls `get`

        Returns:
            A Pandas DataFrame.
        """
        return self.get()

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        import dask

        # simply wrap the input object by dask.delayed
        return cls(dask.delayed(obj))

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `BaseRemotePartition`
            objects.

        Args:
            func: The function to preprocess.

        Returns:
            An object that can be accepted by `apply`.
        """
        # seems that dask does not need any pre-processing
        return func

    @classmethod
    def length_extraction_fn(cls):
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        return width_fn_pandas

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())
