import pandas

from modin.engines.base.frame.partition import BaseFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin import __execution_engine__

if __execution_engine__ == "Dask":
    from distributed.client import _get_global_client


def apply_list_of_funcs(funcs, df):
    for func, kwargs in funcs:
        df = func(df, **kwargs)
    return df


class PandasOnDaskFramePartition(BaseFramePartition):
    """This abstract class holds the data and metadata for a single partition.
        The methods required for implementing this abstract class are listed in
        the section immediately following this.

        The API exposed by the children of this object is used in
        `BaseFrameManager`.

        Note: These objects are treated as immutable by `BaseFrameManager`
        subclasses. There is no logic for updating inplace.
    """

    def __init__(self, future, length=None, width=None, call_queue=None):
        self.future = future
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self.client = _get_global_client()

    def get(self):
        """Flushes the call_queue and returns the data.

        Note: Since this object is a simple wrapper, just return the data.

        Returns:
            The object that was `put`.
        """
        self.drain_call_queue()
        # blocking operation
        if isinstance(self.future, pandas.DataFrame):
            return self.future
        return self.future.result()

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args:
            func: The lambda to apply (may already be correctly formatted)

        Returns:
             A new `BaseFramePartition` containing the object that has had `func`
             applied to it.
        """
        call_queue = self.call_queue + [[func, kwargs]]
        future = self.client.submit(
            apply_list_of_funcs, call_queue, self.future, pure=False
        )
        return PandasOnDaskFramePartition(future)

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnDaskFramePartition(
            self.future, call_queue=self.call_queue + [[func, kwargs]]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        self.future = self.apply(lambda x: x).future
        self.call_queue = []

    def mask(self, row_indices, col_indices):
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

    def __copy__(self):
        return PandasOnDaskFramePartition(
            self.future, self._length_cache, self._width_cache
        )

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

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
        return self.apply(lambda df: df.to_numpy()).get()

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        client = _get_global_client()
        return cls(client.scatter(obj, hash=False))

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `BaseFramePartition`
            objects.

        Args:
            func: The function to preprocess.

        Returns:
            An object that can be accepted by `apply`.
        """
        return func

    @classmethod
    def length_extraction_fn(cls):
        """The function to compute the length of the object in this partition.

        Returns:
            A callable function.
        """
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        """The function to compute the width of the object in this partition.

        Returns:
            A callable function.
        """
        return width_fn_pandas

    _length_cache = None
    _width_cache = None

    def length(self):
        if self._length_cache is None:
            self._length_cache = self.apply(lambda df: len(df)).future
        if isinstance(self._length_cache, type(self.future)):
            self._length_cache = self._length_cache.result()
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            self._width_cache = self.apply(lambda df: len(df.columns)).future
        if isinstance(self._width_cache, type(self.future)):
            self._width_cache = self._width_cache.result()
        return self._width_cache

    @classmethod
    def empty(cls):
        return cls(pandas.DataFrame(), 0, 0)
