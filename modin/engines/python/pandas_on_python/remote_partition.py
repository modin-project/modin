import pandas

from modin.data_management.utils import length_fn_pandas, width_fn_pandas


class PandasOnPythonRemotePartition(object):
    """This abstract class holds the data and metadata for a single partition.
        The methods required for implementing this abstract class are listed in
        the section immediately following this.

        The API exposed by the children of this object is used in
        `BaseBlockPartitions`.

        Note: These objects are treated as immutable by `BaseBlockPartitions`
        subclasses. There is no logic for updating inplace.
    """

    def __init__(self, data):
        self.data = data
        self.call_queue = []

    def get(self):
        """Flushes the call_queue and returns the data.

        Note: Since this object is a simple wrapper, just return the data.

        Returns:
            The object that was `put`.
        """
        if self.call_queue:
            return self.apply(lambda df: df).data
        else:
            return self.data.copy()

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
        self.call_queue.append((func, kwargs))

        def call_queue_closure(data, call_queues):
            result = data.copy()
            for func, kwargs in call_queues:
                try:
                    result = func(result, **kwargs)
                except Exception as e:
                    self.call_queue = []
                    raise e
            return result

        new_data = call_queue_closure(self.data, self.call_queue)
        self.call_queue = []
        return PandasOnPythonRemotePartition(new_data)

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return
        """
        self.call_queue.append((func, kwargs))
        return self

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

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        return cls(obj)

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
            self._length_cache = type(self).length_extraction_fn()(self.data)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            self._width_cache = type(self).width_extraction_fn()(self.data)
        return self._width_cache

    @classmethod
    def empty(cls):
        return cls(pandas.DataFrame())
