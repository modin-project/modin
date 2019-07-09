import pandas
import numpy as np

from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.base.frame.partition import BaseFramePartition


class PandasOnPythonFramePartition(BaseFramePartition):
    """This abstract class holds the data and metadata for a single partition.
        The methods required for implementing this abstract class are listed in
        the section immediately following this.

        The API exposed by the children of this object is used in
        `BaseFrameManager`.

        Note: These objects are treated as immutable by `BaseFrameManager`
        subclasses. There is no logic for updating inplace.
    """

    def __init__(self, data, length=None, width=None, call_queue=[]):
        self.data = data
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Flushes the call_queue and returns the data.

        Note: Since this object is a simple wrapper, just return the data.

        Returns:
            The object that was `put`.
        """
        self.drain_call_queue()
        return self.data.copy()

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

        def call_queue_closure(data, call_queues):
            result = data.copy()
            for func, kwargs in call_queues:
                try:
                    result = func(result, **kwargs)
                except Exception as e:
                    self.call_queue = []
                    raise e
            return result

        self.data = call_queue_closure(self.data, self.call_queue)
        self.call_queue = []
        return PandasOnPythonFramePartition(func(self.data.copy(), **kwargs))

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnPythonFramePartition(
            self.data.copy(), call_queue=self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        self.apply(lambda x: x)

    def mask(self, row_indices=None, col_indices=None):
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        new_obj._length_cache, new_obj._width_cache = len(row_indices), len(col_indices)
        return new_obj

    def split(self, axis, is_transposed, splits):
        """Split partition along axis given list of resulting index groups (splits).

        Args:
            axis: Axis to split along.
            is_transposed: True if the partition needs to be transposed first.
            splits: List of list of indexes to group together.

        Returns:
            Returns PandasOnRayFramePartitions for each of the resulting splits.
        """
        if (
            len(splits) == 1
            and len(splits[0]) == (self.width() if axis else self.length())
            and all(i == splits[0][i] for i in range(len(splits[0])))
        ):
            if is_transposed:
                call_queue = self.call_queue + [(pandas.DataFrame.transpose, {})]
            else:
                call_queue = self.call_queue
            return [
                PandasOnPythonFramePartition(
                    self.data, self._length_cache, self._width_cache, call_queue
                )
            ]
        else:
            partition = self.data
            call_queue = self.call_queue
            # Drain call queue.
            if len(call_queue) > 0:
                for func, kwargs in call_queue:
                    try:
                        partition = func(partition, **kwargs)
                    except ValueError:
                        partition = func(partition.copy(), **kwargs)

            # Orient and cut up partition.
            part = partition.T if is_transposed else partition
            new_parts = [
                part.iloc[:, index] if axis else part.iloc[index] for index in splits
            ]

            # Update self after draining call queue
            self.data = partition
            self.call_queue = []
            self._length_cache = len(partition) if hasattr(partition, "__len__") else 0
            self._width_cache = (
                len(partition.columns) if hasattr(partition, "columns") else 0
            )

            return [PandasOnPythonFramePartition(new_part) for new_part in new_parts]

    @classmethod
    def shuffle(cls, axis, func, part_length, part_width, *partitions, **kwargs):
        """Takes the partitions combines them based on the indices.

        Args:
            axis: Axis to combine the partitions by.
            func: Function to apply after creating the new partition.
            transposed: True if we need to transpose the partitions before combining.
            part_length: Length of the resulting partition.
            part_width: Width of the resulting partition.
            indices: Indices of the paritions to combine.
            *partitions: List of partitions to combine.

        Returns:
            A `BaseFramePartition` object.
        """
        if len(partitions) == 0:
            return pandas.DataFrame()

        fill_value = kwargs.pop("_fill_value", np.NaN)
        call_queues = []
        part_data = []
        for part in partitions:
            if isinstance(part, PandasOnPythonFramePartition):
                part_data.append(part.data)
                call_queues.append(part.call_queue)
            else:
                part_data.append(part)
                call_queues.append(None)

        # Create partition from partitions.
        df_parts = []
        for i in range(len(partitions)):
            call_queue = call_queues[i]
            partition = part_data[i]
            if isinstance(partition, int):
                nan_len = partition
                length = part_length if axis else part_width
                df_part = pandas.DataFrame(
                    np.repeat(fill_value, nan_len * length).reshape(
                        (length, nan_len) if axis else (nan_len, length)
                    )
                )
            else:
                # Drain call queue.
                if len(call_queue) > 0:
                    for func, kwargs in call_queue:
                        try:
                            partition = func(partition, **kwargs)
                        except ValueError:
                            partition = func(partition.copy(), **kwargs)
                df_part = partition

            # Reset index and columns for consistent concat behavior.
            if axis:
                df_part.index = pandas.RangeIndex(len(df_part))
            else:
                df_part.columns = pandas.RangeIndex(len(df_part.columns))
            df_parts.append(df_part)
        df = pandas.concat(df_parts, axis=axis)

        # Make sure internal indices are correct.
        if axis:
            df.columns = pandas.RangeIndex(len(df.columns))
        else:
            df.index = pandas.RangeIndex(len(df))

        # Apply post-shuffle function.
        if func is not None:
            result = func(df, **kwargs)
        else:
            result = df

        return PandasOnPythonFramePartition(result)

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
        """Convert the object stored in this partition to a NumPy Array.

        Returns:
            A NumPy Array.
        """
        return self.apply(lambda df: df.values).get()

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
            self._length_cache = type(self).length_extraction_fn()(self.data)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            self._width_cache = type(self).width_extraction_fn()(self.data)
        return self._width_cache

    @classmethod
    def empty(cls):
        return cls(pandas.DataFrame())
