from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Iterator


class PartitionIterator(Iterator):
    def __init__(self, data_manager, axis, func):
        """PartitionIterator class to define a generator on partitioned data

        Args:
            data_manager (DataManager): Data manager for the dataframe
            axis (int): axis to iterate over
            func (callable): The function to get inner iterables from
                each partition
        """
        self.data_manager = data_manager
        self.axis = axis
        self.index_iter = (
            iter(self.data_manager.columns)
            if axis
            else iter(range(len(self.data_manager.index)))
        )
        self.func = func

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        key = next(self.index_iter)
        if self.axis:
            df = self.data_manager.getitem_column_array([key]).to_pandas()
        else:
            df = self.data_manager.getitem_row_array([key]).to_pandas()
        return next(self.func(df))
