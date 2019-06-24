from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Iterator


class PartitionIterator(Iterator):
    def __init__(self, query_compiler, axis, func):
        """PartitionIterator class to define a generator on partitioned data

        Args:
            query_compiler: Data manager for the dataframe
            axis: axis to iterate over
            func: The function to get inner iterables from
                each partition
        """
        self.query_compiler = query_compiler
        self.axis = axis
        self.index_iter = (
            iter(self.query_compiler.columns)
            if axis
            else iter(range(len(self.query_compiler.index)))
        )
        self.func = func

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        key = next(self.index_iter)
        if self.axis:
            df = self.query_compiler.getitem_column_array([key]).to_pandas()
        else:
            df = self.query_compiler.getitem_row_array([key]).to_pandas()
        return next(self.func(df))
