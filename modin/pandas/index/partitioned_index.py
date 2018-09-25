from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PartitionedIndex(object):

    _index_lengths_cache = None

    def _get_partition_lengths(self):
        if self._index_lengths_cache is None:
            self._index_lengths_cache = [
                obj.apply(len).get() for obj in self.index_partitions[:0]
            ]
        return self._index_lengths_cache

    def _set_partition_lengths(self, new_value):
        self._partition_length_cache = new_value

    index_lengths = property(_get_partition_lengths, _set_partition_lengths)

    def __getitem__(self, key):
        cls = type(self)
        return cls(self.index_partitions[key])


class RayPartitionedIndex(PartitionedIndex):
    def __init__(self, index_partitions):
        self.index_partitions = index_partitions
