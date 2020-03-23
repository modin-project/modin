# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


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
