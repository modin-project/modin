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
