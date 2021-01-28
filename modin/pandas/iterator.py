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

"""Place to define the Modin iterator."""

from collections.abc import Iterator


class PartitionIterator(Iterator):
    """Iterator on partitioned data."""

    def __init__(self, df, axis, func):
        """
        Construct a iterator on partitioned data.

        TODO: add types.

        Parameters
        ----------
        df: DataFrame
            The dataframe to iterate over.
        axis:
            axis to iterate over.
        func:
            The function to get inner iterables from each partition.
        """
        self.df = df
        self.axis = axis
        self.index_iter = (
            zip(
                iter(slice(None) for _ in range(len(self.df.columns))),
                range(len(self.df.columns)),
            )
            if axis
            else zip(
                range(len(self.df.index)),
                iter(slice(None) for _ in range(len(self.df.index))),
            )
        )
        self.func = func

    def __iter__(self):
        """Implement iterator interface."""
        return self

    def __next__(self):
        """Implement iterator interface."""
        key = next(self.index_iter)
        df = self.df.iloc[key]
        return self.func(df)
