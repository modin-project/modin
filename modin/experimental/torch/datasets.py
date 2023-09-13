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
from __future__ import annotations

import math
from typing import Hashable, Sequence, Type

from pandas import DataFrame
from torch.utils.data import Sampler, SequentialSampler

from modin.pandas import DataFrame as ModinDataFrame


class ModinDataLoader:
    "A self explainatory class to convert a DataFrame into a DataLoader that batches rows."

    def __init__(
        self,
        df: DataFrame | ModinDataFrame,
        batch_size: int,
        features: Sequence[Hashable] = (),
        sampler: Type[Sampler] | Sampler = SequentialSampler,
    ) -> None:
        """
        Converts a Pandas/Modin DataFrame into a torch DataLoader.

        NOTE: This function should eventually go into modin/utils.py.

        Parameters
        ----------
        df : DataFrame

        batch_size : int, default: 1

        features : Sequence[Hashable], default: ()
            If specified, only these features will be used.

        sampler: Type[Sampler] | Sampler, default: SequentialSampler
            The sampler to use. By default, iterates over the DataFrame in order.

        Returns
        -------
        DataLoader
            DataLoader object backed by desired data.
        """

        if features:
            df = df[features]

        if isinstance(sampler, type):
            sampler = sampler(df)

        self._df = df
        self._batch_size = batch_size
        self._sampler = sampler

    def __len__(self):
        # Sampler length is always valid.
        return math.ceil(len(self._sampler) / self._batch_size)

    def __iter__(self):
        idx_buffer = []

        for cnt, idx in enumerate(self._sampler):
            idx_buffer.append(idx)

            if self._end_of_batch(cnt):
                yield self._df.iloc[idx_buffer].to_numpy()
                idx_buffer = []

    def _end_of_batch(self, counter: int):
        return (
            counter % self._batch_size == self._batch_size - 1
            or counter == len(self._sampler) - 1
        )
