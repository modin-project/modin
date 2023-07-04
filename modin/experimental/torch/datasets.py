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
from typing import Iterable, Sequence, Hashable

import numpy as np
from pandas import DataFrame

from modin.pandas import DataFrame as ModinDataFrame


class ModinDataLoader:
    "A self explainatory class to convert a DataFrame into a DataLoader that batches rows."

    def __init__(
        self,
        df: DataFrame | ModinDataFrame,
        batch_size: int,
        features: Sequence[Hashable] = (),
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

        Returns
        -------
        DataLoader
            DataLoader object backed by desired data.
        """

        self._batch_size = batch_size

        if features:
            df = df[features]

        self._df = df

    def __len__(self):
        "Batched so the length is reduced."
        return math.ceil(len(self._df) / self._batch_size)

    def __getitem__(self, idx: int):
        "Using iloc to perform batched query."

        idx_start = idx * self._batch_size
        idx_end = min((idx + 1) * self._batch_size, len(self._df))
        return self._df.iloc[idx_start:idx_end].to_numpy()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


PandasDataLoader = ModinDataLoader


def to_dataloader(df: DataFrame | ModinDataFrame, batch_size: int = 1) -> Iterable:
    return ModinDataLoader(df, batch_size=batch_size)
