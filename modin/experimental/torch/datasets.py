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

from typing import Hashable, NoReturn

import numpy as np
from pandas import DataFrame, Series
from torch.utils import data as data_utils
from torch.utils.data import DataLoader, IterableDataset

from modin.pandas import DataFrame as ModinDataFrame
from modin.pandas import Series as ModinSeries


class ModinIterableDataset(IterableDataset):
    """
    PandasIterableDataset is responsible for converting a pandas/modin dataframe into a torch-compatible dataset.
    """

    def __init__(
        self,
        df: DataFrame | ModinDataFrame,
        with_index: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        df : DataFrame | ModinDataFrame
            DataFrame-like object from which the iterable will draw data.

        with_index : bool, default: False
            If true, include the index object with the resulting iterable, similar to `DataFrame.iterrows`.
        """

        super().__init__()
        self._df = df
        self._with_index = with_index

    def __iter__(self):
        worker_info = data_utils.get_worker_info()

        # FIXME: Multi-threaded IterableDataset would generate duplicate samples.
        # See: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        #
        # However, modin's iterrows doesn't have convenient row-skipping functions.
        # And random access (un-batched) is quite slow. Don't know how to fix.
        #
        # This means it's running on a single thread.
        if worker_info is None:
            yield from self._iter()
        else:
            yield from self._multithread_iter()

    def _iter(self):
        """
        Single threaded iterator.
        """
        for idx, row in self._df.iterrows():
            np_row = np.array(row)

            if self._with_index:
                yield idx, np_row
            else:
                yield np_row

    def _multithread_iter(self) -> NoReturn:
        raise NotImplementedError(
            "Using `ModinIterableDataset` in a multi-process context is not supported yet."
        )


PandasIterableDataset = ModinIterableDataset


def to_dataloader(
    df: DataFrame | ModinDataFrame, batch_size: int = 1, with_index: bool = False
) -> DataLoader:
    """
    Converts a Pandas/Modin DataFrame into a torch DataLoader.

    NOTE: This function should eventually go into modin/utils.py.

    Parameters
    ----------
    batch_size : int, default: 1
        Batch size that dataloader uses.

    with_index : bool, default: False
        If true, include the index object with the resulting iterable, similar to `DataFrame.iterrows`.

    Returns
    -------
    DataLoader
        DataLoader object backed by desired data.
    """

    dataset = ModinIterableDataset(df=df, with_index=with_index)
    return DataLoader(dataset, batch_size=batch_size)
