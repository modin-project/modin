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

from types import ModuleType

import pandas as pd
from torch.utils.data import RandomSampler

from modin import pandas as mpd
from modin.experimental.torch.datasets import ModinDataLoader


def _load_test_dataframe(pandas: ModuleType):
    df = pandas.read_csv(
        "https://raw.githubusercontent.com/ponder-org/ponder-datasets/main/USA_Housing.csv"
    )
    return df


def _test_torch_dataloader(pandas: ModuleType):
    df = _load_test_dataframe(pandas)
    loader = ModinDataLoader(
        df,
        batch_size=16,
        features=[
            "AVG_AREA_INCOME",
            "AVG_AREA_HOUSE_AGE",
            "AVG_AREA_NUM_ROOMS",
            "AVG_AREA_NUM_BEDROOMS",
            "POPULATION",
            "PRICE",
        ],
    )
    for batch in loader:
        assert batch.shape[0] <= 16, batch.shape
        assert batch.shape[1] == 6, batch.shape


def test_torch_dataloader():
    _test_torch_dataloader(pd)
    _test_torch_dataloader(mpd)


def test_random():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ponder-org/ponder-datasets/main/USA_Housing.csv"
    )
    loader = ModinDataLoader(
        df,
        batch_size=16,
        features=[
            "AVG_AREA_INCOME",
            "AVG_AREA_HOUSE_AGE",
            "AVG_AREA_NUM_ROOMS",
            "AVG_AREA_NUM_BEDROOMS",
            "POPULATION",
            "PRICE",
        ],
        sampler=RandomSampler,
    )
    for batch in loader:
        assert batch.shape[0] <= 16, batch.shape
        assert batch.shape[1] == 6, batch.shape
