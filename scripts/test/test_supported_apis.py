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

import pathlib
import pandas
import numpy as np

from scripts.supported_apis import (
    get_supported_params,
    generate_csv,
)


def test_supported_apis():
    apis = get_supported_params(pathlib.Path("scripts/test/examples.py"))
    generate_csv(pathlib.Path("test.csv"), apis)
    df_apis = pandas.read_csv("test.csv")
    idx = np.where(df_apis["Method Name"] == "supported_apis_method")[0]
    assert len(idx) == 1
    idx = idx[0]
    assert len(df_apis.columns) == 5
    assert df_apis["Notes"][idx] == "test multiline note"
    assert (
        df_apis["PandasOnRay"][idx] == "Partial"
        and df_apis["PandasOnDask"][idx] == "Harmful"
        and df_apis["OmniSci"][idx] == "-"
    )
