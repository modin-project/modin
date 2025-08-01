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

# While other modin backends raise a warning when defaulting to pandas, it does not make sense to
# do so when we're running on the native pandas backend already. These tests ensure such warnings
# are not raised with the pandas backend.

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import Backend
from modin.tests.pandas.utils import df_equals

pytestmark = [
    pytest.mark.skipif(
        Backend.get() != "Pandas",
        reason="warnings only suppressed on native pandas backend",
        allow_module_level=True,
    ),
    # Error if a default to pandas warning is detected.
    pytest.mark.filterwarnings("error:is not supported by NativeOnNative:UserWarning"),
]


def test_crosstab_no_warning():
    # Example from pandas docs
    # https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
    a = np.array(
        ["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"],
        dtype=object,
    )
    b = np.array(
        ["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"],
        dtype=object,
    )
    c = np.array(
        [
            "dull",
            "dull",
            "shiny",
            "dull",
            "dull",
            "shiny",
            "shiny",
            "dull",
            "shiny",
            "shiny",
            "shiny",
        ],
        dtype=object,
    )
    df_equals(
        pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"]),
        pandas.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"]),
    )


def test_json_normalize_no_warning():
    # Example from pandas docs
    # https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html
    data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}},
        {"id": 2, "name": "Faye Raker"},
    ]
    df_equals(pd.json_normalize(data), pandas.json_normalize(data))
