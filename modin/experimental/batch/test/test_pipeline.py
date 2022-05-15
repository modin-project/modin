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

import pytest
import modin.pandas as pd
import numpy as np
from modin.config import Engine
import warnings

@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="Only Ray supports the Batch Pipeline API",
)
def test_pipeline():
    arr = np.random.randint(0, 1000, (1000, 1000))
    df = pd.DataFrame(arr)
    with pytest.warns(warnings.UserWarning, match="No pipeline exists. Please call `df._build_batch_pipeline` first to create a batch pipeline."):
        df._add_batch_query(lambda df: df)
    with pytest.warns(warnings.UserWarning, match="The Batch Pipeline API is an experimental feature and still under development in Modin.")
        df = df._build_batch_pipeline(lambda df: df, 0)
    query = df._pipeline.nodes_list[0]
    with pytest.warns(warnings.UserWarning, match="Existing pipeline discovered. Please call this function again with `overwrite_existing` set to True to overwrite this pipeline."):
        df = df._build_batch_pipeline(lambda df: df.iloc[0], 0)
    assert df._pipeline.nodes_list[0] == query, "Pipeline was overwritten when `overwrite_existing` was not set to True."
    def add_col(df):
        df['new_col'] = df.sum(axis=1)
        return df
    df = df._build_batch_pipeline(add_col, 1, overwrite_existing=True)
    assert df._pipeline.nodes_list[0] != query, "Pipeline was not overwritten when `overwrite_existing` was set to True."
    df = df._add_batch_query(lambda df: df * -30)
    df = df._add_batch_query(lambda df: df.rename(columns={i:f"col {i}" for i in range(1000)}))
    

    
