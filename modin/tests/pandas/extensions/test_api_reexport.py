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


import pandas

import modin.pandas as pd


def test_extensions_does_not_overwrite_pandas_api():
    # Ensure that importing modin.pandas.api.extensions does not overwrite our re-export
    # of pandas.api submodules.
    import modin.pandas.api.extensions as ext

    # Top-level submodules should remain the same
    assert set(pd.api.__all__) == set(pandas.api.__all__)
    # Methods we define, like ext.register_dataframe_accessor should be different
    assert (
        ext.register_dataframe_accessor
        is not pandas.api.extensions.register_dataframe_accessor
    )
    # Methods from other submodules, like pd.api.types.is_bool_dtype, should be the same
    assert pd.api.types.is_bool_dtype is pandas.api.types.is_bool_dtype
