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

"""Collection of array utility functions for internal use."""

import modin.pandas as pd
import modin.numpy as np

_INTEROPERABLE_TYPES = (pd.DataFrame, pd.Series)


def try_convert_from_interoperable_type(obj, copy=False):
    if isinstance(obj, _INTEROPERABLE_TYPES):
        obj = np.array(obj, copy=copy)
    return obj
