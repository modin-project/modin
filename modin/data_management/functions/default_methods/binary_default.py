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

from .any_default import AnyDefault

import pandas
from pandas.core.dtypes.common import is_list_like


class BinaryDefault(AnyDefault):
    @classmethod
    def build_default_to_pandas(cls, fn, fn_name):
        def bin_ops_wrapper(df, other, *args, **kwargs):
            squeeze_other = kwargs.pop("broadcast", False) or kwargs.pop(
                "squeeze_other", False
            )
            squeeze_self = kwargs.pop("squeeze_self", False)

            if squeeze_other:
                other = other.squeeze(axis=1)

            if squeeze_self:
                df = df.squeeze(axis=1)

            result = fn(df, other, *args, **kwargs)
            if (
                not isinstance(result, pandas.Series)
                and not isinstance(result, pandas.DataFrame)
                and is_list_like(result)
            ):
                result = pandas.DataFrame(result)
            return result

        return super().build_default_to_pandas(bin_ops_wrapper, fn_name)
