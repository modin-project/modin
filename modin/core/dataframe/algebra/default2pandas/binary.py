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

"""Module houses default binary functions builder class."""

import pandas
from pandas.core.dtypes.common import is_list_like

from .default import DefaultMethod


class BinaryDefault(DefaultMethod):
    """Build default-to-pandas methods which executes binary functions."""

    @classmethod
    def build_default_to_pandas(cls, fn, fn_name):
        """
        Build function that do fallback to pandas for passed binary `fn`.

        Parameters
        ----------
        fn : callable
            Binary function to apply to the casted to pandas frame and other operand.
        fn_name : str
            Function name which will be shown in default-to-pandas warning message.

        Returns
        -------
        callable
            Function that takes query compiler, does fallback to pandas and applies binary `fn`
            to the casted to pandas frame.
        """

        def bin_ops_wrapper(df, other, *args, **kwargs):
            """Apply specified binary function to the passed operands."""
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
