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

"""Implement Window public API."""

import pandas
import pandas.core.window.rolling
import pandas.core.resample
import pandas.core.generic
from modin.utils import _inherit_docstrings

# Similar to pandas, sentinel value to use as kwarg in place of None when None has
# special meaning and needs to be distinguished from a user explicitly passing None.
sentinel = object()

# Do not lookup certain attributes in columns or index, as they're used for some
# special purposes, like serving remote context
_ATTRS_NO_LOOKUP = {"____id_pack__", "__name__"}


@_inherit_docstrings(pandas.core.window.rolling.Window)
class Window(object):
    def __init__(
        self,
        dataframe,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        method="single",
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.window_args = [
            window,
            min_periods,
            center,
            win_type,
            on,
            axis,
            closed,
            method,
        ]

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_mean(
                self.window_args, *args, **kwargs
            )
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_sum(
                self.window_args, *args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_var(
                self.window_args, ddof, *args, **kwargs
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_std(
                self.window_args, ddof, *args, **kwargs
            )
        )
