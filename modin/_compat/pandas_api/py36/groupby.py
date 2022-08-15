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

"""Module for 'Python 3.6 pandas' compatibility layer for GroupBy."""

import pandas.core.groupby.generic
import pandas.core.groupby.groupby

from ..abc.groupby import BaseCompatibleSeriesGroupBy, BaseCompatibleDataFrameGroupBy
from modin.utils import _inherit_docstrings


@_inherit_docstrings(pandas.core.groupby.groupby.GroupBy)
class Python36CompatibleDataFrameGroupBy(BaseCompatibleDataFrameGroupBy):
    """Compatibility layer for 'Python 3.6 pandas' for DataFrameGroupBy."""

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, axis=0):
        return self._pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq, axis=axis
        )


@_inherit_docstrings(pandas.core.groupby.generic.SeriesGroupBy)
class Python36CompatibleSeriesGroupBy(BaseCompatibleSeriesGroupBy):
    """Compatibility layer for 'Python 3.6 pandas' for SeriesGroupBy."""

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None):
        return self._pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq
        )
