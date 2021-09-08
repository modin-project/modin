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

"""Function module provides templates for a query compiler default-to-pandas methods."""

from modin.core.dataframe.algebra.d2p.dataframe_default import DataFrameDefault
from modin.core.dataframe.algebra.d2p.datetime_default import DateTimeDefault
from modin.core.dataframe.algebra.d2p.series_default import SeriesDefault
from modin.core.dataframe.algebra.d2p.str_default import StrDefault
from modin.core.dataframe.algebra.d2p.binary_default import BinaryDefault
from modin.core.dataframe.algebra.d2p.any_default import AnyDefault
from modin.core.dataframe.algebra.d2p.resample_default import ResampleDefault
from modin.core.dataframe.algebra.d2p.rolling_default import RollingDefault
from modin.core.dataframe.algebra.d2p.default import DefaultMethod
from modin.core.dataframe.algebra.d2p.cat_default import CatDefault
from modin.core.dataframe.algebra.d2p.groupby_default import GroupByDefault

__all__ = [
    "DataFrameDefault",
    "DateTimeDefault",
    "SeriesDefault",
    "StrDefault",
    "BinaryDefault",
    "AnyDefault",
    "ResampleDefault",
    "RollingDefault",
    "DefaultMethod",
    "CatDefault",
    "GroupByDefault",
]
