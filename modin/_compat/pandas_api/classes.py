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

"""Compatibility layer picking base compat class depending on pandas version."""

from modin._compat import PandasCompatVersion

if PandasCompatVersion.CURRENT == PandasCompatVersion.PY36:
    from .py36 import (
        Python36CompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .py36 import Python36CompatibleDataFrame as DataFrameCompat
    from .py36 import Python36CompatibleSeries as SeriesCompat
    from .py36 import Python36CompatibleDataFrameGroupBy as DataFrameGroupByCompat
    from .py36 import Python36CompatibleSeriesGroupBy as SeriesGroupByCompat
    from .py36 import Python36CompatibleWindow as WindowCompat
    from .py36 import Python36CompatibleRolling as RollingCompat
    from .py36 import Python36CompatibleResampler as ResamplerCompat
elif PandasCompatVersion.CURRENT == PandasCompatVersion.LATEST:
    from .latest import (
        LatestCompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .latest import LatestCompatibleDataFrame as DataFrameCompat
    from .latest import LatestCompatibleSeries as SeriesCompat
    from .latest import LatestCompatibleDataFrameGroupBy as DataFrameGroupByCompat
    from .latest import LatestCompatibleSeriesGroupBy as SeriesGroupByCompat
    from .latest import LatestCompatibleWindow as WindowCompat
    from .latest import LatestCompatibleRolling as RollingCompat
    from .latest import LatestCompatibleResampler as ResamplerCompat

__all__ = [
    "BasePandasDatasetCompat",
    "DataFrameCompat",
    "SeriesCompat",
    "DataFrameGroupByCompat",
    "SeriesGroupByCompat",
    "WindowCompat",
    "RollingCompat",
    "ResamplerCompat",
]
