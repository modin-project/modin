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

from packaging import version
import pandas

if (
    version.parse("1.1.0")
    <= version.parse(pandas.__version__)
    <= version.parse("1.1.5")
):
    from .py36 import (
        Python36CompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .py36 import Python36CompatibleDataFrame as DataFrameCompat
    from .py36 import Python36CompatibilitySeries as SeriesCompat
elif (
    version.parse("1.4.0")
    <= version.parse(pandas.__version__)
    <= version.parse("1.4.99")
):
    from .latest import (
        LatestCompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .latest import LatestCompatibleDataFrame as DataFrameCompat
    from .latest import LatestCompatibleSeries as SeriesCompat
else:
    raise ImportError(f"Unsupported pandas version: {pandas.__version__}")

__all__ = ["BasePandasDatasetCompat", "DataFrameCompat", "SeriesCompat"]
