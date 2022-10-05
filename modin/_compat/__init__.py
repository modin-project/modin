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

pandas_version = version.parse(pandas.__version__)


class PandasCompatVersion:
    """Enum-like class describing if we're working in "pandas compat" or normal mode."""

    PY36 = "py36-compat"
    LATEST = "latest"

    if version.parse("1.1.0") <= pandas_version <= version.parse("1.1.5"):
        CURRENT = PY36
    elif version.parse("1.5.0") <= pandas_version < version.parse("1.6"):
        CURRENT = LATEST
    else:
        raise ImportError(f"Unsupported pandas version: {pandas.__version__}")


__all__ = ["PandasCompatVersion"]
