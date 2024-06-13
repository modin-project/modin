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

import numpy
from numpy import e, euler_gamma, inf, nan, newaxis, pi
from packaging import version

if version.parse(numpy.__version__) < version.parse("2.0.0b1"):
    from numpy import NAN, NINF, NZERO, PINF, PZERO, Inf, Infinity, NaN, infty

__all__ = [
    "e",
    "euler_gamma",
    "inf",
    "nan",
    "newaxis",
    "pi",
]

if version.parse(numpy.__version__) < version.parse("2.0.0b1"):
    __all__ += [
        "Inf",
        "Infinity",
        "NAN",
        "NINF",
        "NZERO",
        "NaN",
        "PINF",
        "PZERO",
        "infty",
    ]
