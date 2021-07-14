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

"""The module contains the functionality that is used when benchmarking Modin commits on OmniSci backend."""

RAND_LOW = 0
RAND_HIGH = 1_000_000_000

BINARY_OP_DATA_SIZE = {
    "big": [
        ((500_000, 20), (1_000_000, 10)),
    ],
    "small": [
        ((10_000, 20), (25_000, 10)),
    ],
}

UNARY_OP_DATA_SIZE = {
    "big": [
        (1_000_000, 10),
    ],
    "small": [
        (10_000, 10),
    ],
}

SERIES_DATA_SIZE = {
    "big": [
        (10_000_000, 1),
    ],
    "small": [
        (100_000, 1),
    ],
}
