# noqa: MD01
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

# noqa: MD02
"""Function examples for docstring testing."""


class weakdict(dict):  # noqa: GL08
    __slots__ = ("__weakref__",)


def optional_square(number: int = 5) -> int:  # noqa
    """
    Square `number`.

    The function from Modin.

    Parameters
    ----------
    number : int
        Some number.

    Notes
    -----
    The `optional_square` Modin function from modin/scripts/examples.py.
    """
    return number**2


def optional_square_empty_parameters(number: int = 5) -> int:
    """
    Parameters
    ----------
    """
    return number**2


def square_summary(number: int) -> int:  # noqa: PR01, GL08
    """
    Square `number`.

    See https://github.com/ray-project/ray.

    Examples
    --------
    The function that will never be used in modin.pandas.DataFrame same as in
    pandas or NumPy.
    """
    return number**2
