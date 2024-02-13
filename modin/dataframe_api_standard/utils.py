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


# MIT License

# Copyright (c) 2023, Marco Gorelli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Any


# Technically, it would be possible to correctly type hint this function
# with a tonne of overloads, but for now, it' not worth it, just use Any
def validate_comparand(left: Any, right: Any) -> Any:
    """Validate comparand, raising if it can't be compared with `left`.

    If `left` and `right` are derived from the same dataframe, then return
    the underlying object of `right`.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):  # pragma: no cover
        # Technically, currently unreachable - but, keeping this in case it
        # becomes reachable in the future.
        msg = "Cannot compare different dataframe objects - please join them first"
        raise ValueError(msg)
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__column_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot compare Column with DataFrame it was not derived from."
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__scalar_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot compare Scalar with DataFrame it was not derived from."
            raise ValueError(msg)
        return right.scalar

    if hasattr(left, "__column_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__column_namespace__") and hasattr(right, "__column_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot compare Columns from different dataframes"
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__column_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot compare Column and Scalar if they don't share the same parent dataframe"
            raise ValueError(msg)
        return right.scalar

    if hasattr(left, "__scalar_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__column_namespace__"):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot combine Scalars from different dataframes"
            raise ValueError(msg)
        return right.scalar

    return right
