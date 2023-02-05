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

"""Module houses array shaping methods for Modin's NumPy API."""
from modin.error_message import ErrorMessage


def ravel(a, order="C"):
    if order != "C":
        ErrorMessage.single_warning(
            "Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order."
        )
    if hasattr(a, "flatten"):
        return a.flatten(order)
    raise NotImplementedError(
        f"Object of type {type(a)} does not have a flatten method to use for raveling."
    )


def shape(a):
    if hasattr(a, "shape"):
        return a.shape
    raise NotImplementedError(
        f"Object of type {type(a)} does not have a shape property."
    )


def transpose(a, axes=None):
    if axes is not None:
        raise NotImplementedError(
            "Modin does not support arrays higher than 2-dimensions. Please use `transpose` with `axis=None` on a 2-dimensional or lower object."
        )
    if hasattr(a, "transpose"):
        return a.transpose()
    raise NotImplementedError(
        f"Object of type {type(a)} does not have a transpose method."
    )
