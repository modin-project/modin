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

import numpy

from modin.error_message import ErrorMessage
from .arr import array


def ravel(a, order="C"):
    if not isinstance(a, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for ravel, not {type(a)}. Defaulting to NumPy."
        )
        return numpy.ravel(a, order=order)
    if order != "C":
        ErrorMessage.single_warning(
            "Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order."
        )
    return a.flatten(order)


def shape(a):
    if not isinstance(a, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for shape, not {type(a)}. Defaulting to NumPy."
        )
        return numpy.shape(a)
    return a.shape


def transpose(a, axes=None):
    if not isinstance(a, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for transpose, not {type(a)}. Defaulting to NumPy."
        )
        return numpy.transpose(a, axes=axes)
    if axes is not None:
        raise NotImplementedError(
            "Modin does not support arrays higher than 2-dimensions. Please use `transpose` with `axis=None` on a 2-dimensional or lower object."
        )
    return a.transpose()
