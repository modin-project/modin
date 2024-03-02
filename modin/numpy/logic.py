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

from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings

from .arr import array
from .utils import try_convert_from_interoperable_type


def _dispatch_logic(operator_name):
    @_inherit_docstrings(getattr(numpy, operator_name))
    def call(x, *args, **kwargs):
        x = try_convert_from_interoperable_type(x)
        if not isinstance(x, array):
            ErrorMessage.bad_type_for_numpy_op(operator_name, type(x))
            return getattr(numpy, operator_name)(x, *args, **kwargs)
        return getattr(x, f"_{operator_name}")(*args, **kwargs)

    return call


all = _dispatch_logic("all")
any = _dispatch_logic("any")
isfinite = _dispatch_logic("isfinite")
isinf = _dispatch_logic("isinf")
isnan = _dispatch_logic("isnan")
isnat = _dispatch_logic("isnat")
isneginf = _dispatch_logic("isneginf")
isposinf = _dispatch_logic("isposinf")
iscomplex = _dispatch_logic("iscomplex")
isreal = _dispatch_logic("isreal")


def isscalar(e):
    if isinstance(e, array):
        return False
    return numpy.isscalar(e)


logical_not = _dispatch_logic("logical_not")
logical_and = _dispatch_logic("logical_and")
logical_or = _dispatch_logic("logical_or")
logical_xor = _dispatch_logic("logical_xor")
greater = _dispatch_logic("greater")
greater_equal = _dispatch_logic("greater_equal")
less = _dispatch_logic("less")
less_equal = _dispatch_logic("less_equal")
equal = _dispatch_logic("equal")
not_equal = _dispatch_logic("not_equal")
