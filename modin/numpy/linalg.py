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

from .arr import array
from .utils import try_convert_from_interoperable_type
from modin.error_message import ErrorMessage


def norm(x, ord=None, axis=None, keepdims=False):
    x = try_convert_from_interoperable_type(x)
    if not isinstance(x, array):
        ErrorMessage.bad_type_for_numpy_op("linalg.norm", type(x))
        return numpy.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return x._norm(ord=ord, axis=axis, keepdims=keepdims)
