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
from packaging import version

from . import linalg
from .arr import array
from .array_creation import ones_like, tri, zeros_like
from .array_shaping import append, hstack, ravel, shape, split, transpose
from .constants import e, euler_gamma, inf, nan, newaxis, pi

if version.parse(numpy.__version__) < version.parse("2.0.0b1"):
    from .constants import (
        NAN,
        NINF,
        NZERO,
        PINF,
        PZERO,
        Inf,
        Infinity,
        NaN,
        infty,
    )

from .logic import (
    all,
    any,
    equal,
    greater,
    greater_equal,
    iscomplex,
    isfinite,
    isinf,
    isnan,
    isnat,
    isneginf,
    isposinf,
    isreal,
    isscalar,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)
from .math import (
    abs,
    absolute,
    add,
    amax,
    amin,
    argmax,
    argmin,
    divide,
    dot,
    exp,
    float_power,
    floor_divide,
    max,
    maximum,
    mean,
    min,
    minimum,
    mod,
    multiply,
    power,
    prod,
    remainder,
    sqrt,
    subtract,
    sum,
    true_divide,
    var,
)
from .trigonometry import tanh


def where(condition, x=None, y=None):
    if condition is True:
        return x
    if condition is False:
        return y
    if hasattr(condition, "where"):
        return condition.where(x=x, y=y)
    raise NotImplementedError(
        f"np.where for condition of type {type(condition)} is not yet supported in Modin."
    )


__all__ = [  # noqa: F405
    "linalg",
    "array",
    "zeros_like",
    "ones_like",
    "ravel",
    "shape",
    "transpose",
    "all",
    "any",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "isneginf",
    "isposinf",
    "iscomplex",
    "isreal",
    "isscalar",
    "logical_not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
    "absolute",
    "abs",
    "add",
    "divide",
    "dot",
    "float_power",
    "floor_divide",
    "power",
    "prod",
    "multiply",
    "remainder",
    "mod",
    "subtract",
    "sum",
    "true_divide",
    "mean",
    "maximum",
    "amax",
    "max",
    "minimum",
    "amin",
    "min",
    "where",
    "e",
    "euler_gamma",
    "inf",
    "nan",
    "newaxis",
    "pi",
    "sqrt",
    "tanh",
    "exp",
    "argmax",
    "argmin",
    "var",
    "split",
    "hstack",
    "append",
    "tri",
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
