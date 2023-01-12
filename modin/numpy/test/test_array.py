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
import pytest
import modin.numpy as np

@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_repr(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    assert repr(modin_arr) == repr(numpy_arr)

@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_shape(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.shape == numpy_arr.shape

@pytest.mark.parametrize("operand1shape", [100, (3, 100)])
@pytest.mark.parametrize("operand2shape", [100, (3, 100)])
@pytest.mark.parametrize("operator", ["__add__", "__sub__", "__truediv__", "__mul__"])
def test_basic_arithmetic_with_broadcast(operand1shape, operand2shape, operator):
    """Test of operators that support broadcasting."""
    operand1 = numpy.random.randint(-100, 100, size=operand1shape)
    operand2 = numpy.random.randint(-100, 100, size=operand2shape)
    modin_result = np.array(operand1).__getattribute__(operator)(np.array(operand2))
    numpy_result = operand1.__getattribute__(operator)(operand2)
    if operator != "__truediv__":
        numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result, err_msg=f"Binary Op {operator} failed.")
    else:
        # Truediv can have precision issues.
        numpy.testing.assert_array_almost_equal(modin_result._to_numpy(), numpy_result, decimal=12, err_msg="Binary Op __truediv__ failed.")

@pytest.mark.parametrize("operator", ["__pow__", "__floordiv__", "__mod__"])
def test_complex_arithmetic(operator):
    """Test of operators that do not yet support broadcasting"""
    operand1 = numpy.random.randint(-100, 100, size=100)
    lower_bound = -100 if operator != "__pow__" else 0
    operand2 = numpy.random.randint(lower_bound, 100, size=100)
    modin_result = np.array(operand1).__getattribute__(operator)(np.array(operand2))
    numpy_result = operand1.__getattribute__(operator)(operand2)
    numpy.testing.assert_array_almost_equal(modin_result._to_numpy(), numpy_result, decimal=12, err_msg=f"Binary Op {operator} failed on 1D arrays.")

    operand1 = numpy.random.randint(-100, 100, size=(10, 10))
    lower_bound = -100 if operator != "__pow__" else 0
    operand2 = numpy.random.randint(lower_bound, 100, size=(10, 10))
    modin_result = np.array(operand1).__getattribute__(operator)(np.array(operand2))
    numpy_result = operand1.__getattribute__(operator)(operand2)
    numpy.testing.assert_array_almost_equal(modin_result._to_numpy(), numpy_result, decimal=12, err_msg=f"Binary Op {operator} failed on 2D arrays.")
