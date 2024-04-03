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

from collections import defaultdict

import pytest

from modin.config import Parameter, update


def make_prefilled(vartype, varinit):
    class Prefilled(Parameter, type=vartype):
        @classmethod
        def _get_raw_from_config(cls):
            return varinit

    return Prefilled


@pytest.fixture
def prefilled_parameter():
    return make_prefilled(str, "init")


def test_equals(prefilled_parameter):
    assert prefilled_parameter.get() == "Init"

    prefilled_parameter.put("value2")
    assert prefilled_parameter.get() == "Value2"


def test_triggers(prefilled_parameter):
    results = defaultdict(int)
    callbacks = []

    def make_callback(name, res=results):
        def callback(p: Parameter):
            res[name] += 1

        # keep reference to callbacks so they won't be removed by GC
        callbacks.append(callback)
        return callback

    prefilled_parameter.once("init", make_callback("init"))
    assert results["init"] == 1

    prefilled_parameter.once("never", make_callback("never"))
    prefilled_parameter.once("once", make_callback("once"))
    prefilled_parameter.subscribe(make_callback("subscribe"))

    prefilled_parameter.put("multi")
    prefilled_parameter.put("once")
    prefilled_parameter.put("multi")
    prefilled_parameter.put("once")

    expected = [("init", 1), ("never", 0), ("once", 1), ("subscribe", 5)]
    for name, val in expected:
        assert results[name] == val, "{} has wrong count".format(name)


@pytest.mark.parametrize(
    "parameter,good,bad",
    [
        (make_prefilled(bool, "false"), {"1": True, False: False}, ["nope", 2]),
        (make_prefilled(int, "10"), {" 15\t": 15, 25: 25}, ["-10", 1.0, "foo"]),
        (
            make_prefilled(dict, "key = value"),
            {
                "KEY1 = VALUE1, KEY2=VALUE2=VALUE3,KEY3=0": {
                    "KEY1": "VALUE1",
                    "KEY2": "VALUE2=VALUE3",
                    "KEY3": 0,
                },
                "KEY=1": {"KEY": 1},
            },
            ["key1=some,string", "key1=value1,key2=", "random string"],
        ),
    ],
)
def test_validation(parameter, good, bad):
    for inval, outval in good.items():
        parameter.put(inval)
        assert parameter.get() == outval
    for inval in bad:
        with pytest.raises(ValueError):
            parameter.put(inval)


@pytest.mark.parametrize("vartype", [bool, int, dict])
def test_init_validation(vartype):
    parameter = make_prefilled(vartype, "bad value")
    with pytest.raises(ValueError):
        parameter.get()


def test_context_manager_update():
    parameter1 = make_prefilled(vartype=bool, varinit="False")

    # simple case, 1 parameter
    assert parameter1.get() is False
    with update(parameter1, True):
        assert parameter1.get() is True
    assert parameter1.get() is False

    # nested case, 1 parameter
    assert parameter1.get() is False
    with update(parameter1, True):
        assert parameter1.get() is True
        with update(parameter1, False):
            assert parameter1.get() is False
            with update(parameter1, False):
                assert parameter1.get() is False
            assert parameter1.get() is False
        assert parameter1.get() is True
    assert parameter1.get() is False

    parameter2 = make_prefilled(vartype=bool, varinit="True")

    # simple case, 2 parameters
    assert parameter1.get() is False
    assert parameter2.get() is True
    with update({parameter1: True, parameter2: False}):
        assert parameter1.get() is True
        assert parameter2.get() is False
    assert parameter1.get() is False
    assert parameter2.get() is True

    # nested case, 2 parameters
    with update({parameter1: True, parameter2: False}):
        assert parameter1.get() is True
        assert parameter2.get() is False
        with update(parameter1, False):
            assert parameter1.get() is False
            assert parameter2.get() is False
            with update(parameter2, True):
                assert parameter1.get() is False
                assert parameter2.get() is True
                with update({parameter1: True, parameter2: False}):
                    assert parameter1.get() is True
                    assert parameter2.get() is False
                assert parameter1.get() is False
                assert parameter2.get() is True
            assert parameter1.get() is False
            assert parameter2.get() is False
        assert parameter1.get() is True
        assert parameter2.get() is False
    assert parameter1.get() is False
    assert parameter2.get() is True
