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

import re
from types import FunctionType

import pandas
import pytest

import modin.pandas as pd
from modin.config import context as config_context
from modin.pandas.api.extensions import register_pd_accessor
from modin.tests.pandas.utils import df_equals, eval_general


@pytest.fixture(
    params=sorted(
        key
        for key, value in pd.__dict__.items()
        if isinstance(value, FunctionType) and value.__module__ == pd.general.__name__
    )
)
def pd_general_function(request):
    return request.param


@pytest.fixture(
    params=sorted(
        key
        for key, value in pd.__dict__.items()
        if isinstance(value, FunctionType) and value.__module__ == pd.io.__name__
    )
)
def pd_io_function(request):
    return request.param


class TestRegisterForAllBackends:
    def test_add_new_function(self):
        expected_string_val = "Some string value"
        method_name = "new_method"

        @register_pd_accessor(method_name)
        def my_method_implementation():
            return expected_string_val

        assert pd.new_method() == expected_string_val

    def test_add_new_non_method(self):
        expected_val = 4
        attribute_name = "four"
        register_pd_accessor(attribute_name)(expected_val)
        assert pd.four == expected_val

    def test_override_io_function(self, pd_io_function):
        sentinel = object()
        register_pd_accessor(pd_io_function)(lambda: sentinel)
        assert getattr(pd, pd_io_function)() == sentinel

    def test_override_general_function(self, pd_general_function):
        sentinel = object()
        register_pd_accessor(pd_general_function)(lambda: sentinel)
        assert getattr(pd, pd_general_function)() == sentinel


class TestRegisterForOneBackend:
    def test_add_new_function(self):
        backend = "Pandas"
        expected_string_val = "Some string value"
        method_name = "new_method"

        @register_pd_accessor(method_name, backend=backend)
        def my_method_implementation():
            return expected_string_val

        with config_context(Backend=backend):
            assert getattr(pd, method_name)() == expected_string_val
        with config_context(Backend="Python_Test"):
            with pytest.raises(
                AttributeError,
                match=re.escape(
                    f"module 'modin.pandas' has no attribute {method_name}"
                ),
            ):
                getattr(pd, method_name)()

    def test_override_function(self):
        backend = "Pandas"
        expected_string_val = "Some string value"

        @register_pd_accessor("to_datetime", backend=backend)
        def my_method_implementation(*args, **kwargs):
            return expected_string_val

        with config_context(Backend=backend):
            # Since there are no query compiler inputs to to_datetime(), use
            # the to_datetime() implementation for Backend.get()
            assert pd.to_datetime(1) == expected_string_val

        with config_context(Backend="Python_Test"):
            # There are no query compiler inputs to to_datetime(), and
            # the current Backend.get() does not have a to_datetime() extension,
            # so fall back to the default to_datetime() implementation, which
            # should return the same result as pandas.to_datetime().
            eval_general(pd, pandas, lambda lib: lib.to_datetime(1))

    def test_add_new_non_method(self):
        backend = "Pandas"
        expected_val = 4
        attribute_name = "four"
        register_pd_accessor(attribute_name, backend=backend)(expected_val)
        with config_context(Backend=backend):
            assert pd.four == expected_val
        with config_context(Backend="Python_Test"):
            assert not hasattr(pd, attribute_name)

    def test_to_datetime_dispatches_to_implementation_for_input(self):

        @register_pd_accessor("to_datetime", backend="Pandas")
        def pandas_to_datetime(*args, **kwargs):
            return "pandas_to_datetime_result"

        with config_context(Backend="Pandas"):
            pandas_backend_series = pd.Series(1)

        with config_context(Backend="Python_Test"):
            python_backend_df = pd.Series(1)

        assert pd.to_datetime(pandas_backend_series) == "pandas_to_datetime_result"
        df_equals(
            pd.to_datetime(python_backend_df),
            pandas.to_datetime(python_backend_df._to_pandas()),
        )

    def test_concat_with_two_different_backends(self):
        with config_context(Backend="Pandas"):
            modin_on_pandas_df = pd.DataFrame({"a": [1, 2, 3]})
        with config_context(Backend="Python_Test"):
            modin_on_python_df = pd.DataFrame({"a": [4, 5, 6]})

        @register_pd_accessor("concat", backend="Pandas")
        def pandas_concat(*args, **kwargs):
            return "pandas_concat_result"

        @register_pd_accessor("concat", backend="Python_Test")
        def python_concat(*args, **kwargs):
            return "python_concat_result"

        # If the backends are different, we dispatch to the concat() override
        # for the backend of the first argument.
        assert (
            pd.concat([modin_on_pandas_df, modin_on_python_df])
            == "pandas_concat_result"
        )

        assert (
            pd.concat([modin_on_python_df, modin_on_pandas_df])
            == "python_concat_result"
        )

    def test_index_class_override(self):
        class FakeIndex:
            def __init__(self, _values):
                pass

            def fake_method(self) -> str:
                return "python_fake_index"

        register_pd_accessor("Index", backend="Python_Test")(FakeIndex)

        with config_context(Backend="Pandas"):
            # Should return an actual native pandas index object
            df_equals(pd.Index([1]).to_series(), pd.Series([1], index=[1]))

        with config_context(Backend="Python_Test"):
            # Should just return a string
            assert pd.Index([1]).fake_method() == "python_fake_index"
