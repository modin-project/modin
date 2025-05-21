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
from unittest import mock

import pandas
import pytest

import modin.pandas as pd
from modin.config import Backend
from modin.config import context as config_context
from modin.pandas.api.extensions import register_dataframe_accessor

default___init__ = pd.DataFrame._extensions[None]["__init__"]


def test_dataframe_extension_simple_method(Backend1):
    expected_string_val = "Some string value"
    method_name = "new_method"
    df = pd.DataFrame([1, 2, 3]).set_backend(Backend1)

    @register_dataframe_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    assert hasattr(pd.DataFrame, method_name)
    assert df.new_method() == expected_string_val


def test_dataframe_extension_non_method(Backend1):
    expected_val = 4
    attribute_name = "four"
    register_dataframe_accessor(name=attribute_name, backend=Backend1)(expected_val)
    df = pd.DataFrame([1, 2, 3]).set_backend(Backend1)

    assert df.four == expected_val


def test_dataframe_extension_accessing_existing_methods(Backend1):
    df = pd.DataFrame([1, 2, 3]).set_backend(Backend1)
    method_name = "self_accessor"
    expected_result = df.sum() / df.count()

    @register_dataframe_accessor(name=method_name, backend=Backend1)
    def my_average(self):
        return self.sum() / self.count()

    assert df.self_accessor().equals(expected_result)


def test_dataframe_extension_overrides_existing_method(Backend1):
    df = pd.DataFrame([3, 2, 1])
    assert df.sort_values(0).iloc[0, 0] == 1

    @register_dataframe_accessor(name="sort_values", backend=Backend1)
    def my_sort_values(self):
        return self

    assert df.set_backend(Backend1).sort_values().iloc[0, 0] == 3


@pytest.mark.parametrize(
    "method_name",
    [
        "pow",
        "__pow__",
        "__ipow__",
    ],
)
def test_dataframe_extension_overrides_pow_github_issue_7495(method_name):
    register_dataframe_accessor(method_name, backend="Pandas")(
        lambda *args, **kwargs: 4
    )
    assert getattr(pd.DataFrame([1]).set_backend("Pandas"), method_name)() == 4


def test_override_pow_and__pow__to_different_implementations():
    register_dataframe_accessor("pow", backend="Pandas")(
        lambda *args, **kwargs: "pow_result"
    )
    register_dataframe_accessor("__pow__", backend="Pandas")(
        lambda *args, **kwargs: "__pow___result"
    )
    df = pd.DataFrame([1]).set_backend("pandas")
    assert df.pow() == "pow_result"
    assert df.__pow__() == "__pow___result"


def test_dataframe_extension_method_uses_superclass_method(Backend1):
    df = pd.DataFrame([3, 2, 1])
    assert df.sort_values(0).iloc[0, 0] == 1

    @register_dataframe_accessor(name="sort_values", backend=Backend1)
    def my_sort_values(self, by):
        return super(pd.DataFrame, self).sort_values(by=by, ascending=False)

    assert df.set_backend(Backend1).sort_values(by=0).iloc[0, 0] == 3


class TestOverride__init__:
    def test_override_one_backend_and_pass_no_query_compilers(self):
        default_backend = Backend.get()
        backend_init = mock.Mock(wraps=default___init__)
        register_dataframe_accessor(name="__init__", backend=default_backend)(
            backend_init
        )
        output_df = pd.DataFrame([1], index=["a"], columns=["b"])
        assert output_df.get_backend() == default_backend
        backend_init.assert_has_calls(
            [
                mock.call(output_df, [1], index=["a"], columns=["b"]),
                # There's a second, internal call to the dataframe constructor that
                # uses a different dataframe as `self`.
                mock.call(mock.ANY, query_compiler=output_df._query_compiler),
            ]
        )

    def test_override_one_backend_and_pass_query_compiler_kwarg(self):
        backend = "Pandas"
        backend_init = mock.Mock(wraps=default___init__)
        register_dataframe_accessor(name="__init__", backend=backend)(backend_init)

        with config_context(Backend=backend):
            input_df = pd.DataFrame()

        backend_init.reset_mock()
        output_df = pd.DataFrame(query_compiler=input_df._query_compiler)
        assert output_df.get_backend() == backend
        backend_init.assert_called_once_with(
            output_df, query_compiler=input_df._query_compiler
        )

    @pytest.mark.parametrize("input_backend", ["Python_Test", "Pandas"])
    def test_override_all_backends_and_pass_query_compiler_kwarg(self, input_backend):
        backend_init = mock.Mock(wraps=default___init__)
        register_dataframe_accessor(name="__init__")(backend_init)

        with config_context(Backend=input_backend):
            input_df = pd.DataFrame()

        backend_init.reset_mock()
        output_df = pd.DataFrame(query_compiler=input_df._query_compiler)
        assert output_df.get_backend() == input_backend
        backend_init.assert_called_once_with(
            output_df, query_compiler=input_df._query_compiler
        )


class TestDunders:
    """
    Make sure to test that we override special "dunder" methods like __len__
    correctly. python calls these methods with DataFrame.__len__(obj)
    rather than getattr(obj, "__len__")().
    source: https://docs.python.org/3/reference/datamodel.html#special-lookup
    """

    def test_len(self, Backend1):
        @register_dataframe_accessor(name="__len__", backend=Backend1)
        def always_get_1(self):
            return 1

        df = pd.DataFrame([1, 2, 3])
        assert len(df) == 3
        backend_df = df.set_backend(Backend1)
        assert len(backend_df) == 1
        assert backend_df.__len__() == 1

    def test_repr(self, Backend1):
        @register_dataframe_accessor(name="__repr__", backend=Backend1)
        def simple_repr(self) -> str:
            return "dataframe_string"

        df = pd.DataFrame([1, 2, 3])
        assert repr(df) == repr(df.modin.to_pandas())
        backend_df = df.set_backend(Backend1)
        assert repr(backend_df) == "dataframe_string"
        assert backend_df.__repr__() == "dataframe_string"


class TestProperty:
    def test_override_columns(self, Backend1):
        df = pd.DataFrame([["a", "b"]])

        def set_columns(self, new_columns):
            self._query_compiler.columns = [f"{v}_custom" for v in new_columns]

        register_dataframe_accessor(name="columns", backend=Backend1)(
            property(
                fget=(lambda self: self._query_compiler.columns[::-1]), fset=set_columns
            )
        )

        assert list(df.columns) == [0, 1]
        backend_df = df.set_backend(Backend1)
        assert list(backend_df.columns) == [1, 0]
        backend_df.columns = [2, 3]
        assert list(backend_df.columns) == [
            "3_custom",
            "2_custom",
        ]

    def test_search_for_missing_attribute_in_overridden_columns(self, Backend1):
        """
        Test a scenario where we override the columns getter, then search for a
        missing dataframe attribute. Modin should look in the dataframe's
        overridden columns for the attribute.
        """
        column_name = "column_name"
        column_getter = mock.Mock(wraps=(lambda self: self._query_compiler.columns))
        register_dataframe_accessor(name="columns", backend=Backend1)(
            property(fget=column_getter)
        )

        df = pd.DataFrame({column_name: ["a"]}).set_backend(Backend1)

        with pytest.raises(
            AttributeError,
            match="'DataFrame' object has no attribute 'non_existent_column'",
        ):
            getattr(df, "non_existent_column")
        column_getter.assert_called_once_with(df)

    def test_add_deletable_property(self, Backend1):
        public_property_name = "property_name"
        private_property_name = "_property_name"

        # register a public property `public_property_name` that is backed by
        # a private attribute `private_property_name`.

        def get_property(self):
            return getattr(self, private_property_name)

        def set_property(self, value):
            setattr(self, private_property_name, value)

        def del_property(self):
            delattr(self, private_property_name)

        register_dataframe_accessor(name=public_property_name, backend=Backend1)(
            property(get_property, set_property, del_property)
        )

        df = pd.DataFrame([0])
        assert not hasattr(df, public_property_name)
        backend_df = df.set_backend(Backend1)
        setattr(backend_df, public_property_name, "value")
        assert hasattr(backend_df, private_property_name)
        assert getattr(backend_df, private_property_name) == "value"
        delattr(backend_df, public_property_name)
        # check that the deletion works.
        assert not hasattr(backend_df, private_property_name)

    def test_non_settable_extension_property(self, Backend1):
        df = pd.DataFrame([0])
        property_name = "property_name"

        register_dataframe_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: 4))
        )

        assert not hasattr(df, property_name)
        backend_df = df.set_backend(Backend1)
        assert getattr(backend_df, property_name) == 4
        with pytest.raises(AttributeError):
            setattr(backend_df, property_name, "value")

    def test_delete_non_deletable_extension_property(self, Backend1):
        property_name = "property_name"

        register_dataframe_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: "value"))
        )

        df = pd.DataFrame([0])
        assert not hasattr(df, property_name)
        backend_df = df.set_backend(Backend1)
        assert hasattr(backend_df, property_name)
        with pytest.raises(AttributeError):
            delattr(backend_df, property_name)


def test_deleting_extension_that_is_not_property_raises_attribute_error(Backend1):
    expected_string_val = "Some string value"
    method_name = "new_method"

    @register_dataframe_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    df = pd.DataFrame([1, 2, 3]).set_backend(Backend1)
    assert hasattr(pd.DataFrame, method_name)
    assert df.new_method() == expected_string_val
    with pytest.raises(AttributeError):
        delattr(df, method_name)


def test_disallowed_extensions(Backend1, non_extendable_attribute_name):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Cannot register an extension with the reserved name {non_extendable_attribute_name}."
        ),
    ):
        register_dataframe_accessor(
            name=non_extendable_attribute_name, backend=Backend1
        )("unused_value")


def test_correct_backend_with_pin(Backend1):
    # Ensures that the correct implementation is used when dispatching an operation on a pinned
    # frame, as an earlier implementation used the wrong extension method while preserving the
    # correct backend.

    @register_dataframe_accessor(name="__repr__", backend=Backend1)
    def my_repr(self):
        return "fake_repr"

    with config_context(AutoSwitchBackend=False, Backend="Python_Test"):
        df = pd.DataFrame([1])
        assert df.get_backend() == "Python_Test"
        assert repr(df) == repr(pandas.DataFrame([1]))
        df.set_backend(Backend1, inplace=True)
        df.pin_backend(inplace=True)
        assert df.get_backend() == Backend1
        assert repr(df) == "fake_repr"


def test_get_extension_from_dataframe_that_is_on_non_default_backend_when_auto_switch_is_false(
    Backend1,
):
    with config_context(AutoSwitchBackend=False, Backend=Backend1):
        pandas_df = pd.DataFrame([1, 2]).move_to("Pandas")
        register_dataframe_accessor("sum", backend="Pandas")(
            lambda df: "small_sum_result"
        )
        assert pandas_df.sum() == "small_sum_result"
