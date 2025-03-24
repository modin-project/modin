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

import pytest

import modin.pandas as pd
from modin.pandas.api.extensions import register_series_accessor
from modin.pandas.api.extensions.extensions import _NON_EXTENDABLE_ATTRIBUTES


def test_series_extension_simple_method(Backend1):
    expected_string_val = "Some string value"
    method_name = "new_method"
    ser = pd.Series([1, 2, 3]).set_backend(Backend1)

    @register_series_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    assert hasattr(pd.Series, method_name)
    assert ser.new_method() == expected_string_val


def test_series_extension_non_method(Backend1):
    expected_val = 4
    attribute_name = "four"
    register_series_accessor(name=attribute_name, backend=Backend1)(expected_val)
    ser = pd.Series([1, 2, 3]).set_backend(Backend1)

    assert ser.four == expected_val


def test_series_extension_accessing_existing_methods(Backend1):
    ser = pd.Series([1, 2, 3]).set_backend(Backend1)
    method_name = "self_accessor"
    expected_result = ser.sum() / ser.count()

    @register_series_accessor(name=method_name, backend=Backend1)
    def my_average(self):
        return self.sum() / self.count()

    assert ser.self_accessor() == expected_result


def test_series_extension_overrides_existing_method(Backend1):
    series = pd.Series([3, 2, 1])
    assert series.sort_values().iloc[0] == 1

    @register_series_accessor(name="sort_values", backend=Backend1)
    def my_sort_values(self):
        return self

    assert series.set_backend(Backend1).sort_values().iloc[0] == 3


def test_series_extension_method_uses_superclass_method(Backend1):
    series = pd.Series([3, 2, 1], name="name")
    assert series.sort_values().iloc[0] == 1

    @register_series_accessor(name="sort_values", backend=Backend1)
    def my_sort_values(self):
        return super(pd.Series, self).sort_values(by="name", ascending=False)

    assert series.set_backend(Backend1).sort_values().iloc[0] == 3


class TestDunders:
    """
    Make sure to test that we override special "dunder" methods like __len__
    correctly. python calls these methods with DataFrame.__len__(obj)
    rather than getattr(obj, "__len__")().
    source: https://docs.python.org/3/reference/datamodel.html#special-lookup
    """

    def test_len(self, Backend1):
        @register_series_accessor(name="__len__", backend=Backend1)
        def always_get_1(self):
            return 1

        series = pd.Series([1, 2, 3])
        assert len(series) == 3
        backend_series = series.set_backend(Backend1)
        assert len(backend_series) == 1
        assert backend_series.__len__() == 1

    def test_repr(self, Backend1):
        @register_series_accessor(name="__repr__", backend=Backend1)
        def simple_repr(self) -> str:
            return "series_string"

        series = pd.Series([1, 2, 3])
        assert repr(series) == repr(series.modin.to_pandas())
        backend_series = series.set_backend(Backend1)
        assert repr(backend_series) == "series_string"
        assert backend_series.__repr__() == "series_string"


class TestProperty:
    def test_override_index(self, Backend1):
        series = pd.Series(["a", "b"])

        def set_index(self, new_index):
            self._query_compiler.index = [f"{v}_custom" for v in new_index]

        register_series_accessor(name="index", backend=Backend1)(
            property(fget=lambda self: self._query_compiler.index[::-1], fset=set_index)
        )

        assert list(series.index) == [0, 1]
        backend_series = series.set_backend(Backend1)
        assert list(backend_series.index) == [1, 0]
        backend_series.index = [2, 3]
        assert list(backend_series.index) == ["3_custom", "2_custom"]

    def test_add_deletable_property(self, Backend1):

        # register a public property `public_property_name` that is backed by
        # a private attribute `private_property_name`.

        public_property_name = "property_name"
        private_property_name = "_property_name"

        def get_property(self):
            return getattr(self, private_property_name)

        def set_property(self, value):
            setattr(self, private_property_name, value)

        def del_property(self):
            delattr(self, private_property_name)

        register_series_accessor(name=public_property_name, backend=Backend1)(
            property(get_property, set_property, del_property)
        )

        series = pd.Series([0])
        assert not hasattr(series, public_property_name)
        backend_series = series.set_backend(Backend1)
        setattr(backend_series, public_property_name, "value")
        assert hasattr(backend_series, private_property_name)
        assert getattr(backend_series, public_property_name) == "value"
        delattr(backend_series, public_property_name)
        # check that the deletion works.
        assert not hasattr(backend_series, private_property_name)

    def test_non_settable_extension_property(self, Backend1):

        property_name = "property_name"
        register_series_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: 4))
        )

        series = pd.Series([0])
        assert not hasattr(series, property_name)
        backend_series = series.set_backend(Backend1)
        assert getattr(backend_series, property_name) == 4
        with pytest.raises(AttributeError):
            setattr(backend_series, property_name, "value")

    def test_delete_non_deletable_extension_property(self, Backend1):

        property_name = "property_name"
        register_series_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: "value"))
        )

        series = pd.Series([0])
        assert not hasattr(series, property_name)
        backend_series = series.set_backend(Backend1)
        with pytest.raises(AttributeError):
            delattr(backend_series, property_name)


def test_deleting_extension_that_is_not_property_raises_attribute_error(Backend1):
    expected_string_val = "Some string value"
    method_name = "new_method"
    series = pd.Series([1, 2, 3]).set_backend(Backend1)

    @register_series_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    assert hasattr(pd.Series, method_name)
    assert series.new_method() == expected_string_val
    with pytest.raises(AttributeError):
        delattr(series, method_name)


@pytest.mark.parametrize("name", _NON_EXTENDABLE_ATTRIBUTES)
def test_disallowed_extensions(Backend1, name):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Cannot register an extension with the reserved name {name}."),
    ):
        register_series_accessor(name=name, backend=Backend1)("unused_value")
