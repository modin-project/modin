import re

import pytest

import modin.pandas as pd
from modin.pandas.api.extensions import register_base_accessor
from modin.pandas.api.extensions.extensions import _NON_EXTENDABLE_ATTRIBUTES
from modin.tests.pandas.utils import df_equals


@pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
def test_add_simple_method(Backend1, data_class):
    expected_string_val = "Some string value"
    method_name = "new_method"
    modin_object = data_class([1, 2, 3]).set_backend(Backend1)

    @register_base_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    assert hasattr(data_class, method_name)
    assert modin_object.new_method() == expected_string_val


@pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
def test_add_non_method(Backend1, data_class):
    expected_val = 4
    attribute_name = "four"
    register_base_accessor(name=attribute_name, backend=Backend1)(expected_val)

    assert data_class().set_backend(Backend1).four == expected_val


@pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
def test_method_uses_existing_methods(Backend1, data_class):
    modin_object = data_class([1, 2, 3]).set_backend(Backend1)
    method_name = "self_accessor"
    expected_result = modin_object.sum() / modin_object.count()

    @register_base_accessor(name=method_name, backend=Backend1)
    def my_average(self):
        return self.sum() / self.count()

    if data_class is pd.DataFrame:
        df_equals(modin_object.self_accessor(), expected_result)
    else:
        assert modin_object.self_accessor() == expected_result


@pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
def test_override_existing_method(Backend1, data_class):
    modin_object = data_class([3, 2, 1])

    @register_base_accessor(name="copy", backend=Backend1)
    def my_copy(self, *args, **kwargs):
        return self + 1

    df_equals(modin_object.set_backend(Backend1).copy(), modin_object + 1)


class TestDunders:
    """
    Make sure to test that we override special "dunder" methods like __len__
    correctly. python calls these methods with DataFrame.__len__(obj)
    rather than getattr(obj, "__len__")().
    source: https://docs.python.org/3/reference/datamodel.html#special-lookup
    """

    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test_len(self, Backend1, data_class):
        @register_base_accessor(name="__len__", backend=Backend1)
        def always_get_1(self):
            return 1

        modin_object = data_class([1, 2, 3])
        assert len(modin_object) == 3
        backend_object = modin_object.set_backend(Backend1)
        assert len(backend_object) == 1
        assert backend_object.__len__() == 1


class TestProperty:
    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test_override_loc(self, Backend1, data_class):
        modin_object = data_class([1, 2, 3])

        @register_base_accessor(name="loc", backend=Backend1)
        @property
        def my_loc(self):
            return self.index[0]

        assert isinstance(modin_object.set_backend(Backend1).loc, int)
        assert modin_object.set_backend(Backend1).loc == 0

    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test_add_deletable_property(self, Backend1, data_class):
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

        register_base_accessor(name=public_property_name, backend=Backend1)(
            property(fget=get_property, fset=set_property, fdel=del_property)
        )

        modin_object = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert not hasattr(modin_object, public_property_name)
        backend_object = modin_object.set_backend(Backend1)
        setattr(backend_object, public_property_name, "value")
        assert getattr(backend_object, public_property_name) == "value"
        delattr(backend_object, public_property_name)
        # check that the deletion works.
        assert not hasattr(backend_object, private_property_name)

    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test_non_settable_extension_property(self, Backend1, data_class):
        modin_object = data_class([0])
        property_name = "property_name"
        register_base_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: 4))
        )

        assert not hasattr(modin_object, property_name)
        backend_object = modin_object.set_backend(Backend1)
        assert getattr(backend_object, property_name) == 4
        with pytest.raises(AttributeError):
            setattr(backend_object, property_name, "value")

    @pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
    def test_delete_non_deletable_extension_property(self, Backend1, data_class):
        modin_object = data_class([0])
        property_name = "property_name"
        register_base_accessor(name=property_name, backend=Backend1)(
            property(fget=(lambda self: "value"))
        )

        assert not hasattr(modin_object, property_name)
        backend_object = modin_object.set_backend(Backend1)
        assert hasattr(backend_object, property_name)
        with pytest.raises(AttributeError):
            delattr(backend_object, property_name)


@pytest.mark.parametrize("data_class", [pd.DataFrame, pd.Series])
def test_deleting_extension_that_is_not_property_raises_attribute_error(
    Backend1, data_class
):
    expected_string_val = "Some string value"
    method_name = "new_method"

    @register_base_accessor(name=method_name, backend=Backend1)
    def my_method_implementation(self):
        return expected_string_val

    modin_object = data_class([0]).set_backend(Backend1)
    assert hasattr(data_class, method_name)
    with pytest.raises(AttributeError):
        delattr(modin_object, method_name)


@pytest.mark.parametrize("name", _NON_EXTENDABLE_ATTRIBUTES)
def test_disallowed_extensions(Backend1, name):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Cannot register an extension with the reserved name {name}."),
    ):
        register_base_accessor(name=name, backend=Backend1)("unused_value")
