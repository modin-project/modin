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

from functools import cached_property

import pytest

import modin.pandas as pd
from modin.config import Backend
from modin.config import context as config_context
from modin.pandas.api.extensions import (
    register_dataframe_groupby_accessor,
    register_series_groupby_accessor,
)
from modin.pandas.groupby import DataFrameGroupBy, SeriesGroupBy
from modin.tests.pandas.utils import default_to_pandas_ignore_string, df_equals
from modin.tests.test_utils import warns_that_defaulting_to_pandas


@pytest.mark.parametrize(
    "get_groupby,register_accessor",
    (
        (lambda df: df.groupby("col0"), register_dataframe_groupby_accessor),
        (lambda df: df.groupby("col0")["col1"], register_series_groupby_accessor),
    ),
)
@config_context(Backend="Pandas")
@pytest.mark.parametrize("extension_backend", [None, "Pandas"])
@pytest.mark.parametrize("method_name", ["new_method", "sum"])
def test_add_simple_method(
    get_groupby, register_accessor, extension_backend, method_name
):
    expected_string_val = "expected_string_val"
    df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]})

    @register_accessor(method_name, backend=extension_backend)
    def new_method(self):
        return expected_string_val

    groupby = get_groupby(df)
    assert hasattr(groupby, method_name)
    assert getattr(groupby, method_name)() == expected_string_val


def test_dataframe_accessor_for_method_that_series_groupby_does_not_override():
    """
    Test sum(), a DataFrameGroupBy method that SeriesGroupBy inherits without overriding.

    Registering an extension method for DataFrameGroupBy should override sum()
    behavior for both DataFrameGroupBy and SeriesGroupBy.
    """
    # Check that SeriesGroupBy inherits sum() from DataFrameGroupBy, with the only
    # difference being that SeriesGroupBy's sum() is wrapped in a method for handling
    # extensions and casting.
    assert DataFrameGroupBy.sum is SeriesGroupBy.sum._wrapped_method_for_casting
    df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]})
    accessor_result = "test_result"
    register_dataframe_groupby_accessor("sum", backend=Backend.get())(
        lambda self, *args, **kwargs: accessor_result
    )
    groupby_sum_result = df.groupby("col0").sum()
    assert groupby_sum_result == accessor_result
    series_groupby_sum_result = df.groupby("col0")["col1"].sum()
    assert series_groupby_sum_result == accessor_result


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_dataframe_accessor_for_method_that_series_groupby_overrides():
    """
    Test describe(), a DataFrameGroupBy method that SeriesGroupBy overrides.

    Registering an extension method for DataFrameGroupBy should not affect
    SeriesGroupBy's describe() method.
    """
    # Check that SeriesGroupBy overrides describe().
    assert (
        DataFrameGroupBy.describe
        is not SeriesGroupBy.describe._wrapped_method_for_casting
    )
    df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]})
    original_series_groupby_describe_result = df.groupby("col0")["col1"].describe()
    accessor_result = "test_result"
    register_dataframe_groupby_accessor("describe", backend=Backend.get())(
        lambda self, *args, **kwargs: accessor_result
    )
    groupby_describe_result = df.groupby("col0").describe()
    assert groupby_describe_result == accessor_result
    series_groupby_describe_result = df.groupby("col0")["col1"].describe()
    df_equals(series_groupby_describe_result, original_series_groupby_describe_result)


@pytest.mark.parametrize(
    "get_groupby,register_accessor",
    (
        (lambda df: df.groupby("col0"), register_dataframe_groupby_accessor),
        (lambda df: df.groupby("col0")["col1"], register_series_groupby_accessor),
    ),
)
class TestProperty:

    @pytest.mark.parametrize("df_backend", ["Pandas", "Python_Test"])
    def test_add_read_only_property_for_all_backends(
        self, df_backend, get_groupby, register_accessor
    ):
        expected_string_val = "expected_string_val"
        property_name = "new_property"

        @register_dataframe_groupby_accessor(property_name)
        @property
        def new_property(self):
            return expected_string_val

        with config_context(Backend=df_backend):
            df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]})
            assert get_groupby(df).new_property == expected_string_val

            with pytest.raises(AttributeError):
                del df.groupby("col0").new_property

            with pytest.raises(AttributeError):
                df.groupby("col0").new_property = "new_value"

    def test_override_ngroups_getter_for_one_backend(
        self, get_groupby, register_accessor
    ):
        accessor_ngroups = -1
        property_name = "ngroups"

        @register_accessor(property_name, backend="Pandas")
        @property
        def ngroups(self):
            return accessor_ngroups

        pandas_df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]}).move_to(
            "pandas"
        )
        groupby = get_groupby(pandas_df)
        assert groupby.ngroups == accessor_ngroups

        # Check that the accessor doesn't work on the Python_Test backend.
        python_test_df = pandas_df.move_to("Python_Test")
        groupby = get_groupby(python_test_df)
        # groupby.ngroups defaults to pandas at the API layer,
        # where it warns that it's doing so, even for dataframes using the
        # Pandas backend.
        with warns_that_defaulting_to_pandas():
            assert groupby.ngroups == 3

    def test_add_ngroups_setter_and_deleter_for_one_backend(
        self, get_groupby, register_accessor
    ):

        def _get_ngroups(self):
            return self._ngroups

        def _delete_ngroups(self):
            delattr(self, "_ngroups")

        def _set_ngroups(self, value):
            self._ngroups = value

        register_accessor("ngroups", backend="Pandas")(
            property(fget=_get_ngroups, fset=_set_ngroups, fdel=_delete_ngroups)
        )

        python_test_df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]}).move_to(
            "python_test"
        )

        python_test_groupby = get_groupby(python_test_df)

        with warns_that_defaulting_to_pandas():
            assert python_test_groupby.ngroups == 3

        with pytest.raises(AttributeError):
            python_test_groupby.ngroups = 4

        with pytest.raises(AttributeError):
            del python_test_groupby.ngroups

        pandas_groupby = get_groupby(python_test_df.move_to("Pandas"))

        assert not hasattr(pandas_groupby, "ngroups")

        pandas_groupby.ngroups = -1

        assert pandas_groupby.ngroups == -1

        # Deleting ngroups should delete the private attribute _ngroups.
        del pandas_groupby.ngroups

        # now getting ngroups should raise an AttributeError because the
        # private attribute _ngroups is missing.
        assert not hasattr(pandas_groupby, "ngroups")

    def test_add_deletable_property_for_one_backend(
        self, get_groupby, register_accessor
    ):
        public_property_name = "property_name"
        private_property_name = "_property_name"

        # register a public property `public_property_name` that is backed by
        # a private attribute `private_property_name`.

        def get_property(self):
            return getattr(self, private_property_name)

        def set_property(self, value):
            setattr(self, private_property_name, value)

        def del_property(self):
            # Note that deleting the public property deletes the private
            # attribute, not the public property itself.
            delattr(self, private_property_name)

        register_accessor(name=public_property_name, backend="Pandas")(
            property(get_property, set_property, del_property)
        )

        python_test_df = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]}).move_to(
            "python_test"
        )

        python_test_groupby = get_groupby(python_test_df)

        assert not hasattr(python_test_groupby, public_property_name)

        pandas_df = python_test_df.move_to("pandas")
        pandas_groupby = get_groupby(pandas_df)

        setattr(pandas_groupby, public_property_name, "value")
        assert getattr(pandas_groupby, public_property_name) == "value"
        delattr(pandas_groupby, public_property_name)
        assert not hasattr(pandas_groupby, private_property_name)

    @pytest.mark.filterwarnings(default_to_pandas_ignore_string)
    def test_override_cached_property(self, get_groupby, register_accessor):
        @cached_property
        def groups(self):
            return {"group": pd.Index(["test"])}

        register_accessor("groups", backend="Pandas")(groups)
        pandas_df = pd.DataFrame({"col0": [1], "col1": [2]}).move_to("pandas")
        assert get_groupby(pandas_df).groups == {"group": pd.Index(["test"])}


def test_deleting_extension_that_is_not_property_raises_attribute_error():
    expected_string_val = "Some string value"
    method_name = "new_method"

    @register_dataframe_groupby_accessor(name=method_name)
    def my_method_implementation(self):
        return expected_string_val

    groupby = pd.DataFrame({"col0": [1, 2, 3], "col1": [4, 5, 6]}).groupby("col0")
    assert hasattr(DataFrameGroupBy, method_name)
    assert getattr(groupby, method_name)() == expected_string_val
    with pytest.raises(AttributeError):
        delattr(groupby, method_name)


@pytest.mark.skipif(Backend.get() == "Pandas", reason="already on pandas backend")
def test_get_extension_from_dataframe_that_is_on_non_default_backend_when_auto_switch_is_false():
    with config_context(AutoSwitchBackend=False):
        pandas_df = pd.DataFrame([1, 2]).move_to("Pandas")
        register_dataframe_groupby_accessor("sum", backend="Pandas")(
            lambda df: "small_sum_result"
        )
        assert pandas_df.groupby(0).sum() == "small_sum_result"
