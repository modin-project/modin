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

import modin.pandas as pd
from modin.pandas.api.extensions import register_dataframe_accessor


def test_dataframe_extension_simple_method():
    expected_string_val = "Some string value"
    method_name = "new_method"
    df = pd.DataFrame([1, 2, 3])

    @register_dataframe_accessor(method_name)
    def my_method_implementation(self):
        return expected_string_val

    assert method_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[method_name] is my_method_implementation
    assert df.new_method() == expected_string_val


def test_dataframe_extension_non_method():
    expected_val = 4
    attribute_name = "four"
    register_dataframe_accessor(attribute_name)(expected_val)
    df = pd.DataFrame([1, 2, 3])

    assert attribute_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[attribute_name] == 4
    assert df.four == expected_val


def test_dataframe_extension_accessing_existing_methods():
    df = pd.DataFrame([1, 2, 3])
    method_name = "self_accessor"
    expected_result = df.sum() / df.count()

    @register_dataframe_accessor(method_name)
    def my_average(self):
        return self.sum() / self.count()

    assert method_name in pd.dataframe._DATAFRAME_EXTENSIONS_.keys()
    assert pd.dataframe._DATAFRAME_EXTENSIONS_[method_name] is my_average
    assert df.self_accessor().equals(expected_result)
