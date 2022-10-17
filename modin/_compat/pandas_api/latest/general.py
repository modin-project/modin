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

"""Module for 'latest pandas' compatibility layer for pandas methods."""

import pandas

from modin.utils import _inherit_docstrings, to_pandas
from modin.logging import enable_logging
from modin.pandas import DataFrame
from modin.error_message import ErrorMessage


@_inherit_docstrings(pandas.from_dummies, apilink="pandas.from_dummies")
@enable_logging
def from_dummies(data, sep=None, default_category=None):
    ErrorMessage.default_to_pandas("from_dummies")
    if isinstance(data, DataFrame):
        data = to_pandas(data)
    return DataFrame(
        pandas.from_dummies(data, sep=sep, default_category=default_category)
    )


@_inherit_docstrings(pandas.pivot_table, apilink="pandas.pivot_table")
@enable_logging
def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
    observed=False,
    sort=True,
):
    if not isinstance(data, DataFrame):
        raise ValueError(
            "can not create pivot table with instance of type {}".format(type(data))
        )

    return data.pivot_table(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
        observed=observed,
        sort=sort,
    )
