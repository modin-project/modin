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

"""Utilities for internal use by the ``HdkOnNativeDataframe``."""


import pandas
import pyarrow

from modin.error_message import ErrorMessage


class LazyProxyCategoricalDtype(pandas.CategoricalDtype):
    """
    Proxy class for lazily retrieving categorical dtypes from arrow tables.

    Parameters
    ----------
    table : pyarrow.Table
        Source table.
    column_name : str
        Column name.
    """

    def __init__(self, table: pyarrow.Table, column_name: str):
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=table is None,
            extra_log="attempted to bind 'None' pyarrow table to a lazy category",
        )
        self._table = table
        self._column_name = column_name
        self._ordered = False
        self._lazy_categories = None

    def _new(self, table: pyarrow.Table, column_name: str) -> pandas.CategoricalDtype:
        """
        Create a new proxy, if either table or column name are different.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.
        column_name : str
            Column name.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._table is None:
            # The table has been materialized, we don't need a proxy anymore.
            return pandas.CategoricalDtype(self.categories)
        elif table is self._table and column_name == self._column_name:
            return self
        else:
            return LazyProxyCategoricalDtype(table, column_name)

    @property
    def _categories(self):  # noqa: GL08
        if self._table is not None:
            chunks = self._table.column(self._column_name).chunks
            cat = pandas.concat([chunk.dictionary.to_pandas() for chunk in chunks])
            self._lazy_categories = self.validate_categories(cat.unique())
            self._table = None  # The table is not required any more
        return self._lazy_categories

    @_categories.setter
    def _set_categories(self, categories):  # noqa: GL08
        self._lazy_categories = categories
        self._table = None


def check_join_supported(type: str):
    """
    Check if join type is supported by HDK.

    Parameters
    ----------
    type : str
        Join type.

    Returns
    -------
    None
    """
    if type not in ("inner", "left"):
        raise NotImplementedError(f"{type} join is not supported by the HDK engine")
