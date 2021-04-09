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

"""This module houses `FeatherDispatcher` class, that is used for
reading `.feather` files.
"""

from modin.engines.base.io.column_stores.column_store_dispatcher import (
    ColumnStoreDispatcher,
)


class FeatherDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.feather` files. Inherits some common for
    columnar store files util functions from `ColumnStoreDispatcher` class.
    """

    @classmethod
    def _read(cls, path, columns=None, **kwargs):
        """Read data from the file path, returning a Modin DataFrame.

        Parameters
        ----------
        path : str or file-like object
            The filepath of the feather file.
        columns : array-like, optional
            Columns to read from file. If not provided, all columns are read.
        **kwargs : dict
            `read_feather` function kwargs.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `pyarrow` engine and local files only are supported for now,
        multi threading is set to False by default.
        pyarrow feather is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/api.html#feather-format
        """
        if columns is None:
            from pyarrow.feather import read_feather

            df = read_feather(path)
        return cls.build_query_compiler(path, df.columns, use_threads=False)
