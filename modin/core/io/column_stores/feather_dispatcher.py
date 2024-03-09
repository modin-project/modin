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

"""Module houses `FeatherDispatcher` class, that is used for reading `.feather` files."""

from pandas.io.common import stringify_path

from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.core.io.file_dispatcher import OpenFile
from modin.utils import import_optional_dependency


class FeatherDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.feather` files."""

    @classmethod
    def _read(cls, path, columns=None, **kwargs):
        """
        Read data from the file path, returning a query compiler.

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
        `PyArrow` engine and local files only are supported for now,
        multi threading is set to False by default.
        PyArrow feather is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/api.html#feather-format
        """
        path = stringify_path(path)
        path = cls.get_path(path)
        if columns is None:
            import_optional_dependency(
                "pyarrow", "pyarrow is required to read feather files."
            )
            from pyarrow import ipc

            with OpenFile(
                path,
                **(kwargs.get("storage_options", None) or {}),
            ) as file:
                # Opens the file to extract its metadata
                reader = ipc.open_file(file)
            # TODO: pyarrow's schema contains much more metadata than just column names, it also
            # has dtypes and index information that we could use when building a dataframe
            index_cols = frozenset(
                col
                for col in reader.schema.pandas_metadata["index_columns"]
                # 'index_columns' field may also contain dictionary fields describing actual
                # RangeIndices, so we're only filtering here for string column names
                if isinstance(col, str)
            )
            # Filtering out the columns that describe the frame's index
            columns = [col for col in reader.schema.names if col not in index_cols]
        return cls.build_query_compiler(
            path,
            columns,
            use_threads=False,
            storage_options=kwargs["storage_options"],
            dtype_backend=kwargs["dtype_backend"],
        )
