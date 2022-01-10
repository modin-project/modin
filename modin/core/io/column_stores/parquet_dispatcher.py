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

"""Module houses `ParquetDispatcher` class, that is used for reading `.parquet` files."""

import os

from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import import_optional_dependency


class ParquetDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.parquet` files."""

    @classmethod
    def _read(cls, path, engine, columns, **kwargs):
        """
        Load a parquet object from the file path, returning a query compiler.

        Parameters
        ----------
        path : str, path object or file-like object
            The filepath of the parquet file in local filesystem or hdfs.
        engine : str
            Parquet library to use (only 'PyArrow' is supported for now).
        columns : list
            If not None, only these columns will be read from the file.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        BaseQueryCompiler
            A new Query Compiler.

        Notes
        -----
        ParquetFile API is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/parquet.html
        """
        import_optional_dependency(
            "pyarrow",
            "pyarrow is required to read parquet files.",
        )
        from pyarrow.parquet import ParquetDataset
        from modin.pandas.io import PQ_INDEX_REGEX

        if isinstance(path, str) and os.path.isdir(path):
            partitioned_columns = set()
            # We do a tree walk of the path directory because partitioned
            # parquet directories have a unique column at each directory level.
            # Thus, we can use os.walk(), which does a dfs search, to walk
            # through the different columns that the data is partitioned on
            for (root, dir_names, files) in os.walk(path):
                if dir_names:
                    partitioned_columns.add(dir_names[0].split("=")[0])
                if files:
                    # Metadata files, git files, .DSStore
                    if files[0][0] == ".":
                        continue
                    break
            partitioned_columns = list(partitioned_columns)
            if len(partitioned_columns):
                ErrorMessage.default_to_pandas("Mixed Partitioning Columns in Parquet")
                return cls.single_worker_read(
                    path, engine=engine, columns=columns, **kwargs
                )

        if not columns:
            import fsspec.core
            from pandas.io.common import is_fsspec_url

            fs, path_ = (
                fsspec.core.url_to_fs(path, **(kwargs.get("storage_options") or {}))
                if is_fsspec_url(path)
                else (None, path)
            )

            dataset = ParquetDataset(path_, filesystem=fs, use_legacy_dataset=False)
            column_names = dataset.schema.names

            if dataset.schema.pandas_metadata is not None:
                index_columns = dataset.schema.pandas_metadata.get("index_columns", [])
                column_names = [c for c in column_names if c not in index_columns]
            columns = [name for name in column_names if not PQ_INDEX_REGEX.match(name)]
        return cls.build_query_compiler(path, columns, **kwargs)
