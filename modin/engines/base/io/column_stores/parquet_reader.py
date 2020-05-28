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

import os

from modin.engines.base.io.column_stores.column_store_reader import ColumnStoreReader
from modin.error_message import ErrorMessage


class ParquetReader(ColumnStoreReader):
    @classmethod
    def read(cls, path, engine, columns, **kwargs):
        """Load a parquet object from the file path, returning a Modin DataFrame.
           Modin only supports pyarrow engine for now.

        Args:
            path: The filepath of the parquet file.
                  We only support local files for now.
            engine: Modin only supports pyarrow reader.
                    This argument doesn't do anything for now.
            kwargs: Pass into parquet's read_pandas function.

        Notes:
            ParquetFile API is used. Please refer to the documentation here
            https://arrow.apache.org/docs/python/parquet.html
        """
        from pyarrow.parquet import ParquetFile, ParquetDataset
        from modin.pandas.io import PQ_INDEX_REGEX

        if os.path.isdir(path):
            partitioned_columns = set()
            directory = True
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
        else:
            directory = False
        if not columns:
            if directory:
                # Path of the sample file that we will read to get the remaining columns
                pd = ParquetDataset(path)
                meta = pd.metadata
                column_names = pd.schema.names
            else:
                meta = ParquetFile(path).metadata
                column_names = meta.schema.names
            if meta is not None:
                # This is how we convert the metadata from pyarrow to a python
                # dictionary, from which we then get the index columns.
                # We use these to filter out from the columns in the metadata since
                # the pyarrow storage has no concept of row labels/index.
                # This ensures that our metadata lines up with the partitions without
                # extra communication steps once we `have done all the remote
                # computation.
                index_columns = eval(
                    meta.metadata[b"pandas"].replace(b"null", b"None")
                ).get("index_columns", [])
                column_names = [c for c in column_names if c not in index_columns]
            columns = [name for name in column_names if not PQ_INDEX_REGEX.match(name)]
        return cls.build_query_compiler(path, columns, **kwargs)
