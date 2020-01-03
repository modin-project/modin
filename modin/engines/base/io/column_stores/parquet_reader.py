import os

from modin.engines.base.io.column_stores.column_store_reader import ColumnStoreReader
from modin.error_message import ErrorMessage


class ParquetReader(ColumnStoreReader):
    @classmethod
    def read(cls, path, engine, columns, **kwargs):
        """Load a parquet object from the file path, returning a DataFrame.
           Ray DataFrame only supports pyarrow engine for now.

        Args:
            path: The filepath of the parquet file.
                  We only support local files for now.
            engine: Ray only support pyarrow reader.
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
            original_path = path
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
                    path = os.path.join(root, files[0])
                    break
            partitioned_columns = list(partitioned_columns)
            if len(partitioned_columns):
                ErrorMessage.default_to_pandas("Partitioned Columns in Parquet")
                return cls.single_worker_read(
                    original_path, engine=engine, columns=columns, **kwargs
                )
        else:
            directory = False

        if not columns:
            if directory:
                # Path of the sample file that we will read to get the remaining columns
                pd = ParquetDataset(path)
                column_names = pd.schema.names
            else:
                pf = ParquetFile(path)
                column_names = pf.metadata.schema.names
            columns = [name for name in column_names if not PQ_INDEX_REGEX.match(name)]
        return cls.build_query_compiler(path, columns, **kwargs)
