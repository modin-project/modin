from modin.engines.base.io.column_stores.column_store_reader import ColumnStoreReader


class FeatherReader(ColumnStoreReader):
    @classmethod
    def read(cls, path, columns=None, **kwargs):
        """Read a pandas.DataFrame from Feather format.
           Ray DataFrame only supports pyarrow engine for now.

        Args:
            path: The filepath of the feather file.
                  We only support local files for now.
                multi threading is set to True by default
            columns: not supported by pandas api, but can be passed here to read only
                specific columns

        Notes:
            pyarrow feather is used. Please refer to the documentation here
            https://arrow.apache.org/docs/python/api.html#feather-format
        """
        if columns is None:
            from pyarrow.feather import FeatherReader

            fr = FeatherReader(path)
            columns = [fr.get_column_name(i) for i in range(fr.num_columns)]
        return cls.build_query_compiler(path, columns, use_threads=False)
