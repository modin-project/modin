import pandas

from modin.engines.base.io import BaseIO


class RayIO(BaseIO):
    @classmethod
    def to_sql(cls, qc, **kwargs):
        """Write records stored in a DataFrame to a SQL database.
        Args:
            qc: the query compiler of the DF that we want to run to_sql on
            kwargs: parameters for pandas.to_sql(**kwargs)
        """
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        empty_df = qc.head(1).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._fold_reduce(1, func)
        # blocking operation
        result.to_pandas()
