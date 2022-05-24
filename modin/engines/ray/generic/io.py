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

from numpy import full
import pandas

from modin.engines.base.io import BaseIO
from modin.db_conn import ModinDatabaseConnection


class RayIO(BaseIO):
    @classmethod
    def to_sql(cls, qc, name, con, schema=None, **kwargs):
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

        empty_df = qc.getitem_row_array([0]).to_pandas().head(0)
        empty_df_con = con
        if isinstance(empty_df_con, ModinDatabaseConnection):
            empty_df_con = con.get_connection()
        empty_df.to_sql(name, con=empty_df_con, schema=schema, **kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):
            df.columns = columns
            partition_con = con
            if isinstance(partition_con, ModinDatabaseConnection):
                partition_con = con.get_connection()
                if con.identity_insert:
                    full_table_name = name
                    if schema:
                        full_table_name = f'{schema}.{full_table_name}'
                    partition_con.execute(f'SET IDENTITY_INSERT {full_table_name} ON')                
            df.to_sql(name, con=partition_con, schema=schema, **kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._apply_full_axis(1, func, new_index=[], new_columns=[])
        # blocking operation
        result.to_pandas()
