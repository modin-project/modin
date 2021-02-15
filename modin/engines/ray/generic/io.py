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

import io
import pandas

from modin.engines.base.io import BaseIO
from ray.util.queue import Queue


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

        empty_df = qc.getitem_row_array([0]).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._apply_full_axis(1, func, new_index=[], new_columns=[])
        # blocking operation
        result.to_pandas()

    @staticmethod
    def _to_csv_check_support(kwargs):
        path_or_buf = kwargs["path_or_buf"]
        compression = kwargs["compression"]
        if not isinstance(path_or_buf, str):
            return False

        # case when the pointer is placed at the beginning of the file.
        if "r" in kwargs["mode"] and "+" in kwargs["mode"]:
            return False

        if compression is None or not compression == "infer":
            return False

        if any((path_or_buf.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False

        return True

    @classmethod
    def to_csv(cls, qc, **kwargs):
        if not cls._to_csv_check_support(kwargs):
            return BaseIO.to_csv(qc, **kwargs)

        # The partition id will be added to the queue, for which the moment
        # of writing to the file has come
        queue = Queue(maxsize=1)

        def func(df, **kw):
            if kw["partition_idx"] != 0:
                if "w" in kwargs["mode"]:
                    kwargs["mode"] = kwargs["mode"].replace("w", "a")
                kwargs["header"] = False

            path_or_buf = kwargs["path_or_buf"]
            if "b" in kwargs["mode"]:
                kwargs["path_or_buf"] = io.BytesIO()
            else:
                kwargs["path_or_buf"] = io.StringIO()

            df.to_csv(**kwargs)
            content = kwargs["path_or_buf"].getvalue()

            while True:
                get_value = queue.get(block=True)
                if get_value == kw["partition_idx"]:
                    break
                queue.put(get_value)

            with open(path_or_buf, mode=kwargs["mode"]) as _f:
                _f.write(content)

            queue.put(get_value + 1)

            return pandas.DataFrame()

        queue.put(0)
        result = qc._modin_frame.broadcast_apply_full_axis(
            1,
            func,
            other=None,
            new_index=[],
            new_columns=[],
            enumerate_partitions=True,
        )
        # blocking operation
        result.to_pandas()
