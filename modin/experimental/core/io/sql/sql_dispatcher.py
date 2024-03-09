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

"""Module houses `ExperimentalSQLDispatcher` class."""

import warnings

import numpy as np
import pandas

from modin.config import NPartitions
from modin.core.io import SQLDispatcher


class ExperimentalSQLDispatcher(SQLDispatcher):
    """Class handles experimental utils for reading SQL queries or database tables."""

    __read_sql_with_offset = None

    @classmethod
    def preprocess_func(cls):  # noqa: RT01
        """Prepare a function for transmission to remote workers."""
        if cls.__read_sql_with_offset is None:
            # sql deps are optional, so import only when needed
            from modin.experimental.core.io.sql.utils import read_sql_with_offset

            cls.__read_sql_with_offset = cls.put(read_sql_with_offset)
        return cls.__read_sql_with_offset

    @classmethod
    def _read(
        cls,
        sql,
        con,
        index_col,
        coerce_float,
        params,
        parse_dates,
        columns,
        chunksize,
        dtype_backend,
        dtype,
        partition_column,
        lower_bound,
        upper_bound,
        max_sessions,
    ):  # noqa: PR01
        """
        Read SQL query or database table into a DataFrame.

        Documentation for parameters can be found at `modin.read_sql`.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler with imported data for further processing.
        """
        # sql deps are optional, so import only when needed
        from modin.experimental.core.io.sql.utils import get_query_info, is_distributed

        if not is_distributed(partition_column, lower_bound, upper_bound):
            message = "Defaulting to Modin core implementation; \
                'partition_column', 'lower_bound', 'upper_bound' must be different from None"
            warnings.warn(message)
            return cls.base_read(
                sql,
                con,
                index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )
        #  starts the distributed alternative
        cols_names, query = get_query_info(sql, con, partition_column)
        num_parts = min(NPartitions.get(), max_sessions if max_sessions else 1)
        num_splits = min(len(cols_names), num_parts)
        diff = (upper_bound - lower_bound) + 1
        min_size = diff // num_parts
        rest = diff % num_parts
        partition_ids = []
        index_ids = []
        end = lower_bound - 1
        func = cls.preprocess_func()
        for part in range(num_parts):
            if rest:
                size = min_size + 1
                rest -= 1
            else:
                size = min_size
            start = end + 1
            end = start + size - 1
            partition_id = cls.deploy(
                func,
                f_args=(
                    partition_column,
                    start,
                    end,
                    num_splits,
                    query,
                    con,
                    index_col,
                    coerce_float,
                    params,
                    parse_dates,
                    columns,
                    chunksize,
                    dtype_backend,
                    dtype,
                ),
                num_returns=num_splits + 1,
            )
            partition_ids.append(
                [cls.frame_partition_cls(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])
        new_index = pandas.RangeIndex(sum(cls.materialize(index_ids)))
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        )
        new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler
