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

import inspect
import pandas
import ray
from typing import Callable

from . import DataFrame
from modin.data_management.factories.dispatcher import EngineDispatcher
from modin.config import IsExperimental


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
    partition_column=None,
    lower_bound=None,
    upper_bound=None,
    max_sessions=None,
):
    """Read SQL query or database table into a DataFrame.

    Args:
        sql: string or SQLAlchemy Selectable (select or text object) SQL query to be executed or a table name.
        con: SQLAlchemy connectable (engine/connection) or database string URI or DBAPI2 connection (fallback mode)
        index_col: Column(s) to set as index(MultiIndex).
        coerce_float: Attempts to convert values of non-string, non-numeric objects (like decimal.Decimal) to
                      floating point, useful for SQL result sets.
        params: List of parameters to pass to execute method. The syntax used
                to pass parameters is database driver dependent. Check your
                database driver documentation for which of the five syntax styles,
                described in PEP 249's paramstyle, is supported.
        parse_dates:
                     - List of column names to parse as dates.
                     - Dict of ``{column_name: format string}`` where format string is
                       strftime compatible in case of parsing string times, or is one of
                       (D, s, ns, ms, us) in case of parsing integer timestamps.
                     - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
                       to the keyword arguments of :func:`pandas.to_datetime`
                       Especially useful with databases without native Datetime support,
                       such as SQLite.
        columns: List of column names to select from SQL table (only used when reading a table).
        chunksize: If specified, return an iterator where `chunksize` is the number of rows to include in each chunk.
        partition_column: column used to share the data between the workers (MUST be a INTEGER column)
        lower_bound: the minimum value to be requested from the partition_column
        upper_bound: the maximum value to be requested from the partition_column
        max_sessions: the maximum number of simultaneous connections allowed to use

    Returns:
        Pandas Dataframe
    """
    assert IsExperimental.get(), "This only works in experimental mode"
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=EngineDispatcher.read_sql(**kwargs))


@ray.remote
def _get_dataframes(json_string, json_to_dataframes):
    return json_to_dataframes(json_string)


def read_json_row_partitions(
    path: str, split_json_string: Callable, json_to_dataframes: Callable
) -> list:
    """
    Read a JSON file to pandas dataframes by splitting it into json chunks.

    Parameters
    ----------
    path : str
        Path to JSON file.

    split_json_string : Callable
        Function that takes a json string and returns a list of strings.

    json_to_dataframes : Callable
        Function that maps a json string to a list of dataframes. This function
        must return the same number of dataframes for each string produced by
        ``split_json_strings``.

    Notes
    -----
    This function ignores indexes of the constituent dataframes. The resulting
    dataframes will always have the default RangeIndex.
    """
    with open(path, "r") as open_file:
        data = open_file.read()
    futures_per_split = [
        _get_dataframes.remote(json_string, json_to_dataframes)
        for json_string in split_json_string(data)
    ]
    row_partitions_per_split = ray.get(futures_per_split)
    row_partitions_per_dataframe = zip(*row_partitions_per_split)
    return [
        pandas.concat(partitions, copy=False, ignore_index=True)
        for partitions in row_partitions_per_dataframe
    ]
