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

"""Implement experimental I/O public API."""

import inspect
import pathlib
from typing import Union, IO, AnyStr, Callable, Optional

import pandas

from . import DataFrame
from modin.config import IsExperimental, Engine
from modin.data_management.factories.dispatcher import EngineDispatcher
from ...pandas import _update_engine


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
    partition_column: Optional[str] = None,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    max_sessions: Optional[int] = None,
) -> DataFrame:
    """
    General documentation is available in `modin.pandas.read_sql`.

    This experimental feature provides distributed reading from a sql file.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the SQLAlchemy
        connectable; str connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.
    index_col : str or list of str, optional
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default: True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional
        List of parameters to pass to execute method. The syntax used to pass
        parameters is database driver dependent. Check your database driver
        documentation for which of the five syntax styles, described in PEP 249’s
        paramstyle, is supported. Eg. for psycopg2, uses %(name)s so use params=
        {‘name’ : ‘value’}.
    parse_dates : list or dict, optional
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, optional
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, optional
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    partition_column : str, optional
        Column used to share the data between the workers (MUST be a INTEGER column).
    lower_bound : int, optional
        The minimum value to be requested from the partition_column.
    upper_bound : int, optional
        The maximum value to be requested from the partition_column.
    max_sessions : int, optional
        The maximum number of simultaneous connections allowed to use.

    Returns
    -------
    modin.DataFrame
    """
    Engine.subscribe(_update_engine)
    assert IsExperimental.get(), "This only works in experimental mode"
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=EngineDispatcher.read_sql(**kwargs))


# CSV and table
def _make_parser_func(sep: str) -> Callable:
    """
    Create a parser function from the given sep.

    Parameters
    ----------
    sep : str
        The separator default to use for the parser.

    Returns
    -------
    Callable
    """

    def parser_func(
        filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
        sep=sep,
        delimiter=None,
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=None,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        cache_dates=True,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal: str = ".",
        lineterminator=None,
        quotechar='"',
        quoting=0,
        escapechar=None,
        comment=None,
        encoding=None,
        dialect=None,
        error_bad_lines=True,
        warn_bad_lines=True,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
    ) -> DataFrame:
        # ISSUE #2408: parse parameter shared with pandas read_csv and read_table and update with provided args
        _pd_read_csv_signature = {
            val.name for val in inspect.signature(pandas.read_csv).parameters.values()
        }
        _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
        if f_locals.get("sep", sep) is False:
            f_locals["sep"] = "\t"

        kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_csv_signature}
        return _read(**kwargs)

    return parser_func


def _read(**kwargs) -> DataFrame:
    """
    General documentation is available in `modin.pandas.read_csv`.

    This experimental feature provides parallel reading from multiple csv files which are
    defined by glob pattern. Works for local files only!

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments in `modin.pandas.read_csv`.

    Returns
    -------
    modin.DataFrame
    """
    from modin.data_management.factories.dispatcher import EngineDispatcher

    Engine.subscribe(_update_engine)

    try:
        pd_obj = EngineDispatcher.read_csv_glob(**kwargs)
    except AttributeError:
        raise AttributeError("read_csv_glob() is only implemented for pandas on Ray.")

    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj

    return DataFrame(query_compiler=pd_obj)


read_csv_glob = _make_parser_func(sep=",")
read_csv_glob.__doc__ = _read.__doc__
