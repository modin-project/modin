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

"""
Implement experimental I/O public API.

Here are the functions that, depending on the name, can be added to the general
functions in `pandas.io.py`, or they can replace them in experimental mode.

Examples
--------
  - io_exp.read_sql replace pandas.io.read_sql
  - io_exp.read_csv_glob is added to pandas.io

Notes
-----
  - to use these functions define `MODIN_EXPERIMENTAL=true`
"""

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
    General documentation in `pandas.read_sql`.

    Experimental features
    ---------------------
    Simultaneous reading from a sql file.

    Experimental parameters
    -----------------------
    ...
    partition_column : str
        column used to share the data between the workers (MUST be a INTEGER column)
    lower_bound : int
        the minimum value to be requested from the partition_column
    upper_bound : int
        the maximum value to be requested from the partition_column
    max_sessions : int
        the maximum number of simultaneous connections allowed to use

    Returns
    -------
    Modin DataFrame
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
    sep: str
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
        filepath_or_buffer = kwargs.pop("filepath_or_buffer")
        return _read(filepath_or_buffer, **kwargs)

    return parser_func


def _read(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr], list], **kwargs
) -> DataFrame:
    """
    General documentation in `pandas.read_csv`.

    Experimental features
    ---------------------
    Simultaneous reading from multiple csv files. Works only for local files.

    Parameters
    ----------
    filepath_or_buffer : str, path object, file-like object or list
        The filepath of the csv file.
    kwargs : dict
        Keyword arguments in pandas.read_csv

    Return
    ------
    Modin DataFrame
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
