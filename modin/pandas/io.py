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
import pathlib
import re
from collections import OrderedDict
from typing import Union, IO, AnyStr, Sequence, Dict, List, Optional, Any
from pandas._typing import FilePathOrBuffer

from modin.error_message import ErrorMessage
from .dataframe import DataFrame

PQ_INDEX_REGEX = re.compile(r"__index_level_\d+__")


# Parquet
def read_parquet(path, engine: str = "auto", columns=None, **kwargs):
    """Load a parquet object from the file path, returning a DataFrame.

    Args:
        path: The filepath of the parquet file.
              We only support local files for now.
        engine: This argument doesn't do anything for now.
        kwargs: Pass into parquet's read_pandas function.
    """
    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(
        query_compiler=EngineDispatcher.read_parquet(
            path=path, columns=columns, engine=engine, **kwargs
        )
    )


# CSV and table
def _make_parser_func(sep):
    """Creates a parser function from the given sep.

    Args:
        sep: The separator default to use for the parser.

    Returns:
        A function object.
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
    ):
        _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
        if kwargs.get("sep", sep) is False:
            kwargs["sep"] = "\t"
        return _read(**kwargs)

    return parser_func


def _read(**kwargs):
    """Read csv file from local disk.
    Args:
        filepath_or_buffer:
              The filepath of the csv file.
              We only support local files for now.
        kwargs: Keyword arguments in pandas.read_csv
    """
    from modin.data_management.dispatcher import EngineDispatcher

    pd_obj = EngineDispatcher.read_csv(**kwargs)
    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    return DataFrame(query_compiler=pd_obj)


read_table = _make_parser_func(sep="\t")
read_csv = _make_parser_func(sep=",")


def read_json(
    path_or_buf=None,
    orient=None,
    typ="frame",
    dtype=None,
    convert_axes=None,
    convert_dates=True,
    keep_default_dates=True,
    numpy=False,
    precise_float=False,
    date_unit=None,
    encoding=None,
    lines=False,
    chunksize=None,
    compression="infer",
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_json(**kwargs))


def read_gbq(
    query: str,
    project_id: Optional[str] = None,
    index_col: Optional[str] = None,
    col_order: Optional[List[str]] = None,
    reauth: bool = False,
    auth_local_webserver: bool = False,
    dialect: Optional[str] = None,
    location: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
    credentials=None,
    use_bqstorage_api: Optional[bool] = None,
    private_key=None,
    verbose=None,
    progress_bar_type: Optional[str] = None,
) -> DataFrame:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_gbq(**kwargs))


def read_html(
    io,
    match=".+",
    flavor=None,
    header=None,
    index_col=None,
    skiprows=None,
    attrs=None,
    parse_dates=False,
    thousands=",",
    encoding=None,
    decimal=".",
    converters=None,
    na_values=None,
    keep_default_na=True,
    displayed_only=True,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_html(**kwargs))


def read_clipboard(sep=r"\s+", **kwargs):  # pragma: no cover
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_clipboard(**kwargs))


def read_excel(
    io,
    sheet_name=0,
    header=0,
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    verbose=False,
    parse_dates=False,
    date_parser=None,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    mangle_dupe_cols=True,
    **kwds,
):

    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwds", {}))

    from modin.data_management.dispatcher import EngineDispatcher

    intermediate = EngineDispatcher.read_excel(**kwargs)
    if isinstance(intermediate, (OrderedDict, dict)):
        parsed = type(intermediate)()
        for key in intermediate.keys():
            parsed[key] = DataFrame(query_compiler=intermediate.get(key))
        return parsed
    else:
        return DataFrame(query_compiler=intermediate)


def read_hdf(
    path_or_buf,
    key=None,
    mode: str = "r",
    errors: str = "strict",
    where=None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    columns=None,
    iterator=False,
    chunksize: Optional[int] = None,
    **kwargs,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_hdf(**kwargs))


def read_feather(path, columns=None, use_threads: bool = True):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_feather(**kwargs))


def read_stata(
    filepath_or_buffer,
    convert_dates=True,
    convert_categoricals=True,
    index_col=None,
    convert_missing=False,
    preserve_dtypes=True,
    columns=None,
    order_categoricals=True,
    chunksize=None,
    iterator=False,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_stata(**kwargs))


def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
):  # pragma: no cover
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_sas(**kwargs))


def read_pickle(
    filepath_or_buffer: FilePathOrBuffer, compression: Optional[str] = "infer"
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_pickle(**kwargs))


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
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

    Returns:
        Modin Dataframe
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    if kwargs.get("chunksize") is not None:
        ErrorMessage.default_to_pandas("Parameters provided [chunksize]")
        df_gen = pandas.read_sql(**kwargs)
        return (
            DataFrame(query_compiler=EngineDispatcher.from_pandas(df)) for df in df_gen
        )
    return DataFrame(query_compiler=EngineDispatcher.read_sql(**kwargs))


def read_fwf(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    colspecs="infer",
    widths=None,
    infer_nrows=100,
    **kwds,
):
    from modin.data_management.dispatcher import EngineDispatcher

    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwds", {}))
    pd_obj = EngineDispatcher.read_fwf(**kwargs)
    # When `read_fwf` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    return DataFrame(query_compiler=pd_obj)


def read_sql_table(
    table_name,
    con,
    schema=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    columns=None,
    chunksize=None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_sql_table(**kwargs))


def read_sql_query(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    chunksize=None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(query_compiler=EngineDispatcher.read_sql_query(**kwargs))


def read_spss(
    path: Union[str, pathlib.Path],
    usecols: Union[Sequence[str], type(None)] = None,
    convert_categoricals: bool = True,
):
    from modin.data_management.dispatcher import EngineDispatcher

    return DataFrame(
        query_compiler=EngineDispatcher.read_spss(path, usecols, convert_categoricals)
    )


def to_pickle(
    obj: Any,
    filepath_or_buffer: Union[str, pathlib.Path],
    compression: Optional[str] = "infer",
    protocol: int = 4,
):
    from modin.data_management.dispatcher import EngineDispatcher

    if isinstance(obj, DataFrame):
        obj = obj._query_compiler
    return EngineDispatcher.to_pickle(
        obj, filepath_or_buffer, compression=compression, protocol=protocol
    )


class ExcelFile(pandas.ExcelFile):
    def __getattribute__(self, item):
        default_behaviors = ["__init__", "__class__"]
        method = super(ExcelFile, self).__getattribute__(item)
        if item not in default_behaviors:
            if callable(method):

                def return_handler(*args, **kwargs):
                    """Replaces the default behavior of methods with inplace kwarg.

                    Note: This function will replace all of the arguments passed to
                        methods of ExcelFile with the pandas equivalent. It will convert
                        Modin DataFrame to pandas DataFrame, etc.

                    Returns:
                        A Modin DataFrame in place of a pandas DataFrame, or the same
                        return type as pandas.ExcelFile.
                    """
                    from .utils import to_pandas

                    # We don't want to constantly be giving this error message for
                    # internal methods.
                    if item[0] != "_":
                        ErrorMessage.default_to_pandas("`{}`".format(item))
                    args = [
                        to_pandas(arg) if isinstance(arg, DataFrame) else arg
                        for arg in args
                    ]
                    kwargs = {
                        k: to_pandas(v) if isinstance(v, DataFrame) else v
                        for k, v in kwargs.items()
                    }
                    obj = super(ExcelFile, self).__getattribute__(item)(*args, **kwargs)
                    if isinstance(obj, pandas.DataFrame):
                        return DataFrame(obj)
                    return obj

                # We replace the method with `return_handler` for inplace operations
                method = return_handler
        return method


def json_normalize(
    data: Union[Dict, List[Dict]],
    record_path: Optional[Union[str, List]] = None,
    meta: Optional[Union[str, List[Union[str, List[str]]]]] = None,
    meta_prefix: Optional[str] = None,
    record_prefix: Optional[str] = None,
    errors: Optional[str] = "raise",
    sep: str = ".",
    max_level: Optional[int] = None,
) -> DataFrame:
    ErrorMessage.default_to_pandas("json_normalize")
    return DataFrame(
        pandas.json_normalize(
            data, record_path, meta, meta_prefix, record_prefix, errors, sep, max_level
        )
    )


def read_orc(
    path: FilePathOrBuffer, columns: Optional[List[str]] = None, **kwargs
) -> DataFrame:
    ErrorMessage.default_to_pandas("read_orc")
    return DataFrame(pandas.read_orc(path, columns, **kwargs))


class HDFStore(pandas.HDFStore):
    def __getattribute__(self, item):
        default_behaviors = ["__init__", "__class__"]
        method = super(HDFStore, self).__getattribute__(item)
        if item not in default_behaviors:
            if callable(method):

                def return_handler(*args, **kwargs):
                    """Replaces the default behavior of methods with inplace kwarg.

                    Note: This function will replace all of the arguments passed to
                        methods of HDFStore with the pandas equivalent. It will convert
                        Modin DataFrame to pandas DataFrame, etc. Currently, pytables
                        does not accept Modin DataFrame objects, so we must convert to
                        pandas.

                    Returns:
                        A Modin DataFrame in place of a pandas DataFrame, or the same
                        return type as pandas.HDFStore.
                    """
                    from .utils import to_pandas

                    # We don't want to constantly be giving this error message for
                    # internal methods.
                    if item[0] != "_":
                        ErrorMessage.default_to_pandas("`{}`".format(item))
                    args = [
                        to_pandas(arg) if isinstance(arg, DataFrame) else arg
                        for arg in args
                    ]
                    kwargs = {
                        k: to_pandas(v) if isinstance(v, DataFrame) else v
                        for k, v in kwargs.items()
                    }
                    obj = super(HDFStore, self).__getattribute__(item)(*args, **kwargs)
                    if isinstance(obj, pandas.DataFrame):
                        return DataFrame(obj)
                    return obj

                # We replace the method with `return_handler` for inplace operations
                method = return_handler
        return method
