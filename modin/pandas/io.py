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
Implement I/O public API as pandas does.

Almost all docstrings for public and magic methods should be inherited from pandas
for better maintability.
Manually add documentation for methods which are not presented in pandas.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
import inspect
import pandas
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.readers import _c_parser_defaults
from pandas._libs.lib import no_default, NoDefault
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    ReadCsvBuffer,
    FilePath,
    StorageOptions,
    IntStrT,
    ReadBuffer,
    IndexLabel,
    ConvertersArg,
    ParseDatesArg,
    XMLParsers,
    DtypeBackend,
)
import pathlib
import pickle
from typing import (
    Union,
    IO,
    AnyStr,
    Sequence,
    Dict,
    List,
    Optional,
    Any,
    Literal,
    Hashable,
    Callable,
    Iterable,
    Pattern,
    Iterator,
)

from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, enable_logging
from .dataframe import DataFrame
from .series import Series
from modin.utils import _inherit_docstrings, expanduser_path_arg


def _read(**kwargs):
    """
    Read csv file from local disk.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments in pandas.read_csv.

    Returns
    -------
    modin.pandas.DataFrame
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    squeeze = kwargs.pop("squeeze", False)
    pd_obj = FactoryDispatcher.read_csv(**kwargs)
    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    result = DataFrame(query_compiler=pd_obj)
    if squeeze:
        return result.squeeze(axis=1)
    return result


@_inherit_docstrings(pandas.read_xml, apilink="pandas.read_xml")
@expanduser_path_arg("path_or_buffer")
@enable_logging
def read_xml(
    path_or_buffer: FilePath | ReadBuffer[bytes] | ReadBuffer[str],
    *,
    xpath: str = "./*",
    namespaces: dict[str, str] | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Sequence[str] | None = None,
    dtype: DtypeArg | None = None,
    converters: ConvertersArg | None = None,
    parse_dates: ParseDatesArg | None = None,
    encoding: str | None = "utf-8",
    parser: XMLParsers = "lxml",
    stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> DataFrame:
    ErrorMessage.default_to_pandas("read_xml")
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(pandas.read_xml(**kwargs))


@_inherit_docstrings(pandas.read_csv, apilink="pandas.read_csv")
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | NoDefault = no_default,
    delimiter: str | None | NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | NoDefault = no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols=None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace: bool = False,
    skiprows=None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates=None,
    infer_datetime_format: bool = no_default,
    keep_date_col: bool = False,
    date_parser=no_default,
    date_format=None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines="error",
    # Internal
    delim_whitespace: bool = False,
    low_memory=_c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: Literal["high", "legacy"] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> DataFrame | TextFileReader:
    # ISSUE #2408: parse parameter shared with pandas read_csv and read_table and update with provided args
    _pd_read_csv_signature = {
        val.name for val in inspect.signature(pandas.read_csv).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_csv_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_table, apilink="pandas.read_table")
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | NoDefault = no_default,
    delimiter: str | None | NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | NoDefault = no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols=None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace: bool = False,
    skiprows=None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates=False,
    infer_datetime_format: bool = no_default,
    keep_date_col: bool = False,
    date_parser=no_default,
    date_format: str = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines="error",
    # Internal
    delim_whitespace=False,
    low_memory=_c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: str | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> DataFrame | TextFileReader:
    # ISSUE #2408: parse parameter shared with pandas read_csv and read_table and update with provided args
    _pd_read_table_signature = {
        val.name for val in inspect.signature(pandas.read_table).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    if f_locals.get("sep", sep) is False or f_locals.get("sep", sep) is no_default:
        f_locals["sep"] = "\t"
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_table_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_parquet, apilink="pandas.read_parquet")
@expanduser_path_arg("path")
@enable_logging
def read_parquet(
    path,
    engine: str = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    use_nullable_dtypes: bool = no_default,
    dtype_backend=no_default,
    **kwargs,
) -> DataFrame:
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    if engine == "fastparquet" and dtype_backend is not no_default:
        raise ValueError(
            "The 'dtype_backend' argument is not supported for the fastparquet engine"
        )

    return DataFrame(
        query_compiler=FactoryDispatcher.read_parquet(
            path=path,
            engine=engine,
            columns=columns,
            storage_options=storage_options,
            use_nullable_dtypes=use_nullable_dtypes,
            dtype_backend=dtype_backend,
            **kwargs,
        )
    )


@_inherit_docstrings(pandas.read_json, apilink="pandas.read_json")
@expanduser_path_arg("path_or_buf")
@enable_logging
def read_json(
    path_or_buf,
    *,
    orient: str | None = None,
    typ: Literal["frame", "series"] = "frame",
    dtype: DtypeArg | None = None,
    convert_axes=None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    lines: bool = False,
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
    engine="ujson",
) -> DataFrame | Series | pandas.io.json._json.JsonReader:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_json(**kwargs))


@_inherit_docstrings(pandas.read_gbq, apilink="pandas.read_gbq")
@enable_logging
def read_gbq(
    query: str,
    project_id: str | None = None,
    index_col: str | None = None,
    col_order: list[str] | None = None,
    reauth: bool = False,
    auth_local_webserver: bool = True,
    dialect: str | None = None,
    location: str | None = None,
    configuration: dict[str, Any] | None = None,
    credentials=None,
    use_bqstorage_api: bool | None = None,
    max_results: int | None = None,
    progress_bar_type: str | None = None,
) -> DataFrame:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_gbq(**kwargs))


@_inherit_docstrings(pandas.read_html, apilink="pandas.read_html")
@expanduser_path_arg("io")
@enable_logging
def read_html(
    io,
    *,
    match: str | Pattern = ".+",
    flavor: str | None = None,
    header: int | Sequence[int] | None = None,
    index_col: int | Sequence[int] | None = None,
    skiprows: int | Sequence[int] | slice | None = None,
    attrs: dict[str, str] | None = None,
    parse_dates: bool = False,
    thousands: str | None = ",",
    encoding: str | None = None,
    decimal: str = ".",
    converters: dict | None = None,
    na_values: Iterable[object] | None = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    extract_links: Literal[None, "header", "footer", "body", "all"] = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> list[DataFrame]:  # noqa: PR01, RT01, D200
    """
    Read HTML tables into a ``DataFrame`` object.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    qcs = FactoryDispatcher.read_html(**kwargs)
    return [DataFrame(query_compiler=qc) for qc in qcs]


@_inherit_docstrings(pandas.read_clipboard, apilink="pandas.read_clipboard")
@enable_logging
def read_clipboard(
    sep=r"\s+",
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
    **kwargs,
):  # pragma: no cover  # noqa: PR01, RT01, D200
    """
    Read text from clipboard and pass to read_csv.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_clipboard(**kwargs))


@_inherit_docstrings(pandas.read_excel, apilink="pandas.read_excel")
@expanduser_path_arg("io")
@enable_logging
def read_excel(
    io,
    sheet_name: str | int | list[IntStrT] | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: list[str] | None = None,
    index_col: int | Sequence[int] | None = None,
    usecols: int
    | str
    | Sequence[int]
    | Sequence[str]
    | Callable[[str], bool]
    | None = None,
    dtype: DtypeArg | None = None,
    engine: Literal[("xlrd", "openpyxl", "odf", "pyxlsb")] | None = None,
    converters: dict[str, Callable] | dict[int, Callable] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
    nrows: int | None = None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: list | dict | bool = False,
    date_parser: Union[Callable, NoDefault] = no_default,
    date_format=None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> DataFrame | dict[IntStrT, DataFrame]:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    intermediate = FactoryDispatcher.read_excel(**kwargs)
    if isinstance(intermediate, (OrderedDict, dict)):
        parsed = type(intermediate)()
        for key in intermediate.keys():
            parsed[key] = DataFrame(query_compiler=intermediate.get(key))
        return parsed
    else:
        return DataFrame(query_compiler=intermediate)


@_inherit_docstrings(pandas.read_hdf, apilink="pandas.read_hdf")
@expanduser_path_arg("path_or_buf")
@enable_logging
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
):  # noqa: PR01, RT01, D200
    """
    Read data from the store into DataFrame.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_hdf(**kwargs))


@_inherit_docstrings(pandas.read_feather, apilink="pandas.read_feather")
@expanduser_path_arg("path")
@enable_logging
def read_feather(
    path,
    columns: Sequence[Hashable] | None = None,
    use_threads: bool = True,
    storage_options: StorageOptions = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_feather(**kwargs))


@_inherit_docstrings(pandas.read_stata)
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_stata(
    filepath_or_buffer,
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Sequence[str] | None = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
) -> DataFrame | pandas.io.stata.StataReader:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_stata(**kwargs))


@_inherit_docstrings(pandas.read_sas, apilink="pandas.read_sas")
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_sas(
    filepath_or_buffer,
    *,
    format: str | None = None,
    index: Hashable | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
) -> DataFrame | pandas.io.sas.sasreader.ReaderBase:  # noqa: PR01, RT01, D200
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_sas(
            filepath_or_buffer=filepath_or_buffer,
            format=format,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            iterator=iterator,
            compression=compression,
        )
    )


@_inherit_docstrings(pandas.read_pickle, apilink="pandas.read_pickle")
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_pickle(
    filepath_or_buffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_pickle(**kwargs))


@_inherit_docstrings(pandas.read_sql, apilink="pandas.read_sql")
@enable_logging
def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
    dtype=None,
):  # noqa: PR01, RT01, D200
    """
    Read SQL query or database table into a DataFrame.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    if kwargs.get("chunksize") is not None:
        ErrorMessage.default_to_pandas("Parameters provided [chunksize]")
        df_gen = pandas.read_sql(**kwargs)
        return (
            DataFrame(query_compiler=FactoryDispatcher.from_pandas(df)) for df in df_gen
        )
    return DataFrame(query_compiler=FactoryDispatcher.read_sql(**kwargs))


@_inherit_docstrings(pandas.read_fwf, apilink="pandas.read_fwf")
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def read_fwf(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    *,
    colspecs="infer",
    widths=None,
    infer_nrows=100,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
    **kwds,
):  # noqa: PR01, RT01, D200
    """
    Read a table of fixed-width formatted lines into DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from pandas.io.parsers.base_parser import parser_defaults

    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwds", {}))
    target_kwargs = parser_defaults.copy()
    target_kwargs.update(kwargs)
    pd_obj = FactoryDispatcher.read_fwf(**target_kwargs)
    # When `read_fwf` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    return DataFrame(query_compiler=pd_obj)


@_inherit_docstrings(pandas.read_sql_table, apilink="pandas.read_sql_table")
@enable_logging
def read_sql_table(
    table_name,
    con,
    schema=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    columns=None,
    chunksize=None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
):  # noqa: PR01, RT01, D200
    """
    Read SQL database table into a DataFrame.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_table(**kwargs))


@_inherit_docstrings(pandas.read_sql_query, apilink="pandas.read_sql_query")
@enable_logging
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list[str] | dict[str, str] | None = None,
    parse_dates: list[str] | dict[str, str] | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> DataFrame | Iterator[DataFrame]:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_query(**kwargs))


@_inherit_docstrings(pandas.to_pickle)
@expanduser_path_arg("filepath_or_buffer")
@enable_logging
def to_pickle(
    obj: Any,
    filepath_or_buffer,
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions = None,
) -> None:
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    if isinstance(obj, DataFrame):
        obj = obj._query_compiler
    return FactoryDispatcher.to_pickle(
        obj,
        filepath_or_buffer=filepath_or_buffer,
        compression=compression,
        protocol=protocol,
        storage_options=storage_options,
    )


@_inherit_docstrings(pandas.read_spss, apilink="pandas.read_spss")
@expanduser_path_arg("path")
@enable_logging
def read_spss(
    path: Union[str, pathlib.Path],
    usecols: Optional[Sequence[str]] = None,
    convert_categoricals: bool = True,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
):  # noqa: PR01, RT01, D200
    """
    Load an SPSS file from the file path, returning a DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_spss(
            path=path,
            usecols=usecols,
            convert_categoricals=convert_categoricals,
            dtype_backend=dtype_backend,
        )
    )


@_inherit_docstrings(pandas.json_normalize, apilink="pandas.json_normalize")
@enable_logging
def json_normalize(
    data: Union[Dict, List[Dict]],
    record_path: Optional[Union[str, List]] = None,
    meta: Optional[Union[str, List[Union[str, List[str]]]]] = None,
    meta_prefix: Optional[str] = None,
    record_prefix: Optional[str] = None,
    errors: Optional[str] = "raise",
    sep: str = ".",
    max_level: Optional[int] = None,
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Normalize semi-structured JSON data into a flat table.
    """
    ErrorMessage.default_to_pandas("json_normalize")
    return DataFrame(
        pandas.json_normalize(
            data, record_path, meta, meta_prefix, record_prefix, errors, sep, max_level
        )
    )


@_inherit_docstrings(pandas.read_orc, apilink="pandas.read_orc")
@expanduser_path_arg("path")
@enable_logging
def read_orc(
    path,
    columns: Optional[List[str]] = None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
    **kwargs,
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Load an ORC object from the file path, returning a DataFrame.
    """
    ErrorMessage.default_to_pandas("read_orc")
    return DataFrame(
        pandas.read_orc(path, columns=columns, dtype_backend=dtype_backend, **kwargs)
    )


@_inherit_docstrings(pandas.HDFStore)
class HDFStore(ClassLogger, pandas.HDFStore):  # noqa: PR01, D200
    """
    Dict-like IO interface for storing pandas objects in PyTables.
    """

    _return_modin_dataframe = True

    def __getattribute__(self, item):
        default_behaviors = ["__init__", "__class__"]
        method = super(HDFStore, self).__getattribute__(item)
        if item not in default_behaviors:
            if callable(method):

                def return_handler(*args, **kwargs):
                    """
                    Replace the default behavior of methods with inplace kwarg.

                    Returns
                    -------
                    A Modin DataFrame in place of a pandas DataFrame, or the same
                    return type as pandas.HDFStore.

                    Notes
                    -----
                    This function will replace all of the arguments passed to
                    methods of HDFStore with the pandas equivalent. It will convert
                    Modin DataFrame to pandas DataFrame, etc. Currently, pytables
                    does not accept Modin DataFrame objects, so we must convert to
                    pandas.
                    """
                    from modin.utils import to_pandas

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
                    if self._return_modin_dataframe and isinstance(
                        obj, pandas.DataFrame
                    ):
                        return DataFrame(obj)
                    return obj

                # We replace the method with `return_handler` for inplace operations
                method = return_handler
        return method


@_inherit_docstrings(pandas.ExcelFile)
class ExcelFile(ClassLogger, pandas.ExcelFile):  # noqa: PR01, D200
    """
    Class for parsing tabular excel sheets into DataFrame objects.
    """

    def __getattribute__(self, item):
        default_behaviors = ["__init__", "__class__"]
        method = super(ExcelFile, self).__getattribute__(item)
        if item not in default_behaviors:
            if callable(method):

                def return_handler(*args, **kwargs):
                    """
                    Replace the default behavior of methods with inplace kwarg.

                    Returns
                    -------
                    A Modin DataFrame in place of a pandas DataFrame, or the same
                    return type as pandas.ExcelFile.

                    Notes
                    -----
                    This function will replace all of the arguments passed to
                    methods of ExcelFile with the pandas equivalent. It will convert
                    Modin DataFrame to pandas DataFrame, etc.
                    """
                    from modin.utils import to_pandas

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


__all__ = [
    "ExcelFile",
    "HDFStore",
    "json_normalize",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_pickle",
]
