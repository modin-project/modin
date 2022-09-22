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

"""Module for 'latest pandas' compatibility layer for input/output methods."""

import inspect
import pandas
from pandas._libs.lib import no_default
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    ReadCsvBuffer,
    FilePath,
    StorageOptions,
    IntStrT,
)
import pickle
from typing import (
    Optional,
    Dict,
    Any,
    OrderedDict,
    List,
    Sequence,
    Literal,
    Hashable,
    Callable,
    Iterable,
)

from modin.utils import _inherit_docstrings, Engine
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from ..common.io import _read
from modin.pandas import DataFrame, _update_engine


@_inherit_docstrings(pandas.read_xml, apilink="pandas.read_xml")
@enable_logging
def read_xml(
    path_or_buffer,
    xpath="./*",
    namespaces=None,
    elems_only=False,
    attrs_only=False,
    names=None,
    dtype=None,
    converters=None,
    parse_dates=None,
    encoding="utf-8",
    parser="lxml",
    stylesheet=None,
    iterparse=None,
    compression="infer",
    storage_options=None,
) -> DataFrame:
    ErrorMessage.default_to_pandas("read_xml")
    Engine.subscribe(_update_engine)
    return DataFrame(
        pandas.read_xml(
            path_or_buffer,
            xpath=xpath,
            namespaces=namespaces,
            elems_only=elems_only,
            attrs_only=attrs_only,
            names=names,
            dtype=dtype,
            converters=converters,
            parse_dates=parse_dates,
            encoding=encoding,
            parser=parser,
            stylesheet=stylesheet,
            iterparse=iterparse,
            compression=compression,
            storage_options=storage_options,
        )
    )


@_inherit_docstrings(pandas.read_csv, apilink="pandas.read_csv")
@enable_logging
def read_csv(
    filepath_or_buffer: "FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]",
    sep=no_default,
    delimiter=None,
    header="infer",
    names=no_default,
    index_col=None,
    usecols=None,
    squeeze=None,
    prefix=no_default,
    mangle_dupe_cols=True,
    dtype: "DtypeArg | None" = None,
    engine: "CSVEngine | None" = None,
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
    parse_dates=None,
    infer_datetime_format=False,
    keep_date_col=False,
    date_parser=None,
    dayfirst=False,
    cache_dates=True,
    iterator=False,
    chunksize=None,
    compression: "CompressionOptions" = "infer",
    thousands=None,
    decimal: "str" = ".",
    lineterminator=None,
    quotechar='"',
    quoting=0,
    escapechar=None,
    comment=None,
    encoding=None,
    encoding_errors: "str | None" = "strict",
    dialect=None,
    error_bad_lines=None,
    warn_bad_lines=None,
    on_bad_lines=None,
    skipfooter=0,
    doublequote=True,
    delim_whitespace=False,
    low_memory=True,
    memory_map=False,
    float_precision=None,
    storage_options: "StorageOptions" = None,
):
    # ISSUE #2408: parse parameter shared with pandas read_csv and read_table and update with provided args
    _pd_read_csv_signature = {
        val.name for val in inspect.signature(pandas.read_csv).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_csv_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_html, apilink="pandas.read_html")
@enable_logging
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
    extract_links=None,
):  # noqa: PR01, RT01, D200
    """
    Read HTML tables into a ``DataFrame`` object.
    """
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_html(
            io,
            match=match,
            flavor=flavor,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            attrs=attrs,
            parse_dates=parse_dates,
            thousands=thousands,
            encoding=encoding,
            decimal=decimal,
            converters=converters,
            na_values=na_values,
            keep_default_na=keep_default_na,
            displayed_only=displayed_only,
            extract_links=extract_links,
        )
    )


@_inherit_docstrings(pandas.read_table, apilink="pandas.read_table")
@enable_logging
def read_table(
    filepath_or_buffer: "FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]",
    sep=no_default,
    delimiter=None,
    header="infer",
    names=no_default,
    index_col=None,
    usecols=None,
    squeeze=None,
    prefix=no_default,
    mangle_dupe_cols=True,
    dtype: "DtypeArg | None" = None,
    engine: "CSVEngine | None" = None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace=False,
    skiprows=None,
    skipfooter=0,
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
    compression: "CompressionOptions" = "infer",
    thousands=None,
    decimal: "str" = ".",
    lineterminator=None,
    quotechar='"',
    quoting=0,
    doublequote=True,
    escapechar=None,
    comment=None,
    encoding=None,
    encoding_errors: "str | None" = "strict",
    dialect=None,
    error_bad_lines=None,
    warn_bad_lines=None,
    on_bad_lines=None,
    delim_whitespace=False,
    low_memory=True,
    memory_map=False,
    float_precision=None,
    storage_options: "StorageOptions" = None,
):
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
@enable_logging
def read_parquet(
    path,
    engine: str = "auto",
    columns=None,
    storage_options: "StorageOptions" = None,
    use_nullable_dtypes: bool = False,
    **kwargs,
):
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_parquet(
            path=path,
            engine=engine,
            columns=columns,
            storage_options=storage_options,
            use_nullable_dtypes=use_nullable_dtypes,
            **kwargs,
        )
    )


@_inherit_docstrings(pandas.read_json, apilink="pandas.read_json")
@enable_logging
def read_json(
    path_or_buf,
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
    encoding_errors="strict",
    lines=False,
    chunksize=None,
    compression="infer",
    nrows: Optional[int] = None,
    storage_options: "StorageOptions" = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_json(**kwargs))


@_inherit_docstrings(pandas.read_gbq, apilink="pandas.read_gbq")
@enable_logging
def read_gbq(
    query: str,
    project_id: Optional[str] = None,
    index_col: Optional[str] = None,
    col_order: Optional[List[str]] = None,
    reauth: bool = False,
    auth_local_webserver: bool = True,
    dialect: Optional[str] = None,
    location: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
    credentials=None,
    use_bqstorage_api: Optional[bool] = None,
    progress_bar_type: Optional[str] = None,
    max_results: Optional[int] = None,
) -> DataFrame:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_gbq(**kwargs))


@_inherit_docstrings(pandas.read_excel, apilink="pandas.read_excel")
@enable_logging
def read_excel(
    io,
    sheet_name: "str | int | list[IntStrT] | None" = 0,
    header: "int | Sequence[int] | None" = 0,
    names=None,
    index_col: "int | Sequence[int] | None" = None,
    usecols=None,
    squeeze: "bool | None" = None,
    dtype: "DtypeArg | None" = None,
    engine: "Literal[('xlrd', 'openpyxl', 'odf', 'pyxlsb')] | None" = None,
    converters=None,
    true_values: "Iterable[Hashable] | None" = None,
    false_values: "Iterable[Hashable] | None" = None,
    skiprows: "Sequence[int] | int | Callable[[int], object] | None" = None,
    nrows: "int | None" = None,
    na_values=None,
    keep_default_na: "bool" = True,
    na_filter: "bool" = True,
    verbose: "bool" = False,
    parse_dates=False,
    date_parser=None,
    thousands: "str | None" = None,
    decimal: "str" = ".",
    comment: "str | None" = None,
    skipfooter: "int" = 0,
    convert_float: "bool | None" = None,
    mangle_dupe_cols: "bool" = True,
    storage_options: "StorageOptions" = None,
) -> "DataFrame | dict[IntStrT, DataFrame]":
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    intermediate = FactoryDispatcher.read_excel(**kwargs)
    if isinstance(intermediate, (OrderedDict, dict)):
        parsed = type(intermediate)()
        for key in intermediate.keys():
            parsed[key] = DataFrame(query_compiler=intermediate.get(key))
        return parsed
    else:
        return DataFrame(query_compiler=intermediate)


@_inherit_docstrings(pandas.read_feather, apilink="pandas.read_feather")
@enable_logging
def read_feather(
    path,
    columns=None,
    use_threads: bool = True,
    storage_options: "StorageOptions" = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_feather(**kwargs))


@_inherit_docstrings(pandas.read_sas, apilink="pandas.read_sas")
@enable_logging
def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
    compression="infer",
):  # pragma: no cover  # noqa: PR01, RT01, D200
    """
    Read SAS files stored as either XPORT or SAS7BDAT format files.
    """
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_sas(
            filepath_or_buffer,
            format=format,
            index=index,
            encoding=encoding,
            chunksize=chunksize,
            iterator=iterator,
            compression=compression,
        )
    )


@_inherit_docstrings(pandas.read_stata)
@enable_logging
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
    compression="infer",
    storage_options: "StorageOptions" = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_stata(**kwargs))


@_inherit_docstrings(pandas.read_pickle, apilink="pandas.read_pickle")
@enable_logging
def read_pickle(
    filepath_or_buffer,
    compression: Optional[str] = "infer",
    storage_options: "StorageOptions" = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_pickle(**kwargs))


@_inherit_docstrings(pandas.read_sql_query, apilink="pandas.read_sql_query")
@enable_logging
def read_sql_query(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    chunksize=None,
    dtype=None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_query(**kwargs))


@_inherit_docstrings(pandas.to_pickle)
@enable_logging
def to_pickle(
    obj: Any,
    filepath_or_buffer,
    compression: "CompressionOptions" = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: "StorageOptions" = None,
):
    Engine.subscribe(_update_engine)
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
