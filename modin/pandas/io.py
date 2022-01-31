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

import inspect
import pickle
import pandas
import pandas._libs.lib as lib
import pathlib
import re
from collections import OrderedDict
from pandas._typing import CompressionOptions, StorageOptions
from typing import Union, IO, AnyStr, Sequence, Dict, List, Optional, Any

from modin.error_message import ErrorMessage
from .dataframe import DataFrame
from modin.utils import _inherit_docstrings, Engine
from . import _update_engine

PQ_INDEX_REGEX = re.compile(r"__index_level_\d+__")


# CSV and table


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
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    squeeze = kwargs.pop("squeeze", False)
    pd_obj = FactoryDispatcher.read_csv(**kwargs)
    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    result = DataFrame(query_compiler=pd_obj)
    if squeeze:
        return result.squeeze(axis=1)
    return result


@_inherit_docstrings(pandas.read_csv)
def read_csv(
    filepath_or_buffer: "FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]",
    sep=lib.no_default,
    delimiter=None,
    header="infer",
    names=lib.no_default,
    index_col=None,
    usecols=None,
    squeeze=None,
    prefix=lib.no_default,
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


@_inherit_docstrings(pandas.read_table)
def read_table(
    filepath_or_buffer: "FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]",
    sep=lib.no_default,
    delimiter=None,
    header="infer",
    names=lib.no_default,
    index_col=None,
    usecols=None,
    squeeze=None,
    prefix=lib.no_default,
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
    _pd_read_csv_signature = {
        val.name for val in inspect.signature(pandas.read_csv).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    if f_locals.get("sep", sep) is False or f_locals.get("sep", sep) is lib.no_default:
        f_locals["sep"] = "\t"
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_csv_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_parquet)
def read_parquet(
    path,
    engine: str = "auto",
    columns=None,
    storage_options: StorageOptions = None,
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


@_inherit_docstrings(pandas.read_json)
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
    encoding_errors="strict",
    lines=False,
    chunksize=None,
    compression="infer",
    nrows: Optional[int] = None,
    storage_options: StorageOptions = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_json(**kwargs))


@_inherit_docstrings(pandas.read_gbq)
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
    progress_bar_type: Optional[str] = None,
    max_results: Optional[int] = None,
) -> DataFrame:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_gbq(**kwargs))


@_inherit_docstrings(pandas.read_html)
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

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_html(**kwargs))


@_inherit_docstrings(pandas.read_clipboard)
def read_clipboard(sep=r"\s+", **kwargs):  # pragma: no cover
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_clipboard(**kwargs))


@_inherit_docstrings(pandas.read_excel)
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


@_inherit_docstrings(pandas.read_hdf)
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

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_hdf(**kwargs))


@_inherit_docstrings(pandas.read_feather)
def read_feather(
    path,
    columns=None,
    use_threads: bool = True,
    storage_options: StorageOptions = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_feather(**kwargs))


@_inherit_docstrings(pandas.read_stata)
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
    storage_options: StorageOptions = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_stata(**kwargs))


@_inherit_docstrings(pandas.read_sas)
def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
):  # pragma: no cover
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sas(**kwargs))


@_inherit_docstrings(pandas.read_pickle)
def read_pickle(
    filepath_or_buffer,
    compression: Optional[str] = "infer",
    storage_options: StorageOptions = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_pickle(**kwargs))


@_inherit_docstrings(pandas.read_sql)
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
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    if kwargs.get("chunksize") is not None:
        ErrorMessage.default_to_pandas("Parameters provided [chunksize]")
        df_gen = pandas.read_sql(**kwargs)
        return (
            DataFrame(query_compiler=FactoryDispatcher.from_pandas(df)) for df in df_gen
        )
    return DataFrame(query_compiler=FactoryDispatcher.read_sql(**kwargs))


@_inherit_docstrings(pandas.read_fwf)
def read_fwf(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    colspecs="infer",
    widths=None,
    infer_nrows=100,
    **kwds,
):
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from pandas.io.parsers.base_parser import parser_defaults

    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwds", {}))
    target_kwargs = parser_defaults.copy()
    target_kwargs.update(kwargs)
    pd_obj = FactoryDispatcher.read_fwf(**target_kwargs)
    # When `read_fwf` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    return DataFrame(query_compiler=pd_obj)


@_inherit_docstrings(pandas.read_sql_table)
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

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_table(**kwargs))


@_inherit_docstrings(pandas.read_sql_query)
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


@_inherit_docstrings(pandas.read_spss)
def read_spss(
    path: Union[str, pathlib.Path],
    usecols: Union[Sequence[str], type(None)] = None,
    convert_categoricals: bool = True,
):
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_spss(path, usecols, convert_categoricals)
    )


@_inherit_docstrings(pandas.to_pickle)
def to_pickle(
    obj: Any,
    filepath_or_buffer,
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions = None,
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


@_inherit_docstrings(pandas.json_normalize)
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
    Engine.subscribe(_update_engine)
    return DataFrame(
        pandas.json_normalize(
            data, record_path, meta, meta_prefix, record_prefix, errors, sep, max_level
        )
    )


@_inherit_docstrings(pandas.read_orc)
def read_orc(path, columns: Optional[List[str]] = None, **kwargs) -> DataFrame:
    ErrorMessage.default_to_pandas("read_orc")
    Engine.subscribe(_update_engine)
    return DataFrame(pandas.read_orc(path, columns, **kwargs))


@_inherit_docstrings(pandas.read_xml)
def read_xml(
    path_or_buffer,
    xpath="./*",
    namespaces=None,
    elems_only=False,
    attrs_only=False,
    names=None,
    encoding="utf-8",
    parser="lxml",
    stylesheet=None,
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
            encoding=encoding,
            parser=parser,
            stylesheet=stylesheet,
            compression=compression,
            storage_options=storage_options,
        )
    )


@_inherit_docstrings(pandas.HDFStore)
class HDFStore(pandas.HDFStore):
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
class ExcelFile(pandas.ExcelFile):
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
