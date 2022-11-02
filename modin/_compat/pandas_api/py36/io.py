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

"""Module for 'Python 3.6 pandas' compatibility layer for input/output methods."""

import inspect
import pandas
import pathlib
from typing import Union, IO, AnyStr, Optional, Dict, Any, List
from collections import OrderedDict

from modin.utils import _inherit_docstrings, Engine
from modin.logging import enable_logging
from modin.pandas import DataFrame, _update_engine
from ..common.io import _read


@_inherit_docstrings(pandas.read_csv)
@enable_logging
def read_csv(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    sep=",",
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
    _pd_read_csv_signature = {
        val.name for val in inspect.signature(pandas.read_csv).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_csv_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_table)
@enable_logging
def read_table(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    sep="\t",
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
    _pd_read_table_signature = {
        val.name for val in inspect.signature(pandas.read_table).parameters.values()
    }
    _, _, _, f_locals = inspect.getargvalues(inspect.currentframe())
    kwargs = {k: v for k, v in f_locals.items() if k in _pd_read_table_signature}
    return _read(**kwargs)


@_inherit_docstrings(pandas.read_parquet)
@enable_logging
def read_parquet(path, engine: str = "auto", columns=None, **kwargs):
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_parquet(
            path=path,
            engine=engine,
            columns=columns,
            **kwargs,
        )
    )


@_inherit_docstrings(pandas.read_sas, apilink="pandas.read_sas")
@enable_logging
def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
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
        )
    )


@_inherit_docstrings(pandas.read_json)
@enable_logging
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
    nrows: Optional[int] = None,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_json(**kwargs))


@_inherit_docstrings(pandas.read_gbq)
@enable_logging
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
    max_results: Optional[int] = None,
) -> DataFrame:
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_gbq(**kwargs))


@_inherit_docstrings(pandas.read_excel)
@enable_logging
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
    na_filter=True,
):
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


@_inherit_docstrings(pandas.read_feather)
@enable_logging
def read_feather(path, columns=None, use_threads: bool = True):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_feather(**kwargs))


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
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_stata(**kwargs))


@_inherit_docstrings(pandas.read_pickle)
@enable_logging
def read_pickle(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    compression: Optional[str] = "infer",
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_pickle(**kwargs))


@_inherit_docstrings(pandas.read_sql_query)
@enable_logging
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

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_query(**kwargs))


@_inherit_docstrings(pandas.to_pickle)
@enable_logging
def to_pickle(
    obj: Any,
    filepath_or_buffer: Union[str, pathlib.Path],
    compression: Optional[str] = "infer",
    protocol: int = 4,  # older pandas supports only protocol <= 4
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
    )
