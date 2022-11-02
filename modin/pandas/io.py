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
import pandas
import pathlib
import re
from typing import Union, IO, AnyStr, Sequence, Dict, List, Optional

from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, enable_logging
from .dataframe import DataFrame
from modin.utils import _inherit_docstrings, Engine
from . import _update_engine

from modin._compat.pandas_api.namespace import (
    read_csv,
    read_parquet,
    read_json,
    read_excel,
    read_html,
    read_feather,
    read_sas,
    read_stata,
    read_pickle,
    read_gbq,
    read_table,
    read_sql_query,
    to_pickle,
    read_xml,
)

PQ_INDEX_REGEX = re.compile(r"__index_level_\d+__")


@_inherit_docstrings(pandas.read_clipboard, apilink="pandas.read_clipboard")
@enable_logging
def read_clipboard(sep=r"\s+", **kwargs):  # pragma: no cover  # noqa: PR01, RT01, D200
    """
    Read text from clipboard and pass to read_csv.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_clipboard(**kwargs))


@_inherit_docstrings(pandas.read_hdf, apilink="pandas.read_hdf")
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

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_hdf(**kwargs))


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
):  # noqa: PR01, RT01, D200
    """
    Read SQL query or database table into a DataFrame.
    """
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


@_inherit_docstrings(pandas.read_fwf, apilink="pandas.read_fwf")
@enable_logging
def read_fwf(
    filepath_or_buffer: Union[str, pathlib.Path, IO[AnyStr]],
    colspecs="infer",
    widths=None,
    infer_nrows=100,
    **kwds,
):  # noqa: PR01, RT01, D200
    """
    Read a table of fixed-width formatted lines into DataFrame.
    """
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from modin._compat.core.base_io import parser_defaults

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
):  # noqa: PR01, RT01, D200
    """
    Read SQL database table into a DataFrame.
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(query_compiler=FactoryDispatcher.read_sql_table(**kwargs))


@_inherit_docstrings(pandas.read_spss, apilink="pandas.read_spss")
@enable_logging
def read_spss(
    path: Union[str, pathlib.Path],
    usecols: Union[Sequence[str], type(None)] = None,
    convert_categoricals: bool = True,
):  # noqa: PR01, RT01, D200
    """
    Load an SPSS file from the file path, returning a DataFrame.
    """
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return DataFrame(
        query_compiler=FactoryDispatcher.read_spss(path, usecols, convert_categoricals)
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
    Engine.subscribe(_update_engine)
    return DataFrame(
        pandas.json_normalize(
            data, record_path, meta, meta_prefix, record_prefix, errors, sep, max_level
        )
    )


@_inherit_docstrings(pandas.read_orc, apilink="pandas.read_orc")
@enable_logging
def read_orc(
    path, columns: Optional[List[str]] = None, **kwargs
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Load an ORC object from the file path, returning a DataFrame.
    """
    ErrorMessage.default_to_pandas("read_orc")
    Engine.subscribe(_update_engine)
    return DataFrame(pandas.read_orc(path, columns, **kwargs))


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
