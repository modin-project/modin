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
Module contains Factories for all of the supported Modin executions.

Factory is a bridge between calls of IO function from high-level API and its
actual implementation in the execution, bound to that factory. Each execution is represented
with a Factory class.
"""

import re
import typing
import warnings

import pandas
from pandas.util._decorators import doc

from modin.config import NativeDataframeMode
from modin.core.io import BaseIO
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler
from modin.utils import get_current_execution

_doc_abstract_factory_class = """
Abstract {role} factory which allows to override the IO module easily.

This class is responsible for dispatching calls of IO-functions to its
actual execution-specific implementations.

Attributes
----------
io_cls : BaseIO
    IO module class of the underlying execution. The place to dispatch calls to.
"""

_doc_factory_class = """
Factory of {execution_name} execution.

This class is responsible for dispatching calls of IO-functions to its
actual execution-specific implementations.

Attributes
----------
io_cls : {execution_name}IO
    IO module class of the underlying execution. The place to dispatch calls to.
"""

_doc_factory_prepare_method = """
Initialize Factory.

Fills in `.io_cls` class attribute with {io_module_name} lazily.
"""

_doc_io_method_raw_template = """
Build query compiler from {source}.

Parameters
----------
{params}

Returns
-------
QueryCompiler
    Query compiler of the selected storage format.
"""

_doc_io_method_template = (
    _doc_io_method_raw_template
    + """
See Also
--------
modin.pandas.{method}
"""
)

_doc_io_method_all_params = """*args : args
    Arguments to pass to the QueryCompiler builder method.
**kwargs : kwargs
    Arguments to pass to the QueryCompiler builder method."""

_doc_io_method_kwargs_params = """**kwargs : kwargs
    Arguments to pass to the QueryCompiler builder method."""


types_dictionary = {"pandas": {"category": pandas.CategoricalDtype}}

supported_executions = (
    "PandasOnRay",
    "PandasOnUnidist",
    "PandasOnDask",
)


class FactoryInfo(typing.NamedTuple):
    """
    Structure that stores information about factory.

    Parameters
    ----------
    engine : str
        Name of underlying execution engine.
    partition : str
        Name of the partition format.
    experimental : bool
        Whether underlying engine is experimental-only.
    """

    engine: str
    partition: str
    experimental: bool


class NotRealFactory(Exception):
    """
    ``NotRealFactory`` exception class.

    Raise when no matching factory could be found.
    """

    pass


@doc(_doc_abstract_factory_class, role="")
class BaseFactory(object):
    io_cls: typing.Type[BaseIO] = None  # The module where the I/O functionality exists.

    @classmethod
    def get_info(cls) -> FactoryInfo:
        """
        Get information about current factory.

        Notes
        -----
        It parses factory name, so it must be conformant with how ``FactoryDispatcher``
        class constructs factory names.
        """
        try:
            experimental, partition, engine = re.match(
                r"^(Experimental)?(.*)On(.*)Factory$", cls.__name__
            ).groups()
        except AttributeError:
            raise NotRealFactory()
        return FactoryInfo(
            engine=engine, partition=partition, experimental=bool(experimental)
        )

    @classmethod
    @doc(
        _doc_factory_prepare_method,
        io_module_name="an underlying execution's IO-module",
    )
    def prepare(cls):
        raise NotImplementedError("Subclasses of BaseFactory must implement prepare")

    @classmethod
    @doc(
        _doc_io_method_template,
        source="pandas DataFrame",
        params="df : pandas.DataFrame",
        method="io.from_pandas",
    )
    def _from_pandas(cls, df):
        if NativeDataframeMode.get() == "Pandas":
            df_copy = df.copy()
            return NativeQueryCompiler(df_copy)
        return cls.io_cls.from_pandas(df)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="Arrow Table",
        params="at : pyarrow.Table",
        method="io.from_arrow",
    )
    def _from_arrow(cls, at):
        return cls.io_cls.from_arrow(at)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a non-pandas object (dict, list, np.array etc...)",
        params=_doc_io_method_all_params,
        method="io.from_non_pandas",
    )
    def _from_non_pandas(cls, *args, **kwargs):
        return cls.io_cls.from_non_pandas(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a DataFrame object supporting exchange protocol `__dataframe__()`",
        params=_doc_io_method_all_params,
        method="io.from_dataframe",
    )
    def _from_dataframe(cls, *args, **kwargs):
        return cls.io_cls.from_dataframe(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a Ray Dataset",
        params="ray_obj : ray.data.Dataset",
        method="modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayIO.from_ray",
    )
    def _from_ray(cls, ray_obj):
        return cls.io_cls.from_ray(ray_obj)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a Dask DataFrame",
        params="dask_obj : dask.dataframe.DataFrame",
        method="modin.core.execution.dask.implementations.pandas_on_dask.io.PandasOnDaskIO.from_dask",
    )
    def _from_dask(cls, dask_obj):
        return cls.io_cls.from_dask(dask_obj)

    @classmethod
    def _from_map(cls, func, iterable, *args, **kwargs):
        """
        Create a Modin `query_compiler` from a map function.

        This method will construct a Modin `query_compiler` split by row partitions.
        The number of row partitions matches the number of elements in the iterable object.

        Parameters
        ----------
        func : callable
            Function to map across the iterable object.
        iterable : Iterable
            An iterable object.
        *args : tuple
            Positional arguments to pass in `func`.
        **kwargs : dict
            Keyword arguments to pass in `func`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data returned by map function.
        """
        return cls.io_cls.from_map(func, iterable, *args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a Parquet file",
        params=_doc_io_method_kwargs_params,
        method="read_parquet",
    )
    def _read_parquet(cls, **kwargs):
        return cls.io_cls.read_parquet(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a CSV file",
        params=_doc_io_method_kwargs_params,
        method="read_csv",
    )
    def _read_csv(cls, **kwargs):
        return cls.io_cls.read_csv(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a JSON file",
        params=_doc_io_method_kwargs_params,
        method="read_json",
    )
    def _read_json(cls, **kwargs):
        return cls.io_cls.read_json(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a Google BigQuery",
        params=_doc_io_method_kwargs_params,
        method="read_gbq",
    )
    def _read_gbq(cls, **kwargs):
        return cls.io_cls.read_gbq(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="an HTML document",
        params=_doc_io_method_kwargs_params,
        method="read_html",
    )
    def _read_html(cls, **kwargs):
        return cls.io_cls.read_html(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="clipboard",
        params=_doc_io_method_kwargs_params,
        method="read_clipboard",
    )
    def _read_clipboard(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_clipboard(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="an Excel file",
        params=_doc_io_method_kwargs_params,
        method="read_excel",
    )
    def _read_excel(cls, **kwargs):
        return cls.io_cls.read_excel(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="an HDFStore",
        params=_doc_io_method_kwargs_params,
        method="read_hdf",
    )
    def _read_hdf(cls, **kwargs):
        return cls.io_cls.read_hdf(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a feather-format object",
        params=_doc_io_method_kwargs_params,
        method="read_feather",
    )
    def _read_feather(cls, **kwargs):
        return cls.io_cls.read_feather(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a Stata file",
        params=_doc_io_method_kwargs_params,
        method="read_stata",
    )
    def _read_stata(cls, **kwargs):
        return cls.io_cls.read_stata(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a SAS file",
        params=_doc_io_method_kwargs_params,
        method="read_sas",
    )
    def _read_sas(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_sas(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a pickled Modin or pandas DataFrame",
        params=_doc_io_method_kwargs_params,
        method="read_pickle",
    )
    def _read_pickle(cls, **kwargs):
        return cls.io_cls.read_pickle(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a SQL query or database table",
        params=_doc_io_method_kwargs_params,
        method="read_sql",
    )
    def _read_sql(cls, **kwargs):
        return cls.io_cls.read_sql(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a table of fixed-width formatted lines",
        params=_doc_io_method_kwargs_params,
        method="read_fwf",
    )
    def _read_fwf(cls, **kwargs):
        return cls.io_cls.read_fwf(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a SQL database table",
        params=_doc_io_method_kwargs_params,
        method="read_sql_table",
    )
    def _read_sql_table(cls, **kwargs):
        return cls.io_cls.read_sql_table(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a SQL query",
        params=_doc_io_method_kwargs_params,
        method="read_sql_query",
    )
    def _read_sql_query(cls, **kwargs):
        return cls.io_cls.read_sql_query(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="an SPSS file",
        params=_doc_io_method_kwargs_params,
        method="read_spss",
    )
    def _read_spss(cls, **kwargs):
        return cls.io_cls.read_spss(**kwargs)

    @classmethod
    def _to_sql(cls, *args, **kwargs):
        """
        Write query compiler content to a SQL database.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
        return cls.io_cls.to_sql(*args, **kwargs)

    @classmethod
    def _to_pickle(cls, *args, **kwargs):
        """
        Pickle query compiler object.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
        return cls.io_cls.to_pickle(*args, **kwargs)

    @classmethod
    def _to_csv(cls, *args, **kwargs):
        """
        Write query compiler content to a CSV file.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        return cls.io_cls.to_csv(*args, **kwargs)

    @classmethod
    def _to_json(cls, *args, **kwargs):
        """
        Write query compiler content to a JSON file.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        return cls.io_cls.to_json(*args, **kwargs)

    @classmethod
    def _to_xml(cls, *args, **kwargs):
        """
        Write query compiler content to a XML file.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        return cls.io_cls.to_xml(*args, **kwargs)

    @classmethod
    def _to_parquet(cls, *args, **kwargs):
        """
        Write query compiler content to a parquet file.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        return cls.io_cls.to_parquet(*args, **kwargs)

    @classmethod
    def _to_ray(cls, modin_obj):
        """
        Write query compiler content to a Ray Dataset.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to write.

        Returns
        -------
        ray.data.Dataset
            A Ray Dataset object.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Ray Dataset if Modin uses a Ray engine.
        """
        return cls.io_cls.to_ray(modin_obj)

    @classmethod
    def _to_dask(cls, modin_obj):
        """
        Write query compiler content to a Dask DataFrame/Series.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to write.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            A Dask DataFrame/Series object.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Dask DataFrame/Series if Modin uses a Dask engine.
        """
        return cls.io_cls.to_dask(modin_obj)

    # experimental methods that don't exist in pandas
    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="CSV files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_csv_glob(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_csv_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_csv_glob(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Pickle files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_pickle_glob(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_pickle_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_pickle_glob(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="SQL files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_sql_distributed(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            extra_parameters = (
                "partition_column",
                "lower_bound",
                "upper_bound",
                "max_sessions",
            )
            if any(
                param in kwargs and kwargs[param] is not None
                for param in extra_parameters
            ):
                warnings.warn(
                    f"Distributed read_sql() was only implemented for {', '.join(supported_executions)} executions."
                )
            for param in extra_parameters:
                del kwargs[param]
            return cls.io_cls.read_sql(**kwargs)
        return cls.io_cls.read_sql_distributed(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Custom text files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_custom_text(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_custom_text()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_custom_text(**kwargs)

    @classmethod
    def _to_pickle_glob(cls, *args, **kwargs):
        """
        Distributed pickle query compiler object.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_to_pickle_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.to_pickle_glob(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Parquet files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_parquet_glob(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_parquet_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_parquet_glob(**kwargs)

    @classmethod
    def _to_parquet_glob(cls, *args, **kwargs):
        """
        Write query compiler content to several parquet files.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_to_parquet_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.to_parquet_glob(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Json files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_json_glob(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_json_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_json_glob(**kwargs)

    @classmethod
    def _to_json_glob(cls, *args, **kwargs):
        """
        Write query compiler content to several json files.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_to_json_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.to_json_glob(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="XML files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_xml_glob(cls, **kwargs):
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_read_xml_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.read_xml_glob(**kwargs)

    @classmethod
    def _to_xml_glob(cls, *args, **kwargs):
        """
        Write query compiler content to several XML files.

        Parameters
        ----------
        *args : args
            Arguments to pass to the writer method.
        **kwargs : kwargs
            Arguments to pass to the writer method.
        """
        current_execution = get_current_execution()
        if current_execution not in supported_executions:
            raise NotImplementedError(
                f"`_to_xml_glob()` is not implemented for {current_execution} execution."
            )
        return cls.io_cls.to_xml_glob(*args, **kwargs)


@doc(_doc_factory_class, execution_name="PandasOnRay")
class PandasOnRayFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``PandasOnRayIO``")
    def prepare(cls):
        from modin.core.execution.ray.implementations.pandas_on_ray.io import (
            PandasOnRayIO,
        )

        cls.io_cls = PandasOnRayIO


@doc(_doc_factory_class, execution_name="PandasOnPython")
class PandasOnPythonFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``PandasOnPythonIO``")
    def prepare(cls):
        from modin.core.execution.python.implementations.pandas_on_python.io import (
            PandasOnPythonIO,
        )

        cls.io_cls = PandasOnPythonIO


@doc(_doc_factory_class, execution_name="PandasOnDask")
class PandasOnDaskFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``PandasOnDaskIO``")
    def prepare(cls):
        from modin.core.execution.dask.implementations.pandas_on_dask.io import (
            PandasOnDaskIO,
        )

        cls.io_cls = PandasOnDaskIO


@doc(_doc_factory_class, execution_name="PandasOnUnidist")
class PandasOnUnidistFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``PandasOnUnidistIO``")
    def prepare(cls):
        from modin.core.execution.unidist.implementations.pandas_on_unidist.io import (
            PandasOnUnidistIO,
        )

        cls.io_cls = PandasOnUnidistIO
