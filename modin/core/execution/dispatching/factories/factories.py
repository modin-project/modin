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

import warnings
import typing
import re

from modin.config import Engine
from modin.utils import _inherit_docstrings
from modin.core.io import BaseIO
from pandas.util._decorators import doc

import pandas


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
    io_cls: BaseIO = None  # The module where the I/O functionality exists.

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
        method="utils.from_pandas",
    )
    def _from_pandas(cls, df):
        return cls.io_cls.from_pandas(df)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="Arrow Table",
        params="at : pyarrow.Table",
        method="utils.from_arrow",
    )
    def _from_arrow(cls, at):
        return cls.io_cls.from_arrow(at)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a non-pandas object (dict, list, np.array etc...)",
        params=_doc_io_method_all_params,
        method="utils.from_non_pandas",
    )
    def _from_non_pandas(cls, *args, **kwargs):
        return cls.io_cls.from_non_pandas(*args, **kwargs)

    @classmethod
    @doc(
        _doc_io_method_template,
        source="a DataFrame object supporting exchange protocol `__dataframe__()`",
        params=_doc_io_method_all_params,
        method="utils.from_dataframe",
    )
    def _from_dataframe(cls, *args, **kwargs):
        return cls.io_cls.from_dataframe(*args, **kwargs)

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


@doc(_doc_factory_class, execution_name="cuDFOnRay")
class CudfOnRayFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``cuDFOnRayIO``")
    def prepare(cls):
        from modin.core.execution.ray.implementations.cudf_on_ray.io import cuDFOnRayIO

        cls.io_cls = cuDFOnRayIO


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


@doc(_doc_abstract_factory_class, role="experimental")
class ExperimentalBaseFactory(BaseFactory):
    @classmethod
    @_inherit_docstrings(BaseFactory._read_sql)
    def _read_sql(cls, **kwargs):
        if Engine.get() != "Ray":
            if "partition_column" in kwargs:
                if kwargs["partition_column"] is not None:
                    warnings.warn(
                        "Distributed read_sql() was only implemented for Ray engine."
                    )
                del kwargs["partition_column"]
            if "lower_bound" in kwargs:
                if kwargs["lower_bound"] is not None:
                    warnings.warn(
                        "Distributed read_sql() was only implemented for Ray engine."
                    )
                del kwargs["lower_bound"]
            if "upper_bound" in kwargs:
                if kwargs["upper_bound"] is not None:
                    warnings.warn(
                        "Distributed read_sql() was only implemented for Ray engine."
                    )
                del kwargs["upper_bound"]
            if "max_sessions" in kwargs:
                if kwargs["max_sessions"] is not None:
                    warnings.warn(
                        "Distributed read_sql() was only implemented for Ray engine."
                    )
                del kwargs["max_sessions"]
        return cls.io_cls.read_sql(**kwargs)


@doc(_doc_factory_class, execution_name="experimental PandasOnRay")
class ExperimentalPandasOnRayFactory(ExperimentalBaseFactory, PandasOnRayFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="``ExperimentalPandasOnRayIO``")
    def prepare(cls):
        from modin.experimental.core.execution.ray.implementations.pandas_on_ray.io import (
            ExperimentalPandasOnRayIO,
        )

        cls.io_cls = ExperimentalPandasOnRayIO

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="CSV files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_csv_glob(cls, **kwargs):
        return cls.io_cls.read_csv_glob(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Pickle files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_pickle_distributed(cls, **kwargs):
        return cls.io_cls.read_pickle_distributed(**kwargs)

    @classmethod
    @doc(
        _doc_io_method_raw_template,
        source="Custom text files",
        params=_doc_io_method_kwargs_params,
    )
    def _read_custom_text(cls, **kwargs):
        return cls.io_cls.read_custom_text(**kwargs)

    @classmethod
    def _to_pickle_distributed(cls, *args, **kwargs):
        """
        Distributed pickle query compiler object.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
        return cls.io_cls.to_pickle_distributed(*args, **kwargs)


@doc(_doc_factory_class, execution_name="experimental PandasOnDask")
class ExperimentalPandasOnDaskFactory(ExperimentalBaseFactory, PandasOnDaskFactory):
    pass


@doc(_doc_factory_class, execution_name="experimental PandasOnPython")
class ExperimentalPandasOnPythonFactory(ExperimentalBaseFactory, PandasOnPythonFactory):
    pass


@doc(_doc_factory_class, execution_name="experimental PyarrowOnRay")
class ExperimentalPyarrowOnRayFactory(BaseFactory):  # pragma: no cover
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="experimental ``PyarrowOnRayIO``")
    def prepare(cls):
        from modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.io import (
            PyarrowOnRayIO,
        )

        cls.io_cls = PyarrowOnRayIO


@doc(_doc_abstract_factory_class, role="experimental remote")
class ExperimentalRemoteFactory(ExperimentalBaseFactory):
    wrapped_factory = BaseFactory

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="an underlying remote")
    def prepare(cls):
        # query_compiler import is needed so remote PandasQueryCompiler
        # has an imported local counterpart;
        # if there isn't such counterpart rpyc generates some bogus
        # class type which raises TypeError()
        # upon checking its isinstance() or issubclass()
        import modin.core.storage_formats.pandas.query_compiler  # noqa: F401
        from modin.experimental.cloud import get_connection

        # import a numpy overrider if it wasn't already imported
        import modin.experimental.pandas.numpy_wrap  # noqa: F401

        class WrappedIO:
            def __init__(self, conn, factory):
                self.__conn = conn
                remote_factory = getattr(
                    conn.modules[factory.__module__], factory.__name__
                )
                remote_factory.prepare()
                self.__io_cls = remote_factory.io_cls
                self.__reads = {
                    name for name in BaseIO.__dict__ if name.startswith("read_")
                }
                self.__wrappers = {}

            def __getattr__(self, name):
                if name in self.__reads:
                    try:
                        wrap = self.__wrappers[name]
                    except KeyError:

                        def wrap(*a, _original=getattr(self.__io_cls, name), **kw):
                            a, kw = self.__conn.deliver(a, kw)
                            return _original(*a, **kw)

                        self.__wrappers[name] = wrap
                else:
                    wrap = getattr(self.__io_cls, name)
                return wrap

        cls.io_cls = WrappedIO(get_connection(), cls.wrapped_factory)


@doc(_doc_factory_class, execution_name="experimental remote PandasOnRay")
class ExperimentalPandasOnCloudrayFactory(ExperimentalRemoteFactory):
    wrapped_factory = PandasOnRayFactory


@doc(_doc_factory_class, execution_name="experimental remote PandasOnPython")
class ExperimentalPandasOnCloudpythonFactory(ExperimentalRemoteFactory):
    wrapped_factory = PandasOnPythonFactory


@doc(_doc_factory_class, execution_name="experimental HdkOnNative")
class ExperimentalHdkOnNativeFactory(BaseFactory):
    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name="experimental ``HdkOnNativeIO``")
    def prepare(cls):
        from modin.experimental.core.execution.native.implementations.hdk_on_native.io import (
            HdkOnNativeIO,
        )

        cls.io_cls = HdkOnNativeIO


@doc(_doc_factory_class, execution_name="experimental remote HdkOnNative")
class ExperimentalHdkOnCloudnativeFactory(ExperimentalRemoteFactory):
    wrapped_factory = ExperimentalHdkOnNativeFactory
