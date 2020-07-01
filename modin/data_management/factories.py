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

import warnings

from modin import execution_engine
from modin.engines.base.io import BaseIO

import pandas

types_dictionary = {"pandas": {"category": pandas.CategoricalDtype}}


class BaseFactory(object):
    """
    Abstract factory which allows to override the io module easily.
    """

    io_cls: BaseIO = None  # The module where the I/O functionality exists.

    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        raise NotImplementedError("Subclasses of BaseFactory must implement prepare")

    @classmethod
    def _from_pandas(cls, df):
        return cls.io_cls.from_pandas(df)

    @classmethod
    def _from_non_pandas(cls, *args, **kwargs):
        return cls.io_cls.from_non_pandas(*args, **kwargs)

    @classmethod
    def _read_parquet(cls, **kwargs):
        return cls.io_cls.read_parquet(**kwargs)

    @classmethod
    def _read_csv(cls, **kwargs):
        return cls.io_cls.read_csv(**kwargs)

    @classmethod
    def _read_json(cls, **kwargs):
        return cls.io_cls.read_json(**kwargs)

    @classmethod
    def _read_gbq(cls, **kwargs):
        return cls.io_cls.read_gbq(**kwargs)

    @classmethod
    def _read_html(cls, **kwargs):
        return cls.io_cls.read_html(**kwargs)

    @classmethod
    def _read_clipboard(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_clipboard(**kwargs)

    @classmethod
    def _read_excel(cls, **kwargs):
        return cls.io_cls.read_excel(**kwargs)

    @classmethod
    def _read_hdf(cls, **kwargs):
        return cls.io_cls.read_hdf(**kwargs)

    @classmethod
    def _read_feather(cls, **kwargs):
        return cls.io_cls.read_feather(**kwargs)

    @classmethod
    def _read_stata(cls, **kwargs):
        return cls.io_cls.read_stata(**kwargs)

    @classmethod
    def _read_sas(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_sas(**kwargs)

    @classmethod
    def _read_pickle(cls, **kwargs):
        return cls.io_cls.read_pickle(**kwargs)

    @classmethod
    def _read_sql(cls, **kwargs):
        return cls.io_cls.read_sql(**kwargs)

    @classmethod
    def _read_fwf(cls, **kwargs):
        return cls.io_cls.read_fwf(**kwargs)

    @classmethod
    def _read_sql_table(cls, **kwargs):
        return cls.io_cls.read_sql_table(**kwargs)

    @classmethod
    def _read_sql_query(cls, **kwargs):
        return cls.io_cls.read_sql_query(**kwargs)

    @classmethod
    def _read_spss(cls, **kwargs):
        return cls.io_cls.read_spss(**kwargs)

    @classmethod
    def _to_sql(cls, *args, **kwargs):
        return cls.io_cls.to_sql(*args, **kwargs)

    @classmethod
    def _to_pickle(cls, *args, **kwargs):
        return cls.io_cls.to_pickle(*args, **kwargs)


class PandasOnRayFactory(BaseFactory):
    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO

        cls.io_cls = PandasOnRayIO


class PandasOnPythonFactory(BaseFactory):
    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        from modin.engines.python.pandas_on_python.io import PandasOnPythonIO

        cls.io_cls = PandasOnPythonIO


class PandasOnDaskFactory(BaseFactory):
    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        from modin.engines.dask.pandas_on_dask.io import PandasOnDaskIO

        cls.io_cls = PandasOnDaskIO


class ExperimentalBaseFactory(BaseFactory):
    @classmethod
    def _read_sql(cls, **kwargs):
        if execution_engine.get() != "Ray":
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


class ExperimentalPandasOnRayFactory(ExperimentalBaseFactory, PandasOnRayFactory):
    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        from modin.experimental.engines.pandas_on_ray.io_exp import (
            ExperimentalPandasOnRayIO,
        )

        cls.io_cls = ExperimentalPandasOnRayIO


class ExperimentalPandasOnPythonFactory(ExperimentalBaseFactory, PandasOnPythonFactory):
    pass


class ExperimentalPyarrowOnRayFactory(BaseFactory):  # pragma: no cover
    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        from modin.experimental.engines.pyarrow_on_ray.io import PyarrowOnRayIO

        cls.io_cls = PyarrowOnRayIO
