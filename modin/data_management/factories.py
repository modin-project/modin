import os
import sys
import warnings

from modin import __execution_engine__ as execution_engine
from modin import __partition_format__ as partition_format


class BaseFactory(object):
    @property
    def io_cls(self):
        """The module where the I/O functionality exists."""
        raise NotImplementedError("Implement in children classes!")

    @classmethod
    def _determine_engine(cls):
        if os.environ.get("MODIN_EXPERIMENTAL", "") == "True":
            return ExperimentalBaseFactory._determine_engine()
        factory_name = partition_format + "On" + execution_engine + "Factory"
        return getattr(sys.modules[__name__], factory_name)

    @classmethod
    def build_manager(cls):
        return cls._determine_engine().build_manager()

    @classmethod
    def from_pandas(cls, df):
        return cls._determine_engine()._from_pandas(df)

    @classmethod
    def _from_pandas(cls, df):
        return cls.io_cls.from_pandas(df)

    @classmethod
    def from_non_pandas(cls, *args, **kwargs):
        return cls._determine_engine()._from_non_pandas(*args, **kwargs)

    @classmethod
    def _from_non_pandas(cls, *args, **kwargs):
        return cls.io_cls.from_non_pandas(*args, **kwargs)

    @classmethod
    def read_parquet(cls, **kwargs):
        return cls._determine_engine()._read_parquet(**kwargs)

    @classmethod
    def _read_parquet(cls, **kwargs):
        return cls.io_cls.read_parquet(**kwargs)

    @classmethod
    def read_csv(cls, **kwargs):
        return cls._determine_engine()._read_csv(**kwargs)

    @classmethod
    def _read_csv(cls, **kwargs):
        return cls.io_cls.read_csv(**kwargs)

    @classmethod
    def read_json(cls, **kwargs):
        return cls._determine_engine()._read_json(**kwargs)

    @classmethod
    def _read_json(cls, **kwargs):
        return cls.io_cls.read_json(**kwargs)

    @classmethod
    def read_gbq(cls, **kwargs):
        return cls._determine_engine()._read_gbq(**kwargs)

    @classmethod
    def _read_gbq(cls, **kwargs):
        return cls.io_cls.read_gbq(**kwargs)

    @classmethod
    def read_html(cls, **kwargs):
        return cls._determine_engine()._read_html(**kwargs)

    @classmethod
    def _read_html(cls, **kwargs):
        return cls.io_cls.read_html(**kwargs)

    @classmethod
    def read_clipboard(cls, **kwargs):  # pragma: no cover
        return cls._determine_engine()._read_clipboard(**kwargs)

    @classmethod
    def _read_clipboard(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_clipboard(**kwargs)

    @classmethod
    def read_excel(cls, **kwargs):
        return cls._determine_engine()._read_excel(**kwargs)

    @classmethod
    def _read_excel(cls, **kwargs):
        return cls.io_cls.read_excel(**kwargs)

    @classmethod
    def read_hdf(cls, **kwargs):
        return cls._determine_engine()._read_hdf(**kwargs)

    @classmethod
    def _read_hdf(cls, **kwargs):
        return cls.io_cls.read_hdf(**kwargs)

    @classmethod
    def read_feather(cls, **kwargs):
        return cls._determine_engine()._read_feather(**kwargs)

    @classmethod
    def _read_feather(cls, **kwargs):
        return cls.io_cls.read_feather(**kwargs)

    @classmethod
    def read_msgpack(cls, **kwargs):
        return cls._determine_engine()._read_msgpack(**kwargs)

    @classmethod
    def _read_msgpack(cls, **kwargs):
        return cls.io_cls.read_msgpack(**kwargs)

    @classmethod
    def read_stata(cls, **kwargs):
        return cls._determine_engine()._read_stata(**kwargs)

    @classmethod
    def _read_stata(cls, **kwargs):
        return cls.io_cls.read_stata(**kwargs)

    @classmethod
    def read_sas(cls, **kwargs):  # pragma: no cover
        return cls._determine_engine()._read_sas(**kwargs)

    @classmethod
    def _read_sas(cls, **kwargs):  # pragma: no cover
        return cls.io_cls.read_sas(**kwargs)

    @classmethod
    def read_pickle(cls, **kwargs):
        return cls._determine_engine()._read_pickle(**kwargs)

    @classmethod
    def _read_pickle(cls, **kwargs):
        return cls.io_cls.read_pickle(**kwargs)

    @classmethod
    def read_sql(cls, **kwargs):
        return cls._determine_engine()._read_sql(**kwargs)

    @classmethod
    def _read_sql(cls, **kwargs):
        return cls.io_cls.read_sql(**kwargs)

    @classmethod
    def read_fwf(cls, **kwargs):
        return cls._determine_engine()._read_fwf(**kwargs)

    @classmethod
    def _read_fwf(cls, **kwargs):
        return cls.io_cls.read_fwf(**kwargs)

    @classmethod
    def read_sql_table(cls, **kwargs):
        return cls._determine_engine()._read_sql_table(**kwargs)

    @classmethod
    def _read_sql_table(cls, **kwargs):
        return cls.io_cls.read_sql_table(**kwargs)

    @classmethod
    def read_sql_query(cls, **kwargs):
        return cls._determine_engine()._read_sql_query(**kwargs)

    @classmethod
    def _read_sql_query(cls, **kwargs):
        return cls.io_cls.read_sql_query(**kwargs)

    @classmethod
    def read_spss(cls, **kwargs):
        return cls._determine_engine()._read_spss(**kwargs)

    @classmethod
    def _read_spss(cls, **kwargs):
        return cls.io_cls.read_spss(**kwargs)

    @classmethod
    def to_sql(cls, *args, **kwargs):
        return cls._determine_engine()._to_sql(*args, **kwargs)

    @classmethod
    def _to_sql(cls, *args, **kwargs):
        return cls.io_cls.to_sql(*args, **kwargs)

    @classmethod
    def to_pickle(cls, *args, **kwargs):
        return cls._determine_engine()._to_pickle(*args, **kwargs)

    @classmethod
    def _to_pickle(cls, *args, **kwargs):
        return cls.io_cls.to_pickle(*args, **kwargs)


class PandasOnRayFactory(BaseFactory):

    from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO

    io_cls = PandasOnRayIO


class PandasOnPythonFactory(BaseFactory):

    from modin.engines.python.pandas_on_python.io import PandasOnPythonIO

    io_cls = PandasOnPythonIO


class PandasOnDaskFactory(BaseFactory):

    from modin.engines.dask.pandas_on_dask_futures.io import PandasOnDaskIO

    io_cls = PandasOnDaskIO


class PyarrowOnRayFactory(BaseFactory):

    if partition_format == "Pyarrow" and not os.environ.get(
        "MODIN_EXPERIMENTAL", False
    ):
        raise ImportError(
            "Pyarrow on Ray is only accessible through the experimental API.\nRun "
            "`import modin.experimental.pandas as pd` to use Pyarrow on Ray."
        )


class ExperimentalBaseFactory(BaseFactory):
    @classmethod
    def _determine_engine(cls):
        factory_name = "Experimental{}On{}Factory".format(
            partition_format, execution_engine
        )
        return getattr(sys.modules[__name__], factory_name)

    @classmethod
    def _read_sql(cls, **kwargs):
        if execution_engine != "Ray":
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

    from modin.experimental.engines.pandas_on_ray.io_exp import (
        ExperimentalPandasOnRayIO,
    )

    io_cls = ExperimentalPandasOnRayIO


class ExperimentalPandasOnPythonFactory(ExperimentalBaseFactory, PandasOnPythonFactory):

    pass


class ExperimentalPyarrowOnRayFactory(BaseFactory):  # pragma: no cover

    from modin.experimental.engines.pyarrow_on_ray.io import PyarrowOnRayIO

    io_cls = PyarrowOnRayIO
