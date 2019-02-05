import sys
import warnings

from modin.data_management.factories import (
    BaseFactory,
    PandasOnRayFactory,
    PandasOnPythonFactory,
)
from modin import __execution_engine__ as execution_engine
from modin import __partition_format__ as partition_format
from .pandas_on_ray.io_exp import ExperimentalPandasOnRayIO


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
        return cls.io_cls.read_sql(**kwargs)


class ExperimentalPandasOnRayFactory(ExperimentalBaseFactory, PandasOnRayFactory):

    io_cls = ExperimentalPandasOnRayIO


class ExperimentalPandasOnPythonFactory(ExperimentalBaseFactory, PandasOnPythonFactory):

    pass
