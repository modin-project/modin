import sys

from modin.data_management.factories import BaseFactory, PandasOnRayFactory
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


class ExperimentalPandasOnRayFactory(ExperimentalBaseFactory, PandasOnRayFactory):

    io_cls = ExperimentalPandasOnRayIO
