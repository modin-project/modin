from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from .. import __execution_engine__ as execution_engine
from .. import __partition_format__ as partition_format
from modin.data_management.query_compiler import PandasQueryCompiler
from modin.data_management.io import pandas_on_ray as io_pandas_on_ray
from .partitioning.partition_collections import RayBlockPartitions
from .partitioning.partition_collections import PythonBlockPartitions


class BaseFactory(object):
    @classmethod
    def _determine_engine(cls):
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
        return cls.data_mgr_cls.from_pandas(df, cls.block_partitions_cls)

    @classmethod
    def read_parquet(cls, **kwargs):
        return cls._determine_engine()._read_parquet(**kwargs)

    @classmethod
    def _read_parquet(cls, **kwargs):
        return cls.io_module.read_parquet(**kwargs)


class PandasOnRayFactory(BaseFactory):

    data_mgr_cls = PandasQueryCompiler
    block_partitions_cls = RayBlockPartitions
    io_module = io_pandas_on_ray


class PandasOnPythonFactory(BaseFactory):

    data_mgr_cls = PandasQueryCompiler
    block_partitions_cls = PythonBlockPartitions
