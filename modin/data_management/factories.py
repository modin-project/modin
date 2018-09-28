from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from .. import __execution_engine__ as execution_engine
from .. import __partition_format__ as partition_format
from .data_manager import PandasDataManager
from .partitioning.partition_collections import RayBlockPartitions


class BaseFactory(object):
    @classmethod
    def _determine_engine(cls):
        factory_name = partition_format + "Backed" + execution_engine + "Factory"
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


class PandasBackedRayFactory(BaseFactory):

    data_mgr_cls = PandasDataManager
    block_partitions_cls = RayBlockPartitions
