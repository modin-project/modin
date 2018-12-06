from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from .query_compiler import PandasQueryCompiler
from .. import __execution_engine__ as execution_engine
from .. import __partition_format__ as partition_format


class BaseFactory(object):
    @property
    def query_compiler_cls(self):
        """The Query Compiler class for this factory."""
        raise NotImplementedError("Implement in children classes!")

    @property
    def block_partitions_cls(self):
        """The Block Partitions class for this factory."""
        raise NotImplementedError("Implement in children classes!")

    @property
    def io_cls(self):
        """The module where the I/O functionality exists."""
        raise NotImplementedError("Implement in children classes!")

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
        return cls.query_compiler_cls.from_pandas(df, cls.block_partitions_cls)

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
    def read_clipboard(cls, **kwargs):
        return cls._determine_engine()._read_clipboard(**kwargs)

    @classmethod
    def _read_clipboard(cls, **kwargs):
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
    def read_sas(cls, **kwargs):
        return cls._determine_engine()._read_sas(**kwargs)

    @classmethod
    def _read_sas(cls, **kwargs):
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


class PandasOnRayFactory(BaseFactory):

    from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO
    from modin.engines.ray.pandas_on_ray.block_partitions import RayBlockPartitions

    query_compiler_cls = PandasQueryCompiler
    block_partitions_cls = RayBlockPartitions
    io_cls = PandasOnRayIO


class PandasOnPythonFactory(BaseFactory):

    from modin.engines.python.pandas_on_python.block_partitions import (
        PythonBlockPartitions,
    )
    from modin.engines.python.pandas_on_python.io import PandasOnPythonIO

    query_compiler_cls = PandasQueryCompiler
    block_partitions_cls = PythonBlockPartitions
    io_cls = PandasOnPythonIO


class PandasOnDaskFactory(BaseFactory):

    from modin.engines.dask.pandas_on_dask_delayed.block_partitions import (
        DaskBlockPartitions,
    )
    from modin.engines.dask.pandas_on_dask_delayed.io import PandasOnDaskIO

    query_compiler_cls = PandasQueryCompiler
    block_partitions_cls = DaskBlockPartitions
    io_cls = PandasOnDaskIO
