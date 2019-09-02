from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask_futures.frame.data import PandasOnDaskFrame


class PandasOnDaskIO(BaseIO):

    frame_cls = PandasOnDaskFrame
    query_compiler_cls = PandasQueryCompiler
