from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.python.pandas_on_python.frame.data import PandasOnPythonFrame


class PandasOnPythonIO(BaseIO):

    frame_cls = PandasOnPythonFrame
    query_compiler_cls = PandasQueryCompiler
