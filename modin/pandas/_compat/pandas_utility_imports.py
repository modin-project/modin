from packaging import version
import pandas


pandas_version = pandas.__version__
if version.parse("1.1.0") <= version.parse(pandas_version) <= version.parse("1.1.5"):
    from .py36.imports import *
elif version.parse("1.4.0") <= version.parse(pandas_version) <= version.parse("1.4.99"):
    from .latest.imports import *
