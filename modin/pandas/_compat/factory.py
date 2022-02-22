from packaging import version
import importlib
import pandas
import sys


class CompatibilityFactory(object):
    @classmethod
    def generate_compatibility_class(cls, classname):
        pandas_version = pandas.__version__
        if (
            version.parse("1.1.0")
            <= version.parse(pandas_version)
            <= version.parse("1.1.5")
        ):
            if classname == "BasePandasDataset":
                from .py36 import Python36CompatibleBasePandasDataset

                return Python36CompatibleBasePandasDataset
            else:
                raise ValueError
        if (
            version.parse("1.4.0")
            <= version.parse(pandas_version)
            <= version.parse("1.4.99")
        ):
            if classname == "BasePandasDataset":
                from .latest import LatestCompatibleBasePandasDataset

                return LatestCompatibleBasePandasDataset
            else:
                raise ValueError
        else:
            raise ValueError
