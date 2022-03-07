from packaging import version
import pandas


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
            elif classname == "DataFrame":
                from .py36 import Python36CompatibleDataFrame

                return Python36CompatibleDataFrame
            elif classname == "Series":
                from .py36 import Python36CompatibilitySeries

                return Python36CompatibilitySeries
            else:
                raise ValueError
        elif (
            version.parse("1.4.0")
            <= version.parse(pandas_version)
            <= version.parse("1.4.99")
        ):
            if classname == "BasePandasDataset":
                from .latest import LatestCompatibleBasePandasDataset

                return LatestCompatibleBasePandasDataset
            elif classname == "DataFrame":
                from .latest import LatestCompatibleDataFrame

                return LatestCompatibleDataFrame
            elif classname == "Series":
                from .latest import LatestCompatibleSeries

                return LatestCompatibleSeries
            else:
                raise ValueError
        else:
            raise ValueError
