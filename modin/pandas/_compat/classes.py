from packaging import version
import pandas

if (
    version.parse("1.1.0")
    <= version.parse(pandas.__version__)
    <= version.parse("1.1.5")
):
    from .py36 import (
        Python36CompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .py36 import Python36CompatibleDataFrame as DataFrameCompat
    from .py36 import Python36CompatibilitySeries as SeriesCompat
elif (
    version.parse("1.4.0")
    <= version.parse(pandas.__version__)
    <= version.parse("1.4.99")
):
    from .latest import (
        LatestCompatibleBasePandasDataset as BasePandasDatasetCompat,
    )
    from .latest import LatestCompatibleDataFrame as DataFrameCompat
    from .latest import LatestCompatibleSeries as SeriesCompat
else:
    raise ImportError(f"Unsupported pandas version: {pandas.__version__}")

__all__ = ["BasePandasDatasetCompat", "DataFrameCompat", "SeriesCompat"]
