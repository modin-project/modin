import pandas

from .partition_manager import PandasOnRayFrameManager
from modin.engines.base.frame.data import BasePandasFrame
from modin import __execution_engine__
from modin.backends.pandas.parsers import find_common_type_cat as find_common_type

if __execution_engine__ == "Ray":
    import ray


class PandasOnRayFrame(BasePandasFrame):

    _frame_mgr_cls = PandasOnRayFrameManager

    @classmethod
    def combine_dtypes(cls, list_of_dtypes, column_names):
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column.
        dtypes = (
            pandas.concat(ray.get(list_of_dtypes), axis=1)
            .apply(lambda row: find_common_type(row.values), axis=1)
            .squeeze(axis=0)
        )
        dtypes.index = column_names
        return dtypes
