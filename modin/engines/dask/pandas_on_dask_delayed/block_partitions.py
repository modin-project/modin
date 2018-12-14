import pandas
import dask

from modin.engines.base.block_partitions import BaseBlockPartitions
from modin.error_message import ErrorMessage
from .axis_partition import DaskColumnPartition, DaskRowPartition
from .remote_partition import DaskRemotePartition


class DaskBlockPartitions(BaseBlockPartitions):
    """This class implements the interface in `BaseBlockPartitions`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = DaskRemotePartition
    _column_partitions_class = DaskColumnPartition
    _row_partition_class = DaskRowPartition

    def __init__(self, partitions):
        self.partitions = partitions

    def _get_partitions(self):
        # We don't filter fot the laziness of Dask Delayed
        return self._partitions_cache

    def _set_partitions(self, new_partitions):
        self._partitions_cache = new_partitions

    partitions = property(_get_partitions, _set_partitions)

    def to_pandas(self, is_transposed=False):
        """Convert this object into a Pandas DataFrame from the partitions.

        Args:
            is_transposed: A flag for telling this object that the external
                representation is transposed, but not the internal.

        Returns:
            A Pandas DataFrame
        """
        # In the case this is transposed, it is easier to just temporarily
        # transpose back then transpose after the conversion. The performance
        # is the same as if we individually transposed the blocks and
        # concatenated them, but the code is much smaller.
        if is_transposed:
            return self.transpose().to_pandas(False).T
        else:
            retrieved_objects = [
                list(dask.compute(*[obj.dask_obj for obj in part])) for part in self.partitions
            ]
            if all(
                isinstance(part, pandas.Series)
                for row in retrieved_objects
                for part in row
            ):
                axis = 0
            elif all(
                isinstance(part, pandas.DataFrame)
                for row in retrieved_objects
                for part in row
            ):
                axis = 1
            else:
                ErrorMessage.catch_bugs_and_request_email(True)
            df_rows = [
                pandas.concat([part for part in row], axis=axis)
                for row in retrieved_objects
                if not all(part.empty for part in row)
            ]
            if len(df_rows) == 0:
                return pandas.DataFrame()
            else:
                return pandas.concat(df_rows)