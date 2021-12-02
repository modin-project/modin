# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Module provides a partition class for ``OmnisciOnNativeDataframe`` frame."""

import pandas

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from ..omnisci_worker import OmnisciServer
import pyarrow


class OmnisciOnNativeDataframePartition(PandasDataframePartition):
    """
    A partition of ``OmnisciOnNativeDataframe`` frame.

    Class holds either a ``pandas.DataFrame`` or ``pyarrow.Table``.

    Parameters
    ----------
    frame_id : str, optional
        A corresponding OmniSci table name or None.
    pandas_df : pandas.DataFrame, optional
        Partition data in pandas format.
    arrow_table : pyarrow.Table, optional
        Partition data in Arrow format.
    length : int, optional
        Length of the partition.
    width : int, optional
        Width of the partition.

    Attributes
    ----------
    frame_id : str
        A corresponding OmniSci table name if partition was imported
        into OmniSci. Otherwise None.
    pandas_df : pandas.DataFrame, optional
        Partition data in pandas format.
    arrow_table : pyarrow.Table
        Partition data in Arrow format. None for partitions holding
        `pandas.DataFrame`.
    _length_cache : int
        Length of the partition.
    _width_cache : int
        Width of the partition.
    """

    def __init__(
        self, frame_id=None, pandas_df=None, arrow_table=None, length=None, width=None
    ):
        self.pandas_df = pandas_df
        self.frame_id = frame_id
        self.arrow_table = arrow_table
        self._length_cache = length
        self._width_cache = width
        self._server = OmnisciServer

    def __del__(self):
        """Deallocate OmniSci resources related to the partition."""
        if self.frame_id is not None:
            self._server.executeDDL(f"DROP TABLE {self.frame_id};")

    def to_pandas(self):
        """
        Transform to pandas format.

        Returns
        -------
        pandas.DataFrame
        """
        obj = self.get()
        if isinstance(obj, (pandas.DataFrame, pandas.Series)):
            return obj
        assert isinstance(obj, pyarrow.Table)
        return obj.to_pandas()

    def get(self):
        """
        Get partition data.

        Returns
        -------
        pandas.DataFrame or pyarrow.Table
        """
        if self.arrow_table is not None:
            return self.arrow_table
        return self.pandas_df

    @classmethod
    def put(cls, obj):
        """
        Create partition from ``pandas.DataFrame`` or ``pandas.Series``.

        Parameters
        ----------
        obj : pandas.Series or pandas.DataFrame
            Source frame.

        Returns
        -------
        OmnisciOnNativeDataframePartition
            The new partition.
        """
        return OmnisciOnNativeDataframePartition(
            pandas_df=obj, length=len(obj.index), width=len(obj.columns)
        )

    def wait(self):
        """
        Wait until the partition data is ready for use.

        Returns
        -------
        pandas.DataFrame
            The partition that is ready to be used.
        """
        if self.arrow_table is not None:
            return self.arrow_table
        return self.pandas_df

    @classmethod
    def put_arrow(cls, obj):
        """
        Create partition from ``pyarrow.Table``.

        Parameters
        ----------
        obj : pyarrow.Table
            Source table.

        Returns
        -------
        OmnisciOnNativeDataframePartition
            The new partition.
        """
        return OmnisciOnNativeDataframePartition(
            arrow_table=obj,
            length=len(obj),
            width=len(obj.columns),
        )
