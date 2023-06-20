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

"""Module provides a partition class for ``HdkOnNativeDataframe`` frame."""
from typing import Optional, Union

import pandas

import pyarrow as pa

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from ..dataframe.utils import arrow_to_pandas
from ..db_worker import DbWorker


class HdkOnNativeDataframePartition(PandasDataframePartition):
    """
    A partition of ``HdkOnNativeDataframe`` frame.

    Class holds either a ``pandas.DataFrame`` or ``pyarrow.Table``.

    Parameters
    ----------
    data :  pandas.DataFrame or pyarrow.Table
        Partition data in either pandas or PyArrow format.
    frame_id : str, optional
        A corresponding HDK table name or None.

    Attributes
    ----------
    _data :  pandas.DataFrame or pyarrow.Table
        Partition data in either pandas or PyArrow format.
    frame_id : str
        A corresponding HDK table name if partition was imported
        into HDK. Otherwise None.
    _length_cache : int
        Length of the partition.
    _width_cache : int
        Width of the partition.
    """

    def __init__(
        self,
        data: Union[pa.Table, pandas.DataFrame],
        frame_id: Optional[str] = None,
    ):
        self._data = data
        self.frame_id = frame_id
        if isinstance(data, pa.Table):
            self._length_cache = data.num_rows
            self._width_cache = data.num_columns
        else:
            assert isinstance(data, pandas.DataFrame)
            self._length_cache = len(data)
            self._width_cache = len(data.columns)

    def __del__(self):
        """Deallocate HDK resources related to the partition."""
        if self.frame_id is not None:
            DbWorker.dropTable(self.frame_id)

    def to_pandas(self):
        """
        Transform to pandas format.

        Returns
        -------
        pandas.DataFrame
        """
        obj = self.get()
        if isinstance(obj, pandas.DataFrame):
            return obj
        assert isinstance(obj, pa.Table)
        return arrow_to_pandas(obj)

    def to_numpy(self, **kwargs):
        """
        Transform to NumPy format.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed in ``to_numpy``.

        Returns
        -------
        np.ndarray
        """
        return self.to_pandas().to_numpy(**kwargs)

    def get(self):
        """
        Get partition data.

        Returns
        -------
        pandas.DataFrame or pyarrow.Table
        """
        return self._data

    @classmethod
    def put(cls, obj):
        """
        Create partition from ``pandas.DataFrame`` or ``pyarrow.Table``.

        Parameters
        ----------
        obj : pandas.DataFrame or pyarrow.Table
            Source frame.

        Returns
        -------
        HdkOnNativeDataframePartition
            The new partition.
        """
        return cls(obj)
