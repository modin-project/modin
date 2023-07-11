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
from typing import Union

import pandas

import pyarrow as pa

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from ..dataframe.utils import arrow_to_pandas
from ..db_worker import DbTable


class HdkOnNativeDataframePartition(PandasDataframePartition):
    """
    A partition of ``HdkOnNativeDataframe`` frame.

    Class holds either a ``DbTable`` or ``pandas.DataFrame`` or ``pyarrow.Table``.

    Parameters
    ----------
    data :  DbTable or pandas.DataFrame or pyarrow.Table
        Partition data in either pandas or PyArrow format.

    Attributes
    ----------
    _data :  DbTable or pandas.DataFrame or pyarrow.Table
        Partition data in either pandas or PyArrow format.
    _length_cache : int
        Length of the partition.
    _width_cache : int
        Width of the partition.
    """

    def __init__(
        self,
        data: Union[DbTable, pa.Table, pandas.DataFrame],
    ):
        super().__init__()
        assert isinstance(data, (DbTable, pa.Table, pandas.DataFrame))
        self._data = data

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
        if isinstance(obj, DbTable):
            obj = obj.to_arrow()
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
        DbTable or pandas.DataFrame or pyarrow.Table
        """
        return self._data

    @classmethod
    def put(cls, obj):
        """
        Create partition from ``DbTable`` or ``pandas.DataFrame`` or ``pyarrow.Table``.

        Parameters
        ----------
        obj : DbTable or pandas.DataFrame or pyarrow.Table
            Source frame.

        Returns
        -------
        HdkOnNativeDataframePartition
            The new partition.
        """
        return cls(obj)

    @property
    def _length_cache(self):
        """
        Number of rows.

        Returns
        -------
        int
        """
        return len(self._data)

    @property
    def _width_cache(self):
        """
        Number of columns.

        Returns
        -------
        int
        """
        if isinstance(self._data, pa.Table):
            return self._data.num_columns
        else:
            return self._data.shape[1]
