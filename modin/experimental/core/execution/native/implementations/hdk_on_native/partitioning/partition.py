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
from pandas._typing import AnyArrayLike

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition

from ..dataframe.utils import ColNameCodec, arrow_to_pandas
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

    def get(self, to_arrow: bool = False) -> Union[DbTable, pandas.DataFrame, pa.Table]:
        """
        Get partition data.

        Parameters
        ----------
        to_arrow : bool, default: False
            Convert the data to ``pyarrow.Table``.

        Returns
        -------
        ``DbTable`` or ``pandas.DataFrame`` or ``pyarrow.Table``
        """
        if to_arrow:
            if isinstance(self._data, pandas.DataFrame):
                self._data = pa.Table.from_pandas(self._data, preserve_index=False)
            elif isinstance(self._data, DbTable):
                return self._data.to_arrow()
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

    def insert(self, idx: int, name: str, value: AnyArrayLike):
        """
        Insert column into this raw partition.

        Parameters
        ----------
        idx : int
        name : str
        value : AnyArrayLike

        Returns
        -------
        tuple of HdkOnNativeDataframePartition, dtype
        """
        data = self._data
        name = ColNameCodec.encode(name)

        if isinstance(data, pandas.DataFrame):
            data = data.copy(False)
            data.insert(idx, name, value)
            dtype = data.dtypes[idx]
        elif isinstance(data, pa.Table):
            try:
                new_data = data.add_column(idx, name, [value])
                dtype = new_data.field(idx).type.to_pandas_dtype()
                data = new_data
            except Exception:
                try:
                    df = pandas.DataFrame({name: value})
                    at = pa.Table.from_pandas(df, preserve_index=False)
                    data = data.add_column(idx, at.field(0), at.column(0))
                    dtype = df.dtypes[0]
                except Exception as err:
                    raise NotImplementedError(repr(err))
        else:
            raise NotImplementedError(f"Insertion into {type(data)}")

        return HdkOnNativeDataframePartition(data), dtype

    @property
    def raw(self):
        """
        True if the partition contains a raw data.

        The raw data is either ``pandas.DataFrame`` or ``pyarrow.Table``.

        Returns
        -------
        bool
        """
        return isinstance(self._data, (pandas.DataFrame, pa.Table))

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
