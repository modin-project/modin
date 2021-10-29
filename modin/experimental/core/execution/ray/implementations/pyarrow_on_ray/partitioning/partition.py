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

"""The module defines interface for a partition with PyArrow storage format and Ray engine."""

import pandas
from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition import (
    PandasOnRayDataframePartition,
)

import ray
import pyarrow


class PyarrowOnRayDataframePartition(PandasOnRayDataframePartition):
    """
    Class provides partition interface specific for PyArrow storage format and Ray engine.

    Inherits functionality from the ``PandasOnRayDataframePartition`` class.

    Parameters
    ----------
    object_id : ray.ObjectRef
        A reference to ``pyarrow.Table`` that needs to be wrapped with this class.
    length : ray.ObjectRef or int, optional
        Length or reference to it of wrapped ``pyarrow.Table``.
    width : ray.ObjectRef or int, optional
        Width or reference to it of wrapped ``pyarrow.Table``.
    ip : ray.ObjectRef or str, optional
        Node IP address or reference to it that holds wrapped ``pyarrow.Table``.
    call_queue : list, optional
        Call queue that needs to be executed on wrapped ``pyarrow.Table``.
    """

    def to_pandas(self):
        """
        Convert the object stored in this partition to a ``pandas.DataFrame``.

        Returns
        -------
        dataframe : pandas.DataFrame or pandas.Series
            Resulting DataFrame or Series.
        """
        dataframe = self.get().to_pandas()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    @classmethod
    def put(cls, obj):
        """
        Put an object in the Plasma store and wrap it in this object.

        Parameters
        ----------
        obj : object
            The object to be put.

        Returns
        -------
        PyarrowOnRayDataframePartition
            A ``PyarrowOnRayDataframePartition`` object.
        """
        return PyarrowOnRayDataframePartition(ray.put(pyarrow.Table.from_pandas(obj)))

    @classmethod
    def _length_extraction_fn(cls):
        """
        Return the callable that extracts the number of rows from the given ``pyarrow.Table``.

        Returns
        -------
        callable
        """
        return lambda table: table.num_rows

    @classmethod
    def _width_extraction_fn(cls):
        """
        Return the callable that extracts the number of columns from the given ``pyarrow.Table``.

        Returns
        -------
        callable
        """
        return lambda table: table.num_columns - (1 if "index" in table.columns else 0)

    @classmethod
    def empty(cls):
        """
        Put empty ``pandas.DataFrame`` in the Plasma store and wrap it in this object.

        Returns
        -------
        PyarrowOnRayDataframePartition
            A ``PyarrowOnRayDataframePartition`` object.
        """
        return cls.put(pandas.DataFrame())
