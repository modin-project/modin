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

"""Module houses class that implements ``PandasDataframe``."""

from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe

from ..partitioning.partition_manager import PandasOnDaskDataframePartitionManager


class PandasOnDaskDataframe(PandasDataframe):
    """
    The class implements the interface in ``PandasDataframe``.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a pandas.Index.
    columns : sequence
        The columns object for the dataframe. Converted to a pandas.Index.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    pandas_backend : {"pyarrow", None}, optional
        Backend used by pandas. None - means default NumPy backend.
    """

    _partition_mgr_cls = PandasOnDaskDataframePartitionManager

    @classmethod
    def reconnect(cls, address, attributes):  # noqa: GL08
        # The main goal is to configure the client for the worker process
        # using the address passed by the custom `__reduce__` function
        try:
            from distributed import default_client

            default_client()
        except ValueError:
            from distributed import Client

            # setup `default_client` for worker process
            _ = Client(address)
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def __reduce__(self):  # noqa: GL08
        from distributed import default_client

        address = default_client().scheduler_info()["address"]
        return self.reconnect, (address, self.__dict__)
