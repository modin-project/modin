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


class PartitionUnwrapper(object):
    """Class responsible for unwrapping axis partitions"""

    @classmethod
    def unwrap(cls, query_compiler, axis, engine=None, bind_ip=False):
        """
        Unwrap axis partitions of an API layer object using its `query_compiler`.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            The query compiler of an API layer object.
        axis : 0 or 1
            The axis to unwrap partitions for (0 - row partitions, 1 - column partitions).
        engine : str
            The engine of the underlying partitions ("Ray", "Dask").
        bind_ip : boolean, default False
            Whether to bind node ip address to each axis partition or not.

        Returns
        -------
        list
            A list of Ray.ObjectRef to axis partitions of an `api_layer_object` if engine="Ray".
            A list of Dask.Future to axis partitions of an `api_layer_object` if engine="Dask".

        Notes
        -----
        In case bind_ip=True, a list containing tuples of Ray.ObjectRef to node ip address
        and axis partitions of an `api_layer_object`, respectively, is returned if engine="Ray".

        In case bind_ip=True, a list containing tuples of Dask.Future to node ip address
        and axis partitions of an `api_layer_object`, respectively, is returned if engine="Dask".
        """
        if (
            engine is not None
            and engine
            not in type(query_compiler._modin_frame._partitions[0][0]).__name__
        ):
            raise ValueError("Engine does not match underlying partitions")
        partitions = query_compiler._modin_frame._frame_mgr_cls.axis_partition(
            query_compiler._modin_frame._partitions, axis ^ 1
        )
        return [
            part.coalesce(bind_ip=bind_ip).unwrap(squeeze=True, bind_ip=bind_ip)
            for part in partitions
        ]


def unwrap_row_partitions(api_layer_object, engine=None, bind_ip=False):
    """
    Unwrap row partitions of the `api_layer_object`.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    engine : str
        The engine of the underlying partitions ("Ray", "Dask").
    bind_ip : boolean, default False
        Whether to bind node ip address to each row partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef to row partitions of the `api_layer_object` if engine="Ray".
        A list of Dask.Future to row partitions of the `api_layer_object` if engine="Dask".

    Notes
    -----
    In case bind_ip=True, a list containing tuples of Ray.ObjectRef to node ip address
    and row partitions of the `api_layer_object`, respectively, is returned if engine="Ray".

    If bind_ip=True, a list containing tuples of Dask.Future to node ip address
    and row partitions of the `api_layer_object`, respectively, is returned if engine="Dask".
    """
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError("Only API Layer objects may be passed in here.")
    return PartitionUnwrapper.unwrap(
        api_layer_object._query_compiler, 0, engine=engine, bind_ip=bind_ip
    )


def unwrap_column_partitions(api_layer_object, engine=None, bind_ip=False):
    """
    Unwrap column partitions of the `api_layer_object`.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    engine : str
        The engine of the underlying partitions ("Ray", "Dask").
    bind_ip : boolean, default False
        Whether to bind node ip address to each column partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef to column partitions of the `api_layer_object` if engine="Ray".
        A list of Dask.Future to column partitions of the `api_layer_object` if engine="Dask".

    Notes
    -----
    In case bind_ip=True, a list containing tuples of Ray.ObjectRef to node ip address
    and column partitions of the `api_layer_object`, respectively, is returned if engine="Ray".

    In case bind_ip=True, a list containing tuples of Dask.Future to node ip address
    and column partitions of the `api_layer_object`, respectively, is returned if engine="Dask".
    """
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError("Only API Layer objects may be passed in here.")
    return PartitionUnwrapper.unwrap(
        api_layer_object._query_compiler, 1, engine=engine, bind_ip=bind_ip
    )
