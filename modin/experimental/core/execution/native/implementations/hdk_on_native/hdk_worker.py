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

"""Module provides ``HdkWorker`` class."""
from typing import Optional, Tuple, List, Union

import pyarrow as pa
import os

from pyhdk.hdk import HDK, QueryNode, ExecutionResult, RelAlgExecutor

from .base_worker import DbTable, BaseDbWorker

from modin.utils import _inherit_docstrings
from modin.config import HdkLaunchParameters, OmnisciFragmentSize, HdkFragmentSize


class HdkTable(DbTable):
    """
    Represents a table in the HDK database.

    Parameters
    ----------
    table : QueryNode or ExecutionResult
    """

    def __init__(self, table: Union[QueryNode, ExecutionResult]):
        self.name = table.table_name
        self._table = table

    def __del__(self):
        """Drop table."""
        # The ExecutionResults are cleared by HDK.
        if not isinstance(self._table, ExecutionResult):
            HdkWorker.dropTable(self.name)

    @property
    @_inherit_docstrings(DbTable.shape)
    def shape(self) -> Tuple[int, int]:
        shape = getattr(self, "_shape", None)
        if shape is None:
            self._shape = shape = self.scan().shape
        return shape

    @property
    @_inherit_docstrings(DbTable.column_names)
    def column_names(self) -> List[str]:
        names = getattr(self, "_column_names", None)
        if names is None:
            self._column_names = names = list(self.scan().schema)
        return names

    @_inherit_docstrings(DbTable.to_arrow)
    def to_arrow(self) -> pa.Table:
        return (
            self._table.to_arrow()
            if isinstance(self._table, ExecutionResult)
            else self._table.run().to_arrow()
        )

    def scan(self):
        """
        Return a scan query node referencing this table.

        Returns
        -------
        QueryNode
        """
        if isinstance(self._table, QueryNode):
            return self._table
        scan = getattr(self, "_scan", None)
        if scan is None:
            self._scan = scan = HdkWorker._hdk().scan(self.name)
        return scan


@_inherit_docstrings(BaseDbWorker)
class HdkWorker(BaseDbWorker):  # noqa: PR01
    """PyHDK based wrapper class for HDK storage format."""

    def __new__(cls, *args, **kwargs):
        instance = getattr(cls, "_instance", None)
        if instance is None:
            cls._instance = instance = object.__new__(cls)
        return instance

    @classmethod
    def dropTable(cls, name: str):
        cls.dropTable = cls._hdk().drop_table
        cls.dropTable(name)

    @classmethod
    def executeDML(cls, query: str):
        return cls.executeRA(query, True)

    @classmethod
    def executeRA(cls, query: str, exec_calcite=False, **exec_args):
        hdk = cls._hdk()
        if exec_calcite or query.startswith("execute calcite"):
            ra = hdk._calcite.process(query, db_name="hdk", legacy_syntax=True)
        else:
            ra = query
        ra_executor = RelAlgExecutor(hdk._executor, hdk._schema_mgr, hdk._data_mgr, ra)
        table = ra_executor.execute(device_type=cls._preferred_device, **exec_args)
        return HdkTable(table)

    @classmethod
    def import_arrow_table(cls, table: pa.Table, name: Optional[str] = None):
        name = cls._genName(name)
        table = cls.cast_to_compatible_types(table)
        fragment_size = cls.compute_fragment_size(table)
        return HdkTable(cls._hdk().import_arrow(table, name, fragment_size))

    @classmethod
    def compute_fragment_size(cls, table):
        """
        Compute fragment size to be used for table import.

        Parameters
        ----------
        table : pyarrow.Table
            A table to import.

        Returns
        -------
        int
            Fragment size to use for import.
        """
        fragment_size = HdkFragmentSize.get()
        if fragment_size is None:
            fragment_size = OmnisciFragmentSize.get()
        if fragment_size is None:
            if bool(HdkLaunchParameters.get()["cpu_only"]):
                cpu_count = os.cpu_count()
                if cpu_count is not None:
                    fragment_size = table.num_rows // cpu_count
                    fragment_size = min(fragment_size, 2**25)
                    fragment_size = max(fragment_size, 2**18)
                else:
                    fragment_size = 0
            else:
                fragment_size = 2**25
        else:
            fragment_size = int(fragment_size)
        return fragment_size

    @classmethod
    def _hdk(cls) -> HDK:
        """
        Initialize and return an HDK instance.

        Returns
        -------
        HDK
        """
        params = HdkLaunchParameters.get()
        cls._preferred_device = "CPU" if bool(params["cpu_only"]) else "GPU"
        cls._hdk_instance = HDK(**params)
        cls._hdk = cls._get_hdk_instance
        return cls._hdk()

    @classmethod
    def _get_hdk_instance(cls) -> HDK:
        """
        Return the initialized HDK instance.

        Returns
        -------
        HDK
        """
        return cls._hdk_instance
