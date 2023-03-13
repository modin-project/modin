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

import pyhdk

from .base_worker import BaseDbWorker

from modin.utils import _inherit_docstrings
from modin.config import HdkLaunchParameters


@_inherit_docstrings(BaseDbWorker)
class HdkWorker(BaseDbWorker):
    """PyHDK based wrapper class for HDK storage format."""

    _config = None
    _storage = None
    _data_mgr = None
    _calcite = None
    _executor = None

    @classmethod
    def setup_engine(cls):
        """
        Initialize PyHDK.

        Do nothing if it is initiliazed already.
        """
        if cls._executor is None:
            cls._config = pyhdk.buildConfig(**HdkLaunchParameters.get())
            cls._storage = pyhdk.storage.ArrowStorage(1)
            cls._data_mgr = pyhdk.storage.DataMgr(cls._config)
            cls._data_mgr.registerDataProvider(cls._storage)

            cls._calcite = pyhdk.sql.Calcite(cls._storage, cls._config)
            cls._executor = pyhdk.Executor(cls._data_mgr, cls._config)

    def __init__(self):
        """Initialize HDK storage format."""
        self.setup_engine()

    @classmethod
    def dropTable(cls, name):
        cls._storage.dropTable(name)

    @classmethod
    def _executeRelAlgJson(cls, ra):
        """
        Execute RelAlg JSON query.

        Parameters
        ----------
        ra : str
            RelAlg JSON string.

        Returns
        -------
        pyarrow.Table
            Execution result.
        """
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            cls._executor, cls._storage, cls._data_mgr, ra
        )
        res = rel_alg_executor.execute()
        return res.to_arrow()

    @classmethod
    def executeDML(cls, query):
        ra = cls._calcite.process(query, db_name="hdk")
        return cls._executeRelAlgJson(ra)

    @classmethod
    def executeRA(cls, query):
        if query.startswith("execute relalg"):
            ra = query.removeprefix("execute relalg")
        else:
            assert query.startswith("execute calcite")
            ra = cls._calcite.process(query, db_name="hdk")
        return cls._executeRelAlgJson(ra)

    @classmethod
    def import_arrow_table(cls, table, name=None):
        name = cls._genName(name)

        table = cls.cast_to_compatible_types(table)
        fragment_size = cls.compute_fragment_size(table)

        opt = pyhdk.storage.TableOptions(fragment_size)
        cls._storage.importArrowTable(table, name, opt)

        return name
