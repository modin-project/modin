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

"""Module provides ``OmnisciWorker`` class."""

import pyarrow

from .base_worker import BaseDbWorker
import os
import sys

if sys.platform == "linux":
    prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

try:
    from omniscidbe import PyDbEngine  # noqa
except ModuleNotFoundError:  # fallback for older omniscidbe4py package naming
    from dbe import PyDbEngine  # noqa
finally:
    if sys.platform == "linux":
        sys.setdlopenflags(prev)

from modin.utils import _inherit_docstrings
from modin.config import OmnisciLaunchParameters


@_inherit_docstrings(BaseDbWorker)
class OmnisciWorker(BaseDbWorker):
    """PyOmniSciDbe based wrapper class for OmniSci storage format."""

    _server = None

    @classmethod
    def start_server(cls):
        """
        Initialize OmniSci server.

        Do nothing if it is initialized already.
        """
        if cls._server is None:
            cls._server = PyDbEngine(**OmnisciLaunchParameters.get())

    @classmethod
    def stop_server(cls):
        """Destroy OmniSci server if any."""
        if cls._server is not None:
            cls._server.reset()
            cls._server = None

    def __init__(self):
        """Initialize OmniSci storage format."""
        self.start_server()

    @classmethod
    def dropTable(cls, name):
        cls._server.executeDDL(f"DROP TABLE IF EXISTS {name};")

    @classmethod
    def _resToArrow(cls, curs):
        """
        Convert execution result to Arrow format.

        Parameters
        ----------
        curs : omniscidbe.PyCursor
            DML/RA execution result.

        Returns
        -------
        pyarrow.Table
            Converted execution result.
        """
        assert curs
        if hasattr(curs, "getArrowTable"):
            at = curs.getArrowTable()
        else:
            rb = curs.getArrowRecordBatch()
            assert rb is not None
            at = pyarrow.Table.from_batches([rb])
        assert at is not None
        return at

    @classmethod
    def executeDML(cls, query):
        r = cls._server.executeDML(query)
        # todo: assert r
        return cls._resToArrow(r)

    @classmethod
    def executeRA(cls, query):
        r = cls._server.executeRA(query)
        # todo: assert r
        return cls._resToArrow(r)

    @classmethod
    def import_arrow_table(cls, table, name=None):
        name = cls._genName(name)

        table = cls.cast_to_compatible_types(table)
        fragment_size = cls.compute_fragment_size(table)

        cls._server.importArrowTable(name, table, fragment_size=fragment_size)

        return name
