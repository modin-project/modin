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

import modin.config as cfg
import modin.pandas as pd

_query_impl = None


def query(sql: str, *args, **kwargs) -> pd.DataFrame:
    """
    Execute SQL query using HDK engine.

    Parameters
    ----------
    sql : str
        SQL query to be executed.
    *args : *tuple
        Positional arguments, passed to the execution engine.
    **kwargs : **dict
        Keyword arguments, passed to the execution engine.

    Returns
    -------
    modin.pandas.DataFrame
        Execution result.
    """
    global _query_impl

    if _query_impl is None:
        if cfg.StorageFormat.get() == "Hdk":
            from modin.experimental.sql.hdk.query import hdk_query as _query_impl
        else:
            raise NotImplementedError

    return _query_impl(sql, *args, **kwargs)
