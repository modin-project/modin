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

"""Compatibility layer for functions from top-level ``pandas`` namespace."""

from modin._compat import PandasCompatVersion

if PandasCompatVersion.CURRENT == PandasCompatVersion.PY36:
    from .py36.io import (
        read_csv,
        read_json,
        read_table,
        read_parquet,
        read_gbq,
        read_excel,
        read_html,
        read_pickle,
        read_stata,
        read_sas,
        read_feather,
        read_sql_query,
        to_pickle,
    )
    from .py36.general import pivot_table

    def read_xml(*args, **kw):
        from pandas import __version__

        raise NotImplementedError(f"Pandas {__version__} does not have read_xml")

    __all__ = [
        "read_csv",
        "read_json",
        "read_table",
        "read_parquet",
        "read_gbq",
        "read_excel",
        "read_html",
        "read_pickle",
        "read_stata",
        "read_sas",
        "read_feather",
        "read_sql_query",
        "to_pickle",
        "pivot_table",
        "read_xml",
    ]
elif PandasCompatVersion.CURRENT == PandasCompatVersion.LATEST:
    from .latest.io import (
        read_xml,
        read_csv,
        read_json,
        read_table,
        read_parquet,
        read_gbq,
        read_excel,
        read_html,
        read_pickle,
        read_sas,
        read_stata,
        read_feather,
        read_sql_query,
        to_pickle,
    )
    from .latest.general import pivot_table, from_dummies
    from pandas import Flags, Float32Dtype, Float64Dtype, ArrowDtype

    __all__ = [
        "from_dummies",
        "read_xml",
        "read_csv",
        "read_json",
        "read_table",
        "read_parquet",
        "read_gbq",
        "read_excel",
        "read_html",
        "read_pickle",
        "read_sas",
        "read_stata",
        "read_feather",
        "read_sql_query",
        "to_pickle",
        "pivot_table",
        "ArrowDtype",
        "Flags",
        "Float32Dtype",
        "Float64Dtype",
    ]
