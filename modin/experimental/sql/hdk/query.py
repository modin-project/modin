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

import pyarrow as pa
from pandas.core.dtypes.common import _get_dtype

import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.utils import (
    ColNameCodec,
)
from modin.experimental.core.execution.native.implementations.hdk_on_native.hdk_worker import (
    HdkWorker,
)
from modin.experimental.core.storage_formats.hdk import DFAlgQueryCompiler
from modin.pandas.utils import from_arrow


def hdk_query(query: str, **kwargs) -> pd.DataFrame:
    """
    Execute SQL queries on the HDK backend.

    DataFrames are referenced in the query by names and are
    passed to this function as name=value arguments.

    Here is an example of a query to three data frames:

    ids = [1, 2, 3]
    first_names = ["James", "Peter", "Claus"]
    last_names = ["Bond", "Pan", "Santa"]
    courses_names = ["Mathematics", "Physics", "Geography"]
    student = pd.DataFrame({"id": ids, "first_name": first_names, "last_name": last_names})
    course = pd.DataFrame({"id": ids, "course_name": courses_names})
    student_course = pd.DataFrame({"student_id": ids, "course_id": [3, 2, 1]})
    query = '''
    SELECT
        student.first_name,
        student.last_name,
        course.course_name
    FROM student
    JOIN student_course
    ON student.id = student_course.student_id
    JOIN course
    ON course.id = student_course.course_id
    ORDER BY
        last_name
    '''
    res = hdk_query(query, student=student, course=course, student_course=student_course)
    print(res)

    Parameters
    ----------
    query : str
        SQL query to be executed.
    **kwargs : **dict
        DataFrames referenced by the query.

    Returns
    -------
    modin.pandas.DataFrame
        Execution result.
    """
    if len(kwargs) > 0:
        query = _build_query(query, kwargs)
    table = HdkWorker().executeDML(query)
    df = from_arrow(table.to_arrow())
    mdf = df._query_compiler._modin_frame
    schema = mdf._partitions[0][0].get().schema
    # HDK returns strings as dictionary. For the proper conversion to
    # Pandas, we need to replace dtypes of the corresponding columns.
    if replace := [
        i for i, col in enumerate(schema) if pa.types.is_dictionary(col.type)
    ]:
        dtypes = mdf._dtypes
        obj_type = _get_dtype(object)
        for i in replace:
            dtypes[i] = obj_type
    return df


def _build_query(query: str, frames: dict) -> str:
    """
    Build query to be executed.

    Table and column names are mapped to the real names
    using the WITH statement.

    Parameters
    ----------
    query : str
        SQL query to be processed.
    frames : dict
        DataFrames referenced by the query.

    Returns
    -------
    str
        SQL query to be executed.
    """
    alias = []
    for name, df in frames.items():
        assert isinstance(df._query_compiler, DFAlgQueryCompiler)
        mf = df._query_compiler._modin_frame
        table = mf.force_import()
        alias.append("WITH " if len(alias) == 0 else "\n),\n")
        alias.extend((name, " AS (\n", "  SELECT\n"))

        for i, col in enumerate(table.column_names):
            alias.append("    " if i == 0 else ",\n    ")
            alias.extend(('"', col, '"', " AS ", '"', ColNameCodec.decode(col), '"'))
        alias.extend(("\n  FROM\n    ", table.name))

    alias.extend(("\n)\n", query))
    return "".join(alias)
