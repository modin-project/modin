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
from pandas.core.dtypes.common import get_dtype, is_categorical_dtype

import modin.pandas as pd
from modin.pandas.utils import from_arrow
from modin.experimental.core.storage_formats.hdk import DFAlgQueryCompiler
from modin.experimental.core.execution.native.implementations.hdk_on_native.hdk_worker import (
    HdkWorker,
)


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
    worker = HdkWorker()
    if len(kwargs) > 0:
        query = _build_query(query, kwargs, worker.import_arrow_table)
    df = from_arrow(worker.executeDML(query))
    mdf = df._query_compiler._modin_frame
    at = mdf._partitions[0][0].get()
    schema = at.schema
    # HDK returns strings as dictionary. For the proper conversion to
    # Pandas, we need to replace dtypes of the corresponding columns.
    if replace := {
        i: f.name for i, f in enumerate(schema) if pa.types.is_dictionary(f.type)
    }:
        dtypes = mdf._dtypes
        obj_type = get_dtype(object)
        for i, n in replace.items():
            n = n[2:]  # Cut the F_ prefix
            skip = False
            # Make sure this column is not Categorical. It only works for the
            # original column names. If a column has been renamed in the query,
            # then the dtype is changed.
            for a in kwargs.values():
                if (
                    isinstance(a, pd.DataFrame)
                    and (n in (dt := a.dtypes))
                    and is_categorical_dtype(dt[n])
                ):
                    skip = True
                    break
            if not skip:
                dtypes[i] = obj_type
    return df


def _build_query(query: str, frames: dict, import_table: callable) -> str:
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
    import_table : callable
        Used to import tables and assign the table names.

    Returns
    -------
    str
        SQL query to be executed.
    """
    alias = []
    for name, df in frames.items():
        assert isinstance(df._query_compiler, DFAlgQueryCompiler)
        mf = df._query_compiler._modin_frame
        if not mf._has_arrow_table():
            mf._execute()
        assert mf._has_arrow_table()
        part = mf._partitions[0][0]
        at = part.get()

        if part.frame_id is None:
            part.frame_id = import_table(at)

        alias.append("WITH " if len(alias) == 0 else "\n),\n")
        alias.extend((name, " AS (\n", "  SELECT\n"))

        for i, col in enumerate(at.column_names):
            alias.append("    " if i == 0 else ",\n    ")
            # Cut the "F_" prefix from the column name
            alias.extend(('"', col, '"', " AS ", '"', col[2:], '"'))
        alias.extend(("\n  FROM\n    ", part.frame_id))

    alias.extend(("\n)\n", query))
    return "".join(alias)
