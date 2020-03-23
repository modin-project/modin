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

from modin.data_management.utils import get_default_chunksize
from io import BytesIO
import pandas


class PyarrowCSVParser:
    def parse(self, **kwargs):
        import pyarrow as pa
        import pyarrow.csv as csv

        fname = kwargs.pop("fname", None)
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        header = kwargs.pop("header", None)
        bio = open(fname, "rb")
        # The header line for the CSV file
        first_line = bio.readline()
        bio.seek(start)
        to_read = header + first_line + bio.read(end - start)
        bio.close()
        table = csv.read_csv(
            BytesIO(to_read), parse_options=csv.ParseOptions(header_rows=1)
        )
        chunksize = get_default_chunksize(table.num_columns, num_splits)
        chunks = [
            pa.Table.from_arrays(table.columns[chunksize * i : chunksize * (i + 1)])
            for i in range(num_splits)
        ]
        return chunks + [
            table.num_rows,
            pandas.Series(
                [t.to_pandas_dtype() for t in table.schema.types],
                index=table.schema.names,
            ),
        ]
