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

"""Module houses Modin parser classes, that are used for data parsing on the workers."""

import pandas
from io import BytesIO

from modin.core.storage_formats.pandas.utils import compute_chunksize


class PyarrowCSVParser:
    """Class for handling CSV files on the workers using PyArrow storage format."""

    def parse(self, fname, num_splits, start, end, header, **kwargs):
        """
        Parse CSV file into PyArrow tables.

        Parameters
        ----------
        fname : str
            Name of the CSV file to parse.
        num_splits : int
            Number of partitions to split the resulted PyArrow table into.
        start : int
            Position in the specified file to start parsing from.
        end : int
            Position in the specified file to end parsing at.
        header : str
            Header line that will be interpret as the first line of the parsed CSV file.
        **kwargs : kwargs
            Serves the compatibility purpose. Does not affect the result.

        Returns
        -------
        list
            List with split parse results and it's metadata:

            - First `num_split` elements are PyArrow tables, representing the corresponding chunk.
            - Next element is the number of rows in the parsed table.
            - Last element is the pandas Series, containing the data-types for each column of the parsed table.
        """
        import pyarrow as pa
        import pyarrow.csv as csv

        bio = open(fname, "rb")
        # The header line for the CSV file
        first_line = bio.readline()
        bio.seek(start)
        to_read = header + first_line + bio.read(end - start)
        bio.close()
        table = csv.read_csv(
            BytesIO(to_read), parse_options=csv.ParseOptions(header_rows=1)
        )
        chunksize = compute_chunksize(table.num_columns, num_splits)
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
