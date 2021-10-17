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

"""Module houses `CSVDispatcher` class, that is used for reading `.csv` files."""

import pandas

from modin.core.io.text.text_file_dispatcher import TextFileDispatcher


class CSVDispatcher(TextFileDispatcher):
    """
    Class handles utils for reading `.csv` files.

    Inherits some common for text files util functions from `TextFileDispatcher` class.
    """

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        Read data from `filepath_or_buffer` according to `kwargs` parameters.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.
        **kwargs : dict
            Parameters of `read_csv` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `skiprows` is handled diferently based on the parameter type because of
        performance reasons. If `skiprows` is integer - rows will be skipped during
        data file partitioning and wouldn't be actually read. If `skiprows` is array
        or callable - full data file will be read and only then rows will be dropped.
        """
        return cls._generic_read(
            filepath_or_buffer,
            callback=pandas.read_csv,
            is_for_fwf_reader=False,
            **kwargs
        )
