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
    """Class handles utils for reading `.csv` files."""

    def read_callback(*args, **kwargs):
        """
        Parse data on each partition.

        Parameters
        ----------
        *args : list
            Positional arguments to be passed to the callback function.
        **kwargs : dict
            Keyword arguments to be passed to the callback function.

        Returns
        -------
        pandas.DataFrame or pandas.io.parsers.TextParser
            Function call result.
        """
        return pandas.read_csv(*args, **kwargs)
