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

"""Module houses `FWFDispatcher` class, that is used for reading of tables with fixed-width formatted lines."""

import pandas

from modin.core.io.text.text_file_dispatcher import TextFileDispatcher


class FWFDispatcher(TextFileDispatcher):
    """Class handles utils for reading of tables with fixed-width formatted lines."""

    read_callback = pandas.read_fwf

    @classmethod
    def check_parameters_support(
        cls,
        filepath_or_buffer,
        read_kwargs: dict,
    ):
        """
        Check support of parameters of `read_fwf` function.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_fwf` function.
        read_kwargs : dict
            Parameters of `read_fwf` function.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        """
        if read_kwargs["infer_nrows"] > 100:
            # If infer_nrows is a significant portion of the number of rows, pandas may be
            # faster.
            return False
        return super().check_parameters_support(filepath_or_buffer, read_kwargs)
