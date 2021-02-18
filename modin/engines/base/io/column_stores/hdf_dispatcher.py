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

import pandas

from modin.engines.base.io.column_stores.column_store_dispatcher import (
    ColumnStoreDispatcher,
)
from modin.error_message import ErrorMessage


class HDFDispatcher(ColumnStoreDispatcher):  # pragma: no cover
    @classmethod
    def _validate_hdf_format(cls, path_or_buf):
        s = pandas.HDFStore(path_or_buf)
        groups = s.groups()
        if len(groups) == 0:
            raise ValueError("No dataset in HDF5 file.")
        candidate_only_group = groups[0]
        format = getattr(candidate_only_group._v_attrs, "table_type", None)
        s.close()
        return format

    @classmethod
    def _read(cls, path_or_buf, **kwargs):
        """Load a h5 file from the file path or buffer, returning a DataFrame.

        Args:
            path: string, buffer or path object
                Path to the file to open, or an open :class:`pandas.HDFStore` object.
            kwargs: Pass into pandas.read_hdf function.

        Returns:
            DataFrame constructed from the h5 file.
        """
        if cls._validate_hdf_format(path_or_buf=path_or_buf) is None:
            ErrorMessage.default_to_pandas(
                "File format seems to be `fixed`. For better distribution consider "
                "saving the file in `table` format. df.to_hdf(format=`table`)."
            )
            return cls.single_worker_read(path_or_buf, **kwargs)

        columns = kwargs.pop("columns", None)
        # Have to do this because of Dask's keyword arguments
        kwargs["_key"] = kwargs.pop("key", None)
        if not columns:
            start = kwargs.pop("start", None)
            stop = kwargs.pop("stop", None)
            empty_pd_df = pandas.read_hdf(path_or_buf, start=0, stop=0, **kwargs)
            if start is not None:
                kwargs["start"] = start
            if stop is not None:
                kwargs["stop"] = stop
            columns = empty_pd_df.columns
        return cls.build_query_compiler(path_or_buf, columns, **kwargs)
