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

"""Module for 'Python 3.6 pandas' compatibility layer for DataFrame."""

from typing import Union, Tuple, Mapping
import pandas
import pandas._libs.lib
from numpy import nan

from ..abc import BaseCompatibleDataFrame
from modin.utils import _inherit_docstrings


@_inherit_docstrings(pandas.DataFrame)
class Python36CompatibleDataFrame(BaseCompatibleDataFrame):  # noqa: PR01
    """Compatibility layer for 'Python 3.6 pandas' for DataFrame."""

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        query_compiler=None,
    ):
        self._init(
            data=data,
            index=index,
            columns=columns,
            dtype=dtype,
            copy=copy,
            query_compiler=query_compiler,
        )

    def applymap(self, func):  # noqa: PR01, RT01, D200
        return self._applymap(func)

    def apply(
        self, func, axis=0, raw=False, result_type=None, args=(), **kwds
    ):  # noqa: PR01, RT01, D200
        return self._apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
        )

    def compare(
        self,
        other,
        align_axis=1,
        keep_shape=False,
        keep_equal=False,
    ):
        return self._compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            # pass the value that would describe 'older pandas' behaviour for our query compiler
            result_names=("self", "other"),
        )

    def corr(self, method="pearson", min_periods=1):
        return self._corr(method=method, min_periods=min_periods, numeric_only=True)

    def corrwith(self, other, axis=0, drop=False, method="pearson"):
        return self._corrwith(other=other, axis=axis, drop=drop, method=method)

    def cov(self, min_periods=None, ddof=1):
        return self._cov(min_periods=min_periods, ddof=ddof, numeric_only=True)

    def explode(self, column: Union[str, Tuple], ignore_index: bool = False):
        exploded = self.__constructor__(
            query_compiler=self._query_compiler.explode(column)
        )
        if ignore_index:
            exploded = exploded.reset_index(drop=True)
        return exploded

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze: bool = pandas._libs.lib.no_default,
        observed=False,
        dropna: bool = True,
    ):
        return self._groupby(
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            dropna=dropna,
        )

    def info(
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None
    ):  # noqa: PR01, D200
        return self._info(
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            null_counts=null_counts,
            show_counts=None,
        )

    def insert(self, loc, column, value, allow_duplicates=False):
        return self._insert(
            loc=loc, column=column, value=value, allow_duplicates=allow_duplicates
        )

    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
    ):
        return self._join(
            other=other,
            on=on,
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            sort=sort,
            validate=None,
        )

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
        observed=False,
    ):  # noqa: PR01, RT01, D200
        return self._pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=None,
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        return self._prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation="linear"):
        return self._quantile(
            q=q,
            axis=axis,
            numeric_only=numeric_only,
            interpolation=interpolation,
            method="single",
        )

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=nan,
        limit=None,
        tolerance=None,
    ):
        index, columns = self._disambiguate_axes_labels(axis, index, columns, labels)
        return super(Python36CompatibleDataFrame, self).reindex(
            index=index,
            columns=columns,
            method=method,
            copy=copy,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
        )

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):
        return self._rename(
            mapper=mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):  # noqa: PR01, RT01, D200
        return self._replace(
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )

    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        return self._sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def to_gbq(
        self,
        destination_table,
        project_id=None,
        chunksize=None,
        reauth=False,
        if_exists="fail",
        auth_local_webserver=False,
        table_schema=None,
        location=None,
        progress_bar=True,
        credentials=None,
    ):
        return self._to_gbq(
            destination_table=destination_table,
            project_id=project_id,
            chunksize=chunksize,
            reauth=reauth,
            if_exists=if_exists,
            auth_local_webserver=auth_local_webserver,
            table_schema=table_schema,
            location=location,
            progress_bar=progress_bar,
            credentials=credentials,
        )

    def to_parquet(
        self,
        path,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        **kwargs,
    ):
        config = {
            "path": path,
            "engine": engine,
            "compression": compression,
            "index": index,
            "partition_cols": partition_cols,
        }
        new_query_compiler = self._query_compiler

        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_parquet(new_query_compiler, **config, **kwargs)

    def to_stata(
        self,
        path,
        convert_dates=None,
        write_index=True,
        byteorder=None,
        time_stamp=None,
        data_label=None,
        variable_labels=None,
        version=114,
        convert_strl=None,
        compression: Union[str, Mapping[str, str], None] = "infer",
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_stata,
            path,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            version=version,
            convert_strl=convert_strl,
            compression=compression,
        )

    def where(
        self,
        cond,
        other=nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):  # noqa: PR01, RT01, D200
        return self._where(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
        )
