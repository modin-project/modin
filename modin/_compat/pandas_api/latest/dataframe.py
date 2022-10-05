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

"""Module for 'latest pandas' compatibility layer for DataFrame."""

from datetime import datetime
from typing import (
    Optional,
    Union,
    IO,
    Hashable,
    Sequence,
)
import pandas
from pandas.util._validators import validate_bool_kwarg
from pandas._libs.lib import no_default, NoDefault
from pandas._typing import (
    CompressionOptions,
    FilePath,
    StorageOptions,
    WriteBuffer,
    Axis,
    Suffixes,
)
from numpy import nan

from ..abc import BaseCompatibleDataFrame


class LatestCompatibleDataFrame(BaseCompatibleDataFrame):  # noqa: PR01
    """Compatibility layer for 'latest pandas' for DataFrame."""

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=None,
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

    def applymap(
        self, func, na_action: Optional[str] = None, **kwargs
    ):  # noqa: PR01, RT01, D200
        return self._applymap(func, na_action=na_action, **kwargs)

    def apply(
        self, func, axis=0, raw=False, result_type=None, args=(), **kwargs
    ):  # noqa: PR01, RT01, D200
        return self._apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs
        )

    def compare(
        self,
        other,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ):
        return self._compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    def corr(self, method="pearson", min_periods=1, numeric_only=no_default):
        return self._corr(
            method=method, min_periods=min_periods, numeric_only=numeric_only
        )

    def corrwith(
        self, other, axis=0, drop=False, method="pearson", numeric_only=no_default
    ):
        return self._corrwith(
            other=other, axis=axis, drop=drop, method=method, numeric_only=numeric_only
        )

    def cov(self, min_periods=None, ddof: Optional[int] = 1, numeric_only=no_default):
        return self._cov(min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=no_default,
        squeeze: bool = no_default,
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

    def idxmax(self, axis=0, skipna=True, numeric_only=False):
        return self._idxmax(axis=axis, skipna=skipna, numeric_only=numeric_only)

    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        return self._idxmin(axis=axis, skipna=skipna, numeric_only=numeric_only)

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[IO[str]] = None,
        max_cols: Optional[int] = None,
        memory_usage: Optional[Union[bool, str]] = None,
        show_counts: Optional[bool] = None,
        null_counts: Optional[bool] = None,
    ):  # noqa: PR01, D200
        return self._info(
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
            null_counts=null_counts,
        )

    def insert(self, loc, column, value, allow_duplicates=no_default):
        return self._insert(
            loc=loc,
            column=column,
            value=value,
            allow_duplicates=False
            if allow_duplicates is no_default
            else allow_duplicates,
        )

    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        validate=None,
    ):
        return self._join(
            other=other,
            on=on,
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            sort=sort,
            validate=validate,
        )

    def isetitem(self, loc, value):
        return self._default_to_pandas(
            pandas.DataFrame.isetitem,
            loc=loc,
            value=value,
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
        sort=True,
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
            sort=sort,
        )

    def prod(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def quantile(
        self,
        q=0.5,
        axis=0,
        numeric_only=no_default,
        interpolation="linear",
        method="single",
    ):
        return self._quantile(
            q=q,
            axis=axis,
            numeric_only=True if numeric_only is no_default else numeric_only,
            interpolation=interpolation,
            method=method,
        )

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=None,
        level=None,
        fill_value=nan,
        limit=None,
        tolerance=None,
    ):  # noqa: PR01, RT01, D200
        index, columns = self._disambiguate_axes_labels(axis, index, columns, labels)
        return super(LatestCompatibleDataFrame, self).reindex(
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
        copy=None,
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
        value=no_default,
        inplace: "bool" = False,
        limit=None,
        regex: "bool" = False,
        method: "str | NoDefault" = no_default,
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
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
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
        auth_local_webserver=True,
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

    def to_orc(self, path=None, *, engine="pyarrow", index=None, engine_kwargs=None):
        return self._default_to_pandas(
            pandas.DataFrame.to_orc,
            path=path,
            engine=engine,
            index=index,
            engine_kwargs=engine_kwargs,
        )

    def to_parquet(
        self,
        path=None,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions = None,
        **kwargs,
    ):
        config = {
            "path": path,
            "engine": engine,
            "compression": compression,
            "index": index,
            "partition_cols": partition_cols,
            "storage_options": storage_options,
        }
        new_query_compiler = self._query_compiler

        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_parquet(new_query_compiler, **config, **kwargs)

    def to_stata(
        self,
        path: "FilePath | WriteBuffer[bytes]",
        convert_dates: "dict[Hashable, str] | None" = None,
        write_index: "bool" = True,
        byteorder: "str | None" = None,
        time_stamp: "datetime.datetime | None" = None,
        data_label: "str | None" = None,
        variable_labels: "dict[Hashable, str] | None" = None,
        version: "int | None" = 114,
        convert_strl: "Sequence[Hashable] | None" = None,
        compression: "CompressionOptions" = "infer",
        storage_options: "StorageOptions" = None,
        *,
        value_labels: "dict[Hashable, dict[float | int, str]] | None" = None,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
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
            storage_options=storage_options,
            value_labels=value_labels,
        )

    def to_xml(
        self,
        path_or_buffer=None,
        index=True,
        root_name="data",
        row_name="row",
        na_rep=None,
        attr_cols=None,
        elem_cols=None,
        namespaces=None,
        prefix=None,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
        parser="lxml",
        stylesheet=None,
        compression="infer",
        storage_options=None,
    ):
        return self.__constructor__(
            query_compiler=self._query_compiler.default_to_pandas(
                pandas.DataFrame.to_xml,
                path_or_buffer=path_or_buffer,
                index=index,
                root_name=root_name,
                row_name=row_name,
                na_rep=na_rep,
                attr_cols=attr_cols,
                elem_cols=elem_cols,
                namespaces=namespaces,
                prefix=prefix,
                encoding=encoding,
                xml_declaration=xml_declaration,
                pretty_print=pretty_print,
                parser=parser,
                stylesheet=stylesheet,
                compression=compression,
                storage_options=storage_options,
            )
        )

    def where(
        self,
        cond,
        other=no_default,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=no_default,
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
