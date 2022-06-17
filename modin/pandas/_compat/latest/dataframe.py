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

from typing import Optional, Union, IO
import pandas
from pandas.util._validators import validate_bool_kwarg
from pandas._libs.lib import no_default
from pandas._typing import StorageOptions
from numpy import nan

from ..abc import BaseCompatibilityDataFrame


class LatestCompatibleDataFrame(BaseCompatibilityDataFrame):
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
        """
        Apply a function to a ``DataFrame`` elementwise.
        """
        return self._applymap(func, na_action=na_action, **kwargs)

    def apply(
        self, func, axis=0, raw=False, result_type=None, args=(), **kwargs
    ):  # noqa: PR01, RT01, D200
        return self._apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs
        )

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
    ):  # noqa: PR01, RT01, D200
        """
        Conform ``DataFrame`` to new index with optional filling logic.
        """
        axis = self._get_axis_number(axis)
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels
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
