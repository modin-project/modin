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

"""Module for 'latest pandas' compatibility layer for Dataset (common DataFrame/Series)."""

import pandas
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype
from pandas.core.window.ewm import ExponentialMovingWindow
from pandas.util._validators import validate_bool_kwarg, validate_ascending
from pandas._libs.lib import no_default, NoDefault
from pandas._typing import (
    StorageOptions,
    CompressionOptions,
    IndexLabel,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    Axis,
)
import pickle as pkl
import numpy as np
from typing import Sequence, Hashable, Optional, TYPE_CHECKING, Union

from ..abc import BaseCompatibleBasePandasDataset
from .utils import create_stat_method
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.pandas.base import BasePandasDataset

nan = np.nan

# See https://github.com/pandas-dev/pandas/blob/v1.4.3/pandas/core/generic.py#L195
bool_t = bool


class LatestCompatibleBasePandasDataset(BaseCompatibleBasePandasDataset):
    """Compatibility layer for 'latest pandas' for Dataset."""

    @_inherit_docstrings(validate_ascending)
    def _validate_ascending(self, ascending):
        return validate_ascending(ascending)

    @_inherit_docstrings(validate_bool_kwarg)
    def _validate_bool_kwarg(self, value, arg_name, **kwargs):
        return validate_bool_kwarg(value, arg_name, **kwargs)

    def between_time(
        self: "BasePandasDataset",
        start_time,
        end_time,
        include_start: "bool_t | NoDefault" = no_default,
        include_end: "bool_t | NoDefault" = no_default,
        inclusive: "str | None" = None,
        axis=None,
    ):
        return self._between_time(
            start_time=start_time,
            end_time=end_time,
            include_start=include_start,
            include_end=include_end,
            inclusive=inclusive,
            axis=axis,
        )

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
    ):
        return self._convert_dtypes(
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
            convert_floating=convert_floating,
        )

    def dropna(
        self, axis=0, how=no_default, thresh=no_default, subset=None, inplace=False
    ):
        return self._dropna(
            axis=axis,
            how=how,
            thresh=thresh,
            subset=subset,
            inplace=inplace,
        )

    @_inherit_docstrings(
        parent=pandas.DataFrame.explode, apilink="pandas.DataFrame.explode"
    )
    def explode(self, column: IndexLabel, ignore_index: bool = False):
        exploded = self.__constructor__(
            query_compiler=self._query_compiler.explode(column)
        )
        if ignore_index:
            exploded = exploded.reset_index(drop=True)
        return exploded

    def ewm(
        self,
        com: "float | None" = None,
        span: "float | None" = None,
        halflife: "float | TimedeltaConvertibleTypes | None" = None,
        alpha: "float | None" = None,
        min_periods: "int | None" = 0,
        adjust: "bool_t" = True,
        ignore_na: "bool_t" = False,
        axis: "Axis" = 0,
        times: "str | np.ndarray | BasePandasDataset | None" = None,
        method: "str" = "single",
    ) -> "ExponentialMovingWindow":
        return self._ewm(
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
            method=method,
        )

    def expanding(self, min_periods=1, center=None, axis=0, method="single"):
        return self._expanding(
            min_periods=min_periods, center=center, axis=axis, method=method
        )

    def kurt(
        self,
        axis: "Axis | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._kurt(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def mad(self, axis=None, skipna=True, level=None):
        validate_bool_kwarg(skipna, "skipna", none_allowed=True)
        return self._mad(axis=axis, skipna=skipna, level=level)

    def mask(
        self,
        cond,
        other=nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=no_default,
    ):
        return self._mask(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
        )

    def max(
        self,
        axis: "int | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._max(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def min(
        self,
        axis: "int | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._min(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    mean = create_stat_method("mean")
    median = create_stat_method("median")

    def rank(
        self: "BasePandasDataset",
        axis=0,
        method: "str" = "average",
        numeric_only: "bool_t | None | NoDefault" = no_default,
        na_option: "str" = "keep",
        ascending: "bool_t" = True,
        pct: "bool_t" = False,
    ):
        return self._rank(
            axis=axis,
            method=method,
            numeric_only=numeric_only,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    def reindex(
        self,
        index=None,
        columns=None,
        copy=True,
        **kwargs,
    ):
        return self._reindex(index=index, columns=columns, copy=copy, **kwargs)

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base: Optional[int] = None,
        on=None,
        level=None,
        origin: Union[str, TimestampConvertibleTypes] = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
        group_keys=no_default,
    ):  # noqa: PR01, RT01, D200
        return self._resample(
            rule=rule,
            axis=axis,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            loffset=loffset,
            base=base,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
            group_keys=None if group_keys is no_default else group_keys,
        )

    def reset_index(
        self,
        level=None,
        drop=False,
        inplace=False,
        col_level=0,
        col_fill="",
        allow_duplicates=no_default,
        names=None,
    ):
        return self._reset_index(
            level=level,
            drop=drop,
            inplace=inplace,
            col_level=col_level,
            col_fill=col_fill,
            allow_duplicates=None
            if allow_duplicates is no_default
            else allow_duplicates,
            names=names,
        )

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method="single",
    ):
        return self._rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
            step=step,
            method=method,
        )

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        ignore_index=False,
    ):
        return self._sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis,
            ignore_index=ignore_index,
        )

    def set_axis(self, labels, axis=0, inplace=no_default, *, copy=no_default):
        return self._set_axis(
            labels=labels,
            axis=axis,
            inplace=False if inplace is no_default else inplace,
            copy=True if copy is no_default else copy,
        )

    def sem(
        self,
        axis=None,
        skipna=True,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        return self._sem(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def set_flags(
        self, *, copy: bool = False, allows_duplicate_labels: Optional[bool] = None
    ):
        return self._default_to_pandas(
            pandas.DataFrame.set_flags,
            copy=copy,
            allows_duplicate_labels=allows_duplicate_labels,
        )

    @property
    def flags(self):
        def flags(df):
            return df.flags

        return self._default_to_pandas(flags)

    def shift(self, periods=1, freq=None, axis=0, fill_value=no_default):
        if fill_value is no_default:
            nan_values = dict()
            for name, dtype in dict(self.dtypes).items():
                nan_values[name] = (
                    pandas.NAT if is_datetime_or_timedelta_dtype(dtype) else pandas.NA
                )

            fill_value = nan_values
        return self._shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

    def skew(
        self,
        axis: "int | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._skew(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def std(
        self,
        axis=None,
        skipna=True,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        return self._std(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors: str = "strict",
        storage_options: StorageOptions = None,
    ):  # pragma: no cover
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_csv(
            self._query_compiler,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            mode=mode,
            encoding=encoding,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
            errors=errors,
            storage_options=storage_options,
        )

    def to_excel(
        self,
        excel_writer,
        sheet_name="Sheet1",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells=True,
        encoding=no_default,
        inf_rep="inf",
        verbose=no_default,
        freeze_panes=None,
        storage_options: StorageOptions = None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_excel",
            excel_writer,
            sheet_name=sheet_name,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            startrow=startrow,
            startcol=startcol,
            engine=engine,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            freeze_panes=freeze_panes,
            storage_options=storage_options,
        )

    def to_json(
        self,
        path_or_buf=None,
        orient=None,
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=False,
        compression="infer",
        index=True,
        indent=None,
        storage_options: StorageOptions = None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_json",
            path_or_buf,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
            indent=indent,
            storage_options=storage_options,
        )

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        bold_rows=False,
        column_format=None,
        longtable=None,
        escape=None,
        encoding=None,
        decimal=".",
        multicolumn=None,
        multicolumn_format=None,
        multirow=None,
        caption=None,
        label=None,
        position=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_latex",
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            bold_rows=bold_rows,
            column_format=column_format,
            longtable=longtable,
            escape=escape,
            encoding=encoding,
            decimal=decimal,
            multicolumn=multicolumn,
            multicolumn_format=multicolumn_format,
            multirow=multirow,
            caption=None,
            label=None,
        )

    def to_markdown(
        self,
        buf=None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions = None,
        **kwargs,
    ):
        return self._default_to_pandas(
            "to_markdown",
            buf=buf,
            mode=mode,
            index=index,
            storage_options=storage_options,
            **kwargs,
        )

    def to_pickle(
        self,
        path,
        compression: CompressionOptions = "infer",
        protocol: int = pkl.HIGHEST_PROTOCOL,
        storage_options: StorageOptions = None,
    ):  # pragma: no cover
        from modin.pandas import to_pickle

        to_pickle(
            self,
            path,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )

    def value_counts(
        self,
        subset: Sequence[Hashable] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ):
        return self._value_counts(
            subset=subset,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )

    def var(
        self, axis=None, skipna=True, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._var(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )
