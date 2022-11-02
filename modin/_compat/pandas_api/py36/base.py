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

"""Module for 'Python 3.6 pandas' compatibility layer for Dataset (common DataFrame/Series)."""

import pandas
from pandas.util._validators import validate_bool_kwarg
import pickle as pkl
from numpy import nan
from typing import Sequence, Hashable

from ..abc import BaseCompatibleBasePandasDataset
from .utils import create_stat_method
from modin.utils import _inherit_docstrings


@_inherit_docstrings(pandas.DataFrame)
class Python36CompatibleBasePandasDataset(BaseCompatibleBasePandasDataset):
    """Compatibility layer for 'Python 3.6 pandas' for Dataset."""

    def _validate_ascending(self, ascending):  # noqa: PR01, RT01
        """Skip validation because pandas 1.1.x allowed anything."""
        return ascending

    @_inherit_docstrings(validate_bool_kwarg)
    def _validate_bool_kwarg(self, value, arg_name, **kwargs):
        return validate_bool_kwarg(value, arg_name)

    def between_time(
        self, start_time, end_time, include_start=True, include_end=True, axis=None
    ):
        return self._between_time(
            start_time=start_time,
            end_time=end_time,
            include_start=include_start,
            include_end=include_end,
            axis=axis,
        )

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
    ):
        return self._convert_dtypes(
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
            convert_floating=None,
        )

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False):
        return self._dropna(
            axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace
        )

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        axis=0,
        times=None,
    ):
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
        )

    def expanding(self, min_periods=1, center=None, axis=0):
        return self._expanding(min_periods=min_periods, center=center, axis=axis)

    def idxmax(self, axis=0, skipna=True):
        return self._idxmax(axis=axis, skipna=skipna)

    def idxmin(self, axis=0, skipna=True):
        return self._idxmin(axis=axis, skipna=skipna)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._kurt(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def mad(self, axis=None, skipna=None, level=None):
        return self._mad(axis=axis, skipna=skipna, level=level)

    def mask(
        self,
        cond,
        other=nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
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

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._max(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._min(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    mean = create_stat_method("mean")
    median = create_stat_method("median")

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
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
        return self._reindex(
            index=index,
            columns=columns,
            copy=copy,
            **kwargs,
        )

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=None,
        on=None,
        level=None,
        origin="start_day",
        offset=None,
    ):
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
            group_keys=None,
        )

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        return self._reset_index(
            level=level,
            drop=drop,
            inplace=inplace,
            col_level=col_level,
            col_fill=col_fill,
            allow_duplicates=None,
            names=None,
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
    ):
        return self._rolling(
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
        )

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
    ):
        return self._sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis,
        )

    def set_axis(self, labels, axis=0, inplace=False):
        return self._set_axis(
            labels=labels, axis=axis, inplace=inplace, copy=not inplace
        )

    def sem(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._sem(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        return self._shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._skew(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def std(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
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
        line_terminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors: str = "strict",
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
            lineterminator=line_terminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
            errors=errors,
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
        encoding=None,
        inf_rep="inf",
        verbose=True,
        freeze_panes=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_excel",
            excel_writer,
            sheet_name,
            na_rep,
            float_format,
            columns,
            header,
            index,
            index_label,
            startrow,
            startcol,
            engine,
            merge_cells,
            encoding,
            inf_rep,
            verbose,
            freeze_panes,
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
        )

    def to_markdown(self, buf=None, mode=None, index: bool = True, **kwargs):
        return self._default_to_pandas(
            "to_markdown", buf=buf, mode=mode, index=index, **kwargs
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

    def to_pickle(
        self, path, compression="infer", protocol=pkl.HIGHEST_PROTOCOL
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_pickle", path, compression=compression, protocol=protocol
        )

    def value_counts(
        self,
        subset: Sequence[Hashable] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
    ):
        return self._value_counts(
            subset=subset,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=True,
        )

    def var(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._var(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )
