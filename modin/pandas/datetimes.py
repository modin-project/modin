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

from .dataframe import DataFrame
from .series import Series


def to_datetime(
    arg,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):
    """Convert the arg to datetime format. If not Ray DataFrame, this falls
       back on pandas.

    Args:
        errors ('raise' or 'ignore'): If 'ignore', errors are silenced.
            Pandas blatantly ignores this argument so we will too.
        dayfirst (bool): Date format is passed in as day first.
        yearfirst (bool): Date format is passed in as year first.
        utc (bool): retuns a UTC DatetimeIndex if True.
        box (bool): If True, returns a DatetimeIndex.
        format (string): strftime to parse time, eg "%d/%m/%Y".
        exact (bool): If True, require an exact format match.
        unit (string, default 'ns'): unit of the arg.
        infer_datetime_format (bool): Whether or not to infer the format.
        origin (string): Define the reference date.

    Returns:
        Type depends on input:

        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp
    """
    if not isinstance(arg, (DataFrame, Series)):
        return pandas.to_datetime(
            arg,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
    return arg._default_to_pandas(
        pandas.to_datetime,
        errors=errors,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        format=format,
        exact=exact,
        unit=unit,
        infer_datetime_format=infer_datetime_format,
        origin=origin,
        cache=cache,
    )
