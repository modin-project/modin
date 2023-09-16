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

"""Module houses default Resamle functions builder class."""

from .default import DefaultMethod


# FIXME: there is no sence of keeping `Resampler` and `ResampleDefault` logic in a different
# classes. They should be combined.
class Resampler:
    """Builder class for resampled aggregation functions."""

    @classmethod
    def build_resample(cls, func, squeeze_self):
        """
        Build function that resamples time-series data and does aggregation.

        Parameters
        ----------
        func : callable
            Aggregation function to execute under resampled frame.
        squeeze_self : bool
            Whether or not to squeeze frame before resampling.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and applies aggregation
            to resampled time-series data.
        """

        def fn(df, resample_kwargs, *args, **kwargs):
            """Resample time-series data of the passed frame and apply specified aggregation."""
            if squeeze_self:
                df = df.squeeze(axis=1)
            resampler = df.resample(**resample_kwargs)

            if type(func) is property:
                return func.fget(resampler)

            return func(resampler, *args, **kwargs)

        return fn


class ResampleDefault(DefaultMethod):
    """Builder for default-to-pandas resampled aggregation functions."""

    OBJECT_TYPE = "Resampler"

    @classmethod
    def register(cls, func, squeeze_self=False, **kwargs):
        """
        Build function that do fallback to pandas and aggregate resampled data.

        Parameters
        ----------
        func : callable
            Aggregation function to execute under resampled frame.
        squeeze_self : bool, default: False
            Whether or not to squeeze frame before resampling.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Function that takes query compiler and does fallback to pandas to resample
            time-series data and apply aggregation on it.
        """
        return super().register(
            Resampler.build_resample(func, squeeze_self),
            fn_name=func.__name__,
            **kwargs
        )
