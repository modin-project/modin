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

from .default import DefaultMethod


class Resampler:
    @classmethod
    def build_resample(cls, func, squeeze_self):
        def fn(df, resample_args, *args, **kwargs):
            if squeeze_self:
                df = df.squeeze(axis=1)
            resampler = df.resample(*resample_args)

            if type(func) == property:
                return func.fget(resampler)

            return func(resampler, *args, **kwargs)

        return fn


class ResampleDefault(DefaultMethod):
    OBJECT_TYPE = "Resampler"

    @classmethod
    def register(cls, func, squeeze_self=False, **kwargs):
        return cls.call(
            Resampler.build_resample(func, squeeze_self),
            fn_name=func.__name__,
            **kwargs
        )
