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
    def __init__(self, obj_type):
        self.obj_type = obj_type

    def __getattr__(self, key):
        def fn(df, resample_args, *args, **kwargs):
            if self.obj_type == "ser":
                df = df.squeeze(axis=1)
            prop = getattr(df.resample(*resample_args), key)
            if callable(prop):
                return prop(*args, **kwargs)
            else:
                return prop

        return fn


class ResampleDefault(DefaultMethod):
    methods_translator = {
        "app": "apply",
        "agg": "aggregate",
    }

    @classmethod
    def register(cls, func, **kwargs):
        splitted = func.split("_")[1:]
        if splitted[-1] == "ser":
            fn = "_".join(splitted[:-1])
            obj_type = "ser"
        elif splitted[-1] == "df":
            fn = "_".join(splitted[:-1])
            obj_type = "df"
        else:
            fn = "_".join(splitted)
            obj_type = "df"

        fn = cls.methods_translator.get(fn, fn)

        return cls.call(fn, obj_type=Resampler(obj_type=obj_type), **kwargs)
