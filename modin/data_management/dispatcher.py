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

import os

from modin import execution_engine, partition_format
from modin.data_management import factories


class FactoryNotFoundError(AttributeError):
    pass


class StubIoEngine(object):
    def __init__(self, factory_name=""):
        self.factory_name = factory_name or "Unknown"

    def __getattribute__(self, name):
        factory_name = self.factory_name  # for closure to bind the value

        def stub(*args, **kw):
            raise NotImplementedError(
                "Method {}.{} is not implemented".format(factory_name, name)
            )

        return stub


class StubFactory(factories.BaseFactory):
    """
    A factory that does nothing more than raise NotImplementedError when any method is called.
    Used for testing purposes.
    """

    io_cls = StubIoEngine()

    @classmethod
    def set_failing_name(cls, factory_name):
        cls.io_cls = StubIoEngine(factory_name)
        return cls


class EngineDispatcher(object):
    """
    This is the 'ingestion' point which knows where to route the work
    """

    __engine: factories.BaseFactory = None

    @classmethod
    def get_engine(cls) -> factories.BaseFactory:
        # mostly for testing
        return cls.__engine

    @classmethod
    def _update_engine(cls, _):
        if os.environ.get("MODIN_EXPERIMENTAL", "").title() == "True":
            factory_fmt, experimental = "Experimental{}On{}Factory", True
        else:
            factory_fmt, experimental = "{}On{}Factory", False
        factory_name = factory_fmt.format(
            partition_format.get(), execution_engine.get()
        )
        try:
            cls.__engine = getattr(factories, factory_name)
        except AttributeError:
            if not experimental:
                # allow missing factories in experimenal mode only
                if hasattr(factories, "Experimental" + factory_name):
                    msg = (
                        "{0} on {1} is only accessible through the experimental API.\nRun "
                        "`import modin.experimental.pandas as pd` to use {0} on {1}."
                    )
                else:
                    msg = (
                        "Cannot find a factory for partition '{}' and execution engine '{}'. "
                        "Potential reason might be incorrect environment variable value for "
                        "MODIN_BACKEND or MODIN_ENGINE"
                    )
                raise FactoryNotFoundError(
                    msg.format(partition_format.get(), execution_engine.get())
                )
            cls.__engine = StubFactory.set_failing_name(factory_name)
        else:
            cls.__engine.prepare()

    @classmethod
    def from_pandas(cls, df):
        return cls.__engine._from_pandas(df)

    @classmethod
    def from_non_pandas(cls, *args, **kwargs):
        return cls.__engine._from_non_pandas(*args, **kwargs)

    @classmethod
    def read_parquet(cls, **kwargs):
        return cls.__engine._read_parquet(**kwargs)

    @classmethod
    def read_csv(cls, **kwargs):
        return cls.__engine._read_csv(**kwargs)

    @classmethod
    def read_json(cls, **kwargs):
        return cls.__engine._read_json(**kwargs)

    @classmethod
    def read_gbq(cls, **kwargs):
        return cls.__engine._read_gbq(**kwargs)

    @classmethod
    def read_html(cls, **kwargs):
        return cls.__engine._read_html(**kwargs)

    @classmethod
    def read_clipboard(cls, **kwargs):  # pragma: no cover
        return cls.__engine._read_clipboard(**kwargs)

    @classmethod
    def read_excel(cls, **kwargs):
        return cls.__engine._read_excel(**kwargs)

    @classmethod
    def read_hdf(cls, **kwargs):
        return cls.__engine._read_hdf(**kwargs)

    @classmethod
    def read_feather(cls, **kwargs):
        return cls.__engine._read_feather(**kwargs)

    @classmethod
    def read_stata(cls, **kwargs):
        return cls.__engine._read_stata(**kwargs)

    @classmethod
    def read_sas(cls, **kwargs):  # pragma: no cover
        return cls.__engine._read_sas(**kwargs)

    @classmethod
    def read_pickle(cls, **kwargs):
        return cls.__engine._read_pickle(**kwargs)

    @classmethod
    def read_sql(cls, **kwargs):
        return cls.__engine._read_sql(**kwargs)

    @classmethod
    def read_fwf(cls, **kwargs):
        return cls.__engine._read_fwf(**kwargs)

    @classmethod
    def read_sql_table(cls, **kwargs):
        return cls.__engine._read_sql_table(**kwargs)

    @classmethod
    def read_sql_query(cls, **kwargs):
        return cls.__engine._read_sql_query(**kwargs)

    @classmethod
    def read_spss(cls, **kwargs):
        return cls.__engine._read_spss(**kwargs)

    @classmethod
    def to_sql(cls, *args, **kwargs):
        return cls.__engine._to_sql(*args, **kwargs)

    @classmethod
    def to_pickle(cls, *args, **kwargs):
        return cls.__engine._to_pickle(*args, **kwargs)


execution_engine.subscribe(EngineDispatcher._update_engine)
partition_format.subscribe(EngineDispatcher._update_engine)
