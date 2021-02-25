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

from modin.config import Engine, Backend, IsExperimental
from modin.data_management.factories import factories
from modin.utils import get_current_backend


class FactoryNotFoundError(AttributeError):
    pass


class StubIoEngine(object):
    def __init__(self, factory_name=""):
        self.factory_name = factory_name or "Unknown"

    def __getattr__(self, name):
        def stub(*args, **kw):
            raise NotImplementedError(
                f"Method {self.factory_name}.{name} is not implemented"
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
        factory_name = get_current_backend() + "Factory"
        try:
            cls.__engine = getattr(factories, factory_name)
        except AttributeError:
            if not IsExperimental.get():
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
                        f"{Backend.varname} or {Engine.varname}"
                    )
                raise FactoryNotFoundError(msg.format(Backend.get(), Engine.get()))
            cls.__engine = StubFactory.set_failing_name(factory_name)
        else:
            cls.__engine.prepare()

    @classmethod
    def from_pandas(cls, df):
        return cls.__engine._from_pandas(df)

    @classmethod
    def from_arrow(cls, at):
        return cls.__engine._from_arrow(at)

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
    def read_csv_glob(cls, **kwargs):
        return cls.__engine._read_csv_glob(**kwargs)

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

    @classmethod
    def to_csv(cls, *args, **kwargs):
        return cls.__engine._to_csv(*args, **kwargs)


Engine.subscribe(EngineDispatcher._update_engine)
Backend.subscribe(EngineDispatcher._update_engine)
