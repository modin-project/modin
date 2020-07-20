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
import warnings
from packaging import version
import collections

from ._version import get_versions


def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the message
    return "{}: {}\n".format(category.__name__, msg)


warnings.formatwarning = custom_formatwarning
# Filter numpy version warnings because they are not relevant
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="Large object of size")
warnings.filterwarnings(
    "ignore",
    message="The pandas.datetime class is deprecated and will be removed from pandas in a future version. "
    "Import from datetime module instead.",
)


def get_execution_engine():
    # In the future, when there are multiple engines and different ways of
    # backing the DataFrame, there will have to be some changed logic here to
    # decide these things. In the meantime, we will use the currently supported
    # execution engine + backing (Pandas + Ray).
    if "MODIN_ENGINE" in os.environ:
        # .title allows variants like ray, RAY, Ray
        return os.environ["MODIN_ENGINE"].title()
    else:
        if "MODIN_DEBUG" in os.environ:
            return "Python"
        else:
            try:
                import ray

            except ImportError:
                pass
            else:
                if version.parse(ray.__version__) != version.parse("0.8.6"):
                    raise ImportError(
                        "Please `pip install modin[ray]` to install compatible Ray version."
                    )
                return "Ray"
            try:
                import dask
                import distributed

            except ImportError:
                raise ImportError(
                    "Please `pip install modin[ray]` or `modin[dask]` to install an engine"
                )
            else:
                if version.parse(dask.__version__) < version.parse(
                    "2.1.0"
                ) or version.parse(distributed.__version__) < version.parse("2.3.2"):
                    raise ImportError(
                        "Please `pip install modin[dask]` to install compatible Dask version."
                    )
                return "Dask"


def get_partition_format():
    # See note above about engine + backing.
    return os.environ.get("MODIN_BACKEND", "Pandas").title()


class Publisher(object):
    def __init__(self, name, value):
        self.name = name
        self.__value = value.title()
        self.__subs = set()
        self.__once = collections.defaultdict(set)

    def subscribe(self, callback):
        self.__subs.add(callback)
        callback(self)

    def once(self, onvalue, callback):
        onvalue = onvalue.title()
        if onvalue == self.__value:
            callback(self)
        else:
            self.__once[onvalue].add(callback)

    def get(self):
        return self.__value

    def _put_nocallback(self, value):
        value = value.title()  # normalize the value
        oldvalue, self.__value = self.__value, value
        return oldvalue

    def _check_callbacks(self, oldvalue):
        if oldvalue == self.__value:
            return
        for callback in self.__subs:
            callback(self)
        once = self.__once.pop(self.__value, ())
        for callback in once:
            callback(self)

    def put(self, value):
        self._check_callbacks(self._put_nocallback(value))


execution_engine = Publisher(name="execution_engine", value=get_execution_engine())
partition_format = Publisher(name="partition_format", value=get_partition_format())


def set_backends(engine=None, partition=None):
    """
    Method to set the _pair_ of execution engine and partition format simultaneously.
    This is needed because there might be cases where switching one by one would be
    impossible, as not all pairs of values are meaningful.

    The method returns pair of old values, so it is easy to return back.
    """
    old_engine, old_partition = None, None
    # defer callbacks until both entities are set
    if engine is not None:
        old_engine = execution_engine._put_nocallback(engine)
    if partition is not None:
        old_partition = partition_format._put_nocallback(partition)
    # execute callbacks if something was changed
    if old_engine is not None:
        execution_engine._check_callbacks(old_engine)
    if old_partition is not None:
        partition_format._check_callbacks(old_partition)

    return old_engine, old_partition


# We don't want these used outside of this file.
del get_execution_engine
del get_partition_format

__version__ = get_versions()["version"]
del get_versions
