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
import sys
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
            if sys.platform != "win32":
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
                    "Please `pip install {}modin[dask]` to install an engine".format(
                        "modin[ray]` or `" if sys.platform != "win32" else ""
                    )
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

    def put(self, value):
        value = value.title()  # normalize the value
        oldvalue, self.__value = self.__value, value
        if oldvalue != value:
            for callback in self.__subs:
                callback(self)
            try:
                once = self.__once[value]
            except KeyError:
                return
            if once:
                for callback in once:
                    callback(self)
            del self.__once[value]


__version__ = "0.6.3"
execution_engine = Publisher(name="execution_engine", value=get_execution_engine())
partition_format = Publisher(name="partition_format", value=get_partition_format())

# We don't want these used outside of this file.
del get_execution_engine
del get_partition_format

__version__ = get_versions()["version"]
del get_versions
