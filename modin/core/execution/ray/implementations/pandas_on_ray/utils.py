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

"""The module holds utility helpers for Modin on Ray internals."""

import ray
from packaging import version

ObjectIDType = ray.ObjectRef
if version.parse(ray.__version__) >= version.parse("1.2.0"):
    from ray.util.client.common import ClientObjectRef

    ObjectIDType = (ray.ObjectRef, ClientObjectRef)


def deserialize(obj):
    if isinstance(obj, ObjectIDType):
        return ray.get(obj)
    elif isinstance(obj, (tuple, list)) and any(
        isinstance(o, ObjectIDType) for o in obj
    ):
        return ray.get(list(obj))
    elif isinstance(obj, dict) and any(
        isinstance(val, ObjectIDType) for val in obj.values()
    ):
        return dict(zip(obj.keys(), ray.get(list(obj.values()))))
    else:
        return obj
