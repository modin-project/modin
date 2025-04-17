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

"""The module holds utility and initialization routines for Modin on unidist."""

import unidist
import unidist.config as unidist_cfg

import modin.config as modin_cfg

from .engine_wrapper import UnidistWrapper


def initialize_unidist():
    """
    Initialize unidist based on ``modin.config`` variables and internal defaults.
    """

    if unidist_cfg.Backend.get() != "mpi":
        raise RuntimeError(
            f"Modin only supports MPI through unidist for now, got unidist backend '{unidist_cfg.Backend.get()}'"
        )

    if not unidist.is_initialized():
        modin_cfg.CpuCount.subscribe(
            lambda cpu_count: unidist_cfg.CpuCount.put(cpu_count.get())
        )
        unidist_cfg.MpiRuntimeEnv.put(
            {"env_vars": {"PYTHONWARNINGS": "ignore::FutureWarning"}}
        )
        unidist.init()

    num_cpus = sum(v["CPU"] for v in unidist.cluster_resources().values())
    modin_cfg.NPartitions._put(num_cpus)
    modin_cfg.CpuCount._put(num_cpus)


def deserialize(obj):  # pragma: no cover
    """
    Deserialize a unidist object.

    Parameters
    ----------
    obj : unidist.ObjectRef, iterable of unidist.ObjectRef, or mapping of keys to unidist.ObjectRef
        Object(s) to deserialize.

    Returns
    -------
    obj
        The deserialized object(s).
    """
    if unidist.is_object_ref(obj):
        return UnidistWrapper.materialize(obj)
    elif isinstance(obj, (tuple, list)):
        # Unidist will error if any elements are not ObjectRef, but we still want unidist to
        # perform batch deserialization for us -- thus, we must submit only the list elements
        # that are ObjectRef, deserialize them, and restore them to their correct list index
        ref_indices, refs = [], []
        for i, unidist_ref in enumerate(obj):
            if unidist.is_object_ref(unidist_ref):
                ref_indices.append(i)
                refs.append(unidist_ref)
        unidist_result = UnidistWrapper.materialize(refs)
        new_lst = list(obj)
        for i, deser_item in zip(ref_indices, unidist_result):
            new_lst[i] = deser_item
        # Check that all objects have been deserialized
        assert not any(unidist.is_object_ref(o) for o in new_lst)
        return new_lst
    elif isinstance(obj, dict) and any(
        unidist.is_object_ref(val) for val in obj.values()
    ):
        return dict(zip(obj.keys(), deserialize(tuple(obj.values()))))
    else:
        return obj
