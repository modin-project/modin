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

"""Module for 'Python 3.6 pandas' compatibility layer for Resample objects."""

import pandas

from modin._compat.pandas_api.abc.resample import BaseCompatibleResampler
from modin.utils import _inherit_docstrings, append_to_docstring


@append_to_docstring("Compatibility layer for 'Python 3.6 pandas' for Resampler.")
@_inherit_docstrings(pandas.core.resample.Resampler)
class Python36CompatibleResampler(BaseCompatibleResampler):
    def __init__(
        self,
        dataframe,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=0,
        on=None,
        level=None,
        origin="start_day",
        offset=None,
    ):
        self._init(
            dataframe=dataframe,
            rule=rule,
            axis=axis,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            loffset=loffset,
            base=base,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
        )

    def _get_groups(self):
        """
        Compute the resampled groups.

        Returns
        -------
        PandasGroupby
            Groups as specified by resampling arguments.
        """
        df = self._dataframe if self.axis == 0 else self._dataframe.T
        groups = df.groupby(
            pandas.Grouper(
                key=self.resample_kwargs["on"],
                freq=self.resample_kwargs["rule"],
                closed=self.resample_kwargs["closed"],
                label=self.resample_kwargs["label"],
                convention=self.resample_kwargs["convention"],
                loffset=self.resample_kwargs["loffset"],
                base=self.resample_kwargs["base"],
                level=self.resample_kwargs["level"],
                origin=self.resample_kwargs["origin"],
                offset=self.resample_kwargs["offset"],
            )
        )
        return groups

    @classmethod
    def _make(cls, **kwargs):  # noqa: PR01
        """Create Resampler potentially skipping unsupported parameters."""
        group_keys = kwargs.pop("group_keys", None)
        assert group_keys is None, f"Unexpected argument: {group_keys}"
        return cls(**kwargs)
