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

"""The module holds base class implementing required I/O over Ray."""

from modin.core.io import BaseIO


class RayIO(BaseIO):
    """Base class for doing I/O operations over Ray."""

    @classmethod
    def from_ray(cls, ray_obj):
        """
        Create a Modin `query_compiler` from a Ray Dataset.

        Parameters
        ----------
        ray_obj : ray.data.Dataset
            The Ray Dataset to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Ray Dataset.

        Notes
        -----
        This function must be implemented in every subclass
        otherwise NotImplementedError will be raised.
        """
        raise NotImplementedError(
            f"Modin dataset can't be created from `ray.data.Dataset` using {cls}."
        )

    @classmethod
    def to_ray(cls, modin_obj):
        """
        Convert a Modin DataFrame/Series to a Ray Dataset.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        ray.data.Dataset
            Converted object with type depending on input.

        Notes
        -----
        This function must be implemented in every subclass
        otherwise NotImplementedError will be raised.
        """
        raise NotImplementedError(
            f"`ray.data.Dataset` can't be created from Modin DataFrame/Series using {cls}."
        )
