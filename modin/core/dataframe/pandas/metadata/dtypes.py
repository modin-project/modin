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

"""Module contains class ``ModinDtypes``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import pandas
from pandas._typing import DtypeObj, IndexLabel
from pandas.core.dtypes.cast import find_common_type

if TYPE_CHECKING:
    from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
    from .index import ModinIndex

from modin.error_message import ErrorMessage


class DtypesDescriptor:
    """
    Describes partial dtypes.

    Parameters
    ----------
    known_dtypes : dict[IndexLabel, DtypeObj] or pandas.Series, optional
        Columns that we know dtypes for.
    cols_with_unknown_dtypes : list[IndexLabel], optional
        Column names that have unknown dtypes. If specified together with `remaining_dtype`, must describe all
        columns with unknown dtypes, otherwise, the missing columns will be assigned to `remaining_dtype`.
        If `cols_with_unknown_dtypes` is incomplete, you must specify `know_all_names=False`.
    remaining_dtype : DtypeObj, optional
        Dtype for columns that are not present neither in `known_dtypes` nor in `cols_with_unknown_dtypes`.
        This parameter is intended to describe columns that we known dtypes for, but don't know their
        names yet. Note, that this parameter DOESN'T describe dtypes for columns from `cols_with_unknown_dtypes`.
    parent_df : PandasDataframe, optional
        Dataframe object for which we describe dtypes. This dataframe will be used to compute
        missing dtypes on ``.materialize()``.
    columns_order : dict[int, IndexLabel], optional
        Order of columns in the dataframe. If specified, must describe all the columns of the dataframe.
    know_all_names : bool, default: True
        Whether `known_dtypes` and `cols_with_unknown_dtypes` contain all column names for this dataframe besides those,
        that are being described by `remaining_dtype`.
        One can't pass `know_all_names=False` together with `remaining_dtype` as this creates ambiguity
        on how to interpret missing columns (whether they belong to `remaining_dtype` or not).
    _schema_is_known : bool, optional
        Whether `known_dtypes` describe all columns in the dataframe. This parameter intended mostly
        for internal use.
    """

    def __init__(
        self,
        known_dtypes: Optional[Union[dict[IndexLabel, DtypeObj], pandas.Series]] = None,
        cols_with_unknown_dtypes: Optional[list[IndexLabel]] = None,
        remaining_dtype: Optional[DtypeObj] = None,
        parent_df: Optional[PandasDataframe] = None,
        columns_order: Optional[dict[int, IndexLabel]] = None,
        know_all_names: bool = True,
        _schema_is_known: Optional[bool] = None,
    ):
        if not know_all_names and remaining_dtype is not None:
            raise ValueError(
                "It's not allowed to pass 'remaining_dtype' and 'know_all_names=False' at the same time."
            )
        # columns with known dtypes
        self._known_dtypes: dict[IndexLabel, DtypeObj] = (
            {} if known_dtypes is None else dict(known_dtypes)
        )
        if known_dtypes is not None and len(self._known_dtypes) != len(known_dtypes):
            raise NotImplementedError(
                "Duplicated column names are not yet supported by DtypesDescriptor"
            )
        # columns with unknown dtypes (they're not described by 'remaining_dtype')
        if cols_with_unknown_dtypes is not None and len(
            set(cols_with_unknown_dtypes)
        ) != len(cols_with_unknown_dtypes):
            raise NotImplementedError(
                "Duplicated column names are not yet supported by DtypesDescriptor"
            )
        self._cols_with_unknown_dtypes: list[IndexLabel] = (
            [] if cols_with_unknown_dtypes is None else cols_with_unknown_dtypes
        )
        # whether 'known_dtypes' describe all columns in the dataframe
        self._schema_is_known: Optional[bool] = _schema_is_known
        if self._schema_is_known is None:
            self._schema_is_known = False
            if (
                # if 'cols_with_unknown_dtypes' was explicitly specified as an empty list and
                # we don't have any 'remaining_dtype', then we assume that 'known_dtypes' are complete
                cols_with_unknown_dtypes is not None
                and know_all_names
                and remaining_dtype is None
                and len(self._known_dtypes) > 0
            ):
                self._schema_is_known = len(cols_with_unknown_dtypes) == 0

        self._know_all_names: bool = know_all_names
        # a common dtype for columns that are not present in 'known_dtypes' nor in 'cols_with_unknown_dtypes'
        self._remaining_dtype: Optional[DtypeObj] = remaining_dtype
        self._parent_df: Optional[PandasDataframe] = parent_df
        if columns_order is None:
            self._columns_order: Optional[dict[int, IndexLabel]] = None
            # try to compute '._columns_order' using 'parent_df'
            self.columns_order
        else:
            if remaining_dtype is not None:
                raise ValueError(
                    "Passing 'columns_order' and 'remaining_dtype' is ambiguous. You have to manually "
                    + "complete 'known_dtypes' using the information from 'columns_order' and 'remaining_dtype'."
                )
            elif not self._know_all_names:
                raise ValueError(
                    "Passing 'columns_order' and 'know_all_names=False' is ambiguous. You have to manually "
                    + "complete 'cols_with_unknown_dtypes' using the information from 'columns_order' "
                    + "and pass 'know_all_names=True'."
                )
            elif len(columns_order) != (
                len(self._cols_with_unknown_dtypes) + len(self._known_dtypes)
            ):
                raise ValueError(
                    "The length of 'columns_order' doesn't match to 'known_dtypes' and 'cols_with_unknown_dtypes'"
                )
            self._columns_order: Optional[dict[int, IndexLabel]] = columns_order

    def update_parent(self, new_parent: PandasDataframe):
        """
        Set new parent dataframe.

        Parameters
        ----------
        new_parent : PandasDataframe
        """
        self._parent_df = new_parent
        LazyProxyCategoricalDtype.update_dtypes(self._known_dtypes, new_parent)
        # try to compute '._columns_order' using 'new_parent'
        self.columns_order

    @property
    def columns_order(self) -> Optional[dict[int, IndexLabel]]:
        """
        Get order of columns for the described dataframe if available.

        Returns
        -------
        dict[int, IndexLabel] or None
        """
        if self._columns_order is not None:
            return self._columns_order
        if self._parent_df is None or not self._parent_df.has_materialized_columns:
            return None

        actual_columns = self._parent_df.columns
        self._normalize_self_levels(actual_columns)

        self._columns_order = {i: col for i, col in enumerate(actual_columns)}
        # we got information about new columns and thus can potentially
        # extend our knowledge about missing dtypes
        if len(self._columns_order) > (
            len(self._known_dtypes) + len(self._cols_with_unknown_dtypes)
        ):
            new_cols = [
                col
                for col in self._columns_order.values()
                if col not in self._known_dtypes
                and col not in self._cols_with_unknown_dtypes
            ]
            if self._remaining_dtype is not None:
                self._known_dtypes.update(
                    {col: self._remaining_dtype for col in new_cols}
                )
                self._remaining_dtype = None
                if len(self._cols_with_unknown_dtypes) == 0:
                    self._schema_is_known = True
            else:
                self._cols_with_unknown_dtypes.extend(new_cols)
        self._know_all_names = True
        return self._columns_order

    def __repr__(self):  # noqa: GL08
        return (
            f"DtypesDescriptor:\n\tknown dtypes: {self._known_dtypes};\n\t"
            + f"remaining dtype: {self._remaining_dtype};\n\t"
            + f"cols with unknown dtypes: {self._cols_with_unknown_dtypes};\n\t"
            + f"schema is known: {self._schema_is_known};\n\t"
            + f"has parent df: {self._parent_df is not None};\n\t"
            + f"columns order: {self._columns_order};\n\t"
            + f"know all names: {self._know_all_names}"
        )

    def __str__(self):  # noqa: GL08
        return self.__repr__()

    def lazy_get(
        self, ids: list[Union[IndexLabel, int]], numeric_index: bool = False
    ) -> DtypesDescriptor:
        """
        Get dtypes descriptor for a subset of columns without triggering any computations.

        Parameters
        ----------
        ids : list of index labels or positional indexers
            Columns for the subset.
        numeric_index : bool, default: False
            Whether `ids` are positional indixes or column labels to take.

        Returns
        -------
        DtypesDescriptor
            Descriptor that describes dtypes for columns specified in `ids`.
        """
        if len(set(ids)) != len(ids):
            raise NotImplementedError(
                "Duplicated column names are not yet supported by DtypesDescriptor"
            )

        if numeric_index:
            if self.columns_order is not None:
                ids = [self.columns_order[i] for i in ids]
            else:
                raise ValueError(
                    "Can't lazily get columns by positional indixers if the columns order is unknown"
                )

        result = {}
        unknown_cols = []
        columns_order = {}
        for i, col in enumerate(ids):
            columns_order[i] = col
            if col in self._cols_with_unknown_dtypes:
                unknown_cols.append(col)
                continue
            dtype = self._known_dtypes.get(col)
            if dtype is None and self._remaining_dtype is None:
                unknown_cols.append(col)
            elif dtype is None and self._remaining_dtype is not None:
                result[col] = self._remaining_dtype
            else:
                result[col] = dtype
        remaining_dtype = self._remaining_dtype if len(unknown_cols) != 0 else None
        return DtypesDescriptor(
            result,
            unknown_cols,
            remaining_dtype,
            self._parent_df,
            columns_order=columns_order,
        )

    def copy(self) -> DtypesDescriptor:
        """
        Get a copy of this descriptor.

        Returns
        -------
        DtypesDescriptor
        """
        return type(self)(
            # should access '.columns_order' first, as it may compute columns order
            # and complete the metadata for 'self'
            columns_order=(
                None if self.columns_order is None else self.columns_order.copy()
            ),
            known_dtypes=self._known_dtypes.copy(),
            cols_with_unknown_dtypes=self._cols_with_unknown_dtypes.copy(),
            remaining_dtype=self._remaining_dtype,
            parent_df=self._parent_df,
            know_all_names=self._know_all_names,
            _schema_is_known=self._schema_is_known,
        )

    def set_index(self, new_index: Union[pandas.Index, ModinIndex]) -> DtypesDescriptor:
        """
        Set new column names for this descriptor.

        Parameters
        ----------
        new_index : pandas.Index or ModinIndex

        Returns
        -------
        DtypesDescriptor
            New descriptor with updated column names.

        Notes
        -----
        Calling this method on a descriptor that returns ``None`` for ``.columns_order``
        will result into information lose.
        """
        if len(new_index) != len(set(new_index)):
            raise NotImplementedError(
                "Duplicated column names are not yet supported by DtypesDescriptor"
            )

        if self.columns_order is None:
            # we can't map new columns to old columns and lost all dtypes :(
            return DtypesDescriptor(
                cols_with_unknown_dtypes=new_index,
                columns_order={i: col for i, col in enumerate(new_index)},
                parent_df=self._parent_df,
                know_all_names=True,
            )

        new_self = self.copy()
        renamer = {old_c: new_index[i] for i, old_c in new_self.columns_order.items()}
        new_self._known_dtypes = {
            renamer[old_col]: value for old_col, value in new_self._known_dtypes.items()
        }
        new_self._cols_with_unknown_dtypes = [
            renamer[old_col] for old_col in new_self._cols_with_unknown_dtypes
        ]
        new_self._columns_order = {
            i: renamer[old_col] for i, old_col in new_self._columns_order.items()
        }
        return new_self

    def equals(self, other: DtypesDescriptor) -> bool:
        """
        Compare two descriptors for equality.

        Parameters
        ----------
        other : DtypesDescriptor

        Returns
        -------
        bool
        """
        return (
            self._known_dtypes == other._known_dtypes
            and set(self._cols_with_unknown_dtypes)
            == set(other._cols_with_unknown_dtypes)
            and self._remaining_dtype == other._remaining_dtype
            and self._schema_is_known == other._schema_is_known
            and self.columns_order == other.columns_order
            and self._know_all_names == other._know_all_names
        )

    @property
    def is_materialized(self) -> bool:
        """
        Whether this descriptor contains information about all dtypes in the dataframe.

        Returns
        -------
        bool
        """
        return self._schema_is_known

    def _materialize_all_names(self):
        """Materialize missing column names."""
        if self._know_all_names:
            return

        all_cols = self._parent_df.columns
        self._normalize_self_levels(all_cols)
        for col in all_cols:
            if (
                col not in self._known_dtypes
                and col not in self._cols_with_unknown_dtypes
            ):
                self._cols_with_unknown_dtypes.append(col)

        self._know_all_names = True

    def _materialize_cols_with_unknown_dtypes(self):
        """Compute dtypes for cols specified in `._cols_with_unknown_dtypes`."""
        if (
            len(self._known_dtypes) == 0
            and len(self._cols_with_unknown_dtypes) == 0
            and not self._know_all_names
        ):
            # here we have to compute dtypes for all columns in the dataframe,
            # so avoiding columns materialization by setting 'subset=None'
            subset = None
        else:
            if not self._know_all_names:
                self._materialize_all_names()
            subset = self._cols_with_unknown_dtypes

        if subset is None or len(subset) > 0:
            self._known_dtypes.update(self._parent_df._compute_dtypes(subset))

        self._know_all_names = True
        self._cols_with_unknown_dtypes = []

    def materialize(self):
        """Complete information about dtypes."""
        if self.is_materialized:
            return
        if self._parent_df is None:
            raise RuntimeError(
                "It's not allowed to call '.materialize()' before '._parent_df' is specified."
            )

        self._materialize_cols_with_unknown_dtypes()

        if self._remaining_dtype is not None:
            cols = self._parent_df.columns
            self._normalize_self_levels(cols)
            self._known_dtypes.update(
                {
                    col: self._remaining_dtype
                    for col in cols
                    if col not in self._known_dtypes
                }
            )

        # we currently not guarantee for dtypes to be in a proper order:
        # https://github.com/modin-project/modin/blob/8a332c1597c54d36f7ccbbd544e186b689f9ceb1/modin/pandas/test/utils.py#L644-L646
        # so restoring the order only if it's possible
        if self.columns_order is not None:
            assert len(self.columns_order) == len(self._known_dtypes)
            self._known_dtypes = {
                self.columns_order[i]: self._known_dtypes[self.columns_order[i]]
                for i in range(len(self.columns_order))
            }

        self._schema_is_known = True
        self._remaining_dtype = None
        self._parent_df = None

    def to_series(self) -> pandas.Series:
        """
        Convert descriptor to a pandas Series.

        Returns
        -------
        pandas.Series
        """
        self.materialize()
        return pandas.Series(self._known_dtypes)

    def get_dtypes_set(self) -> set[DtypeObj]:
        """
        Get a set of dtypes from the descriptor.

        Returns
        -------
        set[DtypeObj]
        """
        if len(self._cols_with_unknown_dtypes) > 0 or not self._know_all_names:
            self._materialize_cols_with_unknown_dtypes()
        known_dtypes: set[DtypeObj] = set(self._known_dtypes.values())
        if self._remaining_dtype is not None:
            known_dtypes.add(self._remaining_dtype)
        return known_dtypes

    @classmethod
    def _merge_dtypes(
        cls, values: list[Union[DtypesDescriptor, pandas.Series, None]]
    ) -> DtypesDescriptor:
        """
        Union columns described by ``values`` and compute common dtypes for them.

        Parameters
        ----------
        values : list of DtypesDescriptors, pandas.Series or Nones

        Returns
        -------
        DtypesDescriptor
        """
        known_dtypes = {}
        cols_with_unknown_dtypes = []
        know_all_names = True
        dtypes_are_unknown = False

        # index - joined column names, columns - dtypes taken from 'values'
        #        0     1     2      3
        #  col1  int   bool  float  int
        #  col2  int   int   int    int
        #  colN  bool  bool  bool   int
        dtypes_matrix = pandas.DataFrame()

        for i, val in enumerate(values):
            if isinstance(val, cls):
                know_all_names &= val._know_all_names
                dtypes = val._known_dtypes.copy()
                dtypes.update({col: "unknown" for col in val._cols_with_unknown_dtypes})
                if val._remaining_dtype is not None:
                    # we can't process remaining dtypes, so just discarding them
                    know_all_names = False

                # setting a custom name to the Series to prevent duplicated names
                # in the 'dtypes_matrix'
                series = pandas.Series(dtypes, name=i)
                dtypes_matrix = pandas.concat([dtypes_matrix, series], axis=1)
                if not (val._know_all_names and val._remaining_dtype is None):
                    dtypes_matrix.fillna(
                        value={
                            # If we encountered a 'NaN' while 'val' describes all the columns, then
                            # it means, that the missing columns for this instance will be filled with NaNs (floats),
                            # otherwise, it may indicate missing columns that this 'val' has no info about,
                            # meaning that we shouldn't try computing a new dtype for this column,
                            # so marking it as 'unknown'
                            i: "unknown",
                        },
                        inplace=True,
                    )
            elif isinstance(val, pandas.Series):
                dtypes_matrix = pandas.concat([dtypes_matrix, val], axis=1)
            elif val is None:
                # one of the 'dtypes' is None, meaning that we wouldn't been infer a valid result dtype,
                # however, we're continuing our loop so we would at least know the columns we're missing
                # dtypes for
                dtypes_are_unknown = True
                know_all_names = False
            else:
                raise NotImplementedError(type(val))

        if dtypes_are_unknown:
            return DtypesDescriptor(
                cols_with_unknown_dtypes=dtypes_matrix.index.tolist(),
                know_all_names=know_all_names,
            )

        def combine_dtypes(row):
            if (row == "unknown").any():
                return "unknown"
            row = row.fillna(pandas.api.types.pandas_dtype("float"))
            return find_common_type(list(row.values))

        dtypes = dtypes_matrix.apply(combine_dtypes, axis=1)

        for col, dtype in dtypes.items():
            if dtype == "unknown":
                cols_with_unknown_dtypes.append(col)
            else:
                known_dtypes[col] = dtype

        return DtypesDescriptor(
            known_dtypes,
            cols_with_unknown_dtypes,
            remaining_dtype=None,
            know_all_names=know_all_names,
        )

    @classmethod
    def concat(
        cls, values: list[Union[DtypesDescriptor, pandas.Series, None]], axis: int = 0
    ) -> DtypesDescriptor:
        """
        Concatenate dtypes descriptors into a single descriptor.

        Parameters
        ----------
        values : list of DtypesDescriptors and pandas.Series
        axis : int, default: 0
            If ``axis == 0``: concatenate column names. This implements the logic of
            how dtypes are combined on ``pd.concat([df1, df2], axis=1)``.
            If ``axis == 1``: perform a union join for the column names described by
            `values` and then find common dtypes for the columns appeared to be in
            an intersection. This implements the logic of how dtypes are combined on
            ``pd.concat([df1, df2], axis=0).dtypes``.

        Returns
        -------
        DtypesDescriptor
        """
        if axis == 1:
            return cls._merge_dtypes(values)
        known_dtypes = {}
        cols_with_unknown_dtypes = []
        schema_is_known = True
        # some default value to not mix it with 'None'
        remaining_dtype = "default"
        know_all_names = True

        for val in values:
            if isinstance(val, cls):
                all_new_cols = (
                    list(val._known_dtypes.keys()) + val._cols_with_unknown_dtypes
                )
                if any(
                    col in known_dtypes or col in cols_with_unknown_dtypes
                    for col in all_new_cols
                ):
                    raise NotImplementedError(
                        "Duplicated column names are not yet supported by DtypesDescriptor"
                    )
                know_all_names &= val._know_all_names
                known_dtypes.update(val._known_dtypes)
                cols_with_unknown_dtypes.extend(val._cols_with_unknown_dtypes)
                if know_all_names:
                    if (
                        remaining_dtype == "default"
                        and val._remaining_dtype is not None
                    ):
                        remaining_dtype = val._remaining_dtype
                    elif (
                        remaining_dtype != "default"
                        and val._remaining_dtype is not None
                        and remaining_dtype != val._remaining_dtype
                    ):
                        remaining_dtype = None
                        know_all_names = False
                else:
                    remaining_dtype = None
                schema_is_known &= val._schema_is_known
            elif isinstance(val, pandas.Series):
                if any(
                    col in known_dtypes or col in cols_with_unknown_dtypes
                    for col in val.index
                ):
                    raise NotImplementedError(
                        "Duplicated column names are not yet supported by DtypesDescriptor"
                    )
                known_dtypes.update(val)
            elif val is None:
                remaining_dtype = None
                schema_is_known = False
                know_all_names = False
            else:
                raise NotImplementedError(type(val))
        return cls(
            known_dtypes,
            cols_with_unknown_dtypes,
            None if remaining_dtype == "default" else remaining_dtype,
            parent_df=None,
            _schema_is_known=schema_is_known,
            know_all_names=know_all_names,
        )

    @staticmethod
    def _normalize_levels(columns, reference=None):
        """
        Normalize levels of MultiIndex column names.

        The function fills missing levels with empty strings as pandas do:
        '''
        >>> columns = ["a", ("l1", "l2"), ("l1a", "l2a", "l3a")]
        >>> _normalize_levels(columns)
        [("a", "", ""), ("l1", "l2", ""), ("l1a", "l2a", "l3a")]
        >>> # with a reference
        >>> idx = pandas.MultiIndex(...)
        >>> idx.nlevels
        4
        >>> _normalize_levels(columns, reference=idx)
        [("a", "", "", ""), ("l1", "l2", "", ""), ("l1a", "l2a", "l3a", "")]
        '''

        Parameters
        ----------
        columns : sequence
            Labels to normalize. If dictionary, will replace keys with normalized columns.
        reference : pandas.Index, optional
            An index to match the number of levels with. If reference is a MultiIndex, then the reference number
            of levels should not be greater than the maximum number of levels in `columns`. If not specified,
            the `columns` themselves become a `reference`.

        Returns
        -------
        sequence
            Column values with normalized levels.
        dict[hashable, hashable]
            Mapping from old column names to new names, only contains column names that
            were changed.

        Raises
        ------
        ValueError
            When the reference number of levels is greater than the maximum number of levels
            in `columns`.
        """
        if reference is None:
            reference = columns

        if isinstance(reference, pandas.Index):
            max_nlevels = reference.nlevels
        else:
            max_nlevels = 1
            for col in reference:
                if isinstance(col, tuple):
                    max_nlevels = max(max_nlevels, len(col))

        # if the reference is a regular flat index, then no actions are required (the result will be
        # a flat index containing tuples of different lengths, this behavior fully matches pandas).
        # Yes, this shortcut skips the 'if max_columns_nlevels > max_nlevels' below check on purpose.
        if max_nlevels == 1:
            return columns, {}

        max_columns_nlevels = 1
        for col in columns:
            if isinstance(col, tuple):
                max_columns_nlevels = max(max_columns_nlevels, len(col))

        if max_columns_nlevels > max_nlevels:
            raise ValueError(
                f"The reference number of levels is greater than the maximum number of levels in columns: {max_columns_nlevels} > {max_nlevels}"
            )

        new_columns = []
        old_to_new_mapping = {}
        for col in columns:
            old_col = col
            if not isinstance(col, tuple):
                col = (col,)
            col = col + ("",) * (max_nlevels - len(col))
            new_columns.append(col)
            if old_col != col:
                old_to_new_mapping[old_col] = col

        return new_columns, old_to_new_mapping

    def _normalize_self_levels(self, reference=None):
        """
        Call ``self._normalize_levels()`` for known and unknown dtypes of this object.

        Parameters
        ----------
        reference : pandas.Index, optional
        """
        _, old_to_new_mapping = self._normalize_levels(
            self._known_dtypes.keys(), reference
        )
        for old_col, new_col in old_to_new_mapping.items():
            value = self._known_dtypes.pop(old_col)
            self._known_dtypes[new_col] = value
        self._cols_with_unknown_dtypes, _ = self._normalize_levels(
            self._cols_with_unknown_dtypes, reference
        )


class ModinDtypes:
    """
    A class that hides the various implementations of the dtypes needed for optimization.

    Parameters
    ----------
    value : pandas.Series, callable, DtypesDescriptor or ModinDtypes, optional
    """

    def __init__(
        self,
        value: Optional[Union[Callable, pandas.Series, DtypesDescriptor, ModinDtypes]],
    ):
        if callable(value) or isinstance(value, pandas.Series):
            self._value = value
        elif isinstance(value, DtypesDescriptor):
            self._value = value.to_series() if value.is_materialized else value
        elif isinstance(value, type(self)):
            self._value = value.copy()._value
        elif isinstance(value, None):
            self._value = DtypesDescriptor()
        else:
            raise ValueError(f"ModinDtypes doesn't work with '{value}'")

    def __repr__(self):  # noqa: GL08
        return f"ModinDtypes:\n\tvalue type: {type(self._value)};\n\tvalue:\n\t{self._value}"

    def __str__(self):  # noqa: GL08
        return self.__repr__()

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Series)

    def get_dtypes_set(self) -> set[DtypeObj]:
        """
        Get a set of dtypes from the descriptor.

        Returns
        -------
        set[DtypeObj]
        """
        if isinstance(self._value, DtypesDescriptor):
            return self._value.get_dtypes_set()
        if not self.is_materialized:
            self.get()
        return set(self._value.values)

    def maybe_specify_new_frame_ref(self, new_parent: PandasDataframe) -> ModinDtypes:
        """
        Set a new parent for the stored value if needed.

        Parameters
        ----------
        new_parent : PandasDataframe

        Returns
        -------
        ModinDtypes
            A copy of ``ModinDtypes`` with updated parent.
        """
        new_self = self.copy()
        if new_self.is_materialized:
            LazyProxyCategoricalDtype.update_dtypes(new_self._value, new_parent)
            return new_self
        if isinstance(self._value, DtypesDescriptor):
            new_self._value.update_parent(new_parent)
            return new_self
        return new_self

    def lazy_get(self, ids: list, numeric_index: bool = False) -> ModinDtypes:
        """
        Get new ``ModinDtypes`` for a subset of columns without triggering any computations.

        Parameters
        ----------
        ids : list of index labels or positional indexers
            Columns for the subset.
        numeric_index : bool, default: False
            Whether `ids` are positional indixes or column labels to take.

        Returns
        -------
        ModinDtypes
            ``ModinDtypes`` that describes dtypes for columns specified in `ids`.
        """
        if isinstance(self._value, DtypesDescriptor):
            res = self._value.lazy_get(ids, numeric_index)
            return ModinDtypes(res)
        elif callable(self._value):
            new_self = self.copy()
            old_value = new_self._value
            new_self._value = lambda: (
                old_value().iloc[ids] if numeric_index else old_value()[ids]
            )
            return new_self
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=not self.is_materialized
        )
        return ModinDtypes(self._value.iloc[ids] if numeric_index else self._value[ids])

    @classmethod
    def concat(cls, values: list, axis: int = 0) -> ModinDtypes:
        """
        Concatenate dtypes.

        Parameters
        ----------
        values : list of DtypesDescriptors, pandas.Series, ModinDtypes and Nones
        axis : int, default: 0
            If ``axis == 0``: concatenate column names. This implements the logic of
            how dtypes are combined on ``pd.concat([df1, df2], axis=1)``.
            If ``axis == 1``: perform a union join for the column names described by
            `values` and then find common dtypes for the columns appeared to be in
            an intersection. This implements the logic of how dtypes are combined on
            ``pd.concat([df1, df2], axis=0).dtypes``.

        Returns
        -------
        ModinDtypes
        """
        preprocessed_vals = []
        for val in values:
            if isinstance(val, cls):
                val = val._value
            if isinstance(val, (DtypesDescriptor, pandas.Series)) or val is None:
                preprocessed_vals.append(val)
            else:
                raise NotImplementedError(type(val))

        try:
            desc = DtypesDescriptor.concat(preprocessed_vals, axis=axis)
        except NotImplementedError as e:
            # 'DtypesDescriptor' doesn't support duplicated labels, however, if all values are pandas Series,
            # we still can perform concatenation using pure pandas
            if (
                # 'pd.concat(axis=1)' fails on duplicated labels anyway, so doing this logic
                # only in case 'axis=0'
                axis == 0
                and "duplicated" not in e.args[0].lower()
                or not all(isinstance(val, pandas.Series) for val in values)
            ):
                raise e
            desc = pandas.concat(values)
        return ModinDtypes(desc)

    def set_index(self, new_index: Union[pandas.Index, ModinIndex]) -> ModinDtypes:
        """
        Set new column names for stored dtypes.

        Parameters
        ----------
        new_index : pandas.Index or ModinIndex

        Returns
        -------
        ModinDtypes
            New ``ModinDtypes`` with updated column names.
        """
        new_self = self.copy()
        if self.is_materialized:
            new_self._value.index = new_index
            return new_self
        elif callable(self._value):
            old_val = new_self._value
            new_self._value = lambda: old_val().set_axis(new_index)
            return new_self
        elif isinstance(new_self._value, DtypesDescriptor):
            new_self._value = new_self._value.set_index(new_index)
            return new_self
        else:
            raise NotImplementedError()

    def get(self) -> pandas.Series:
        """
        Get the materialized internal representation.

        Returns
        -------
        pandas.Series
        """
        if not self.is_materialized:
            if callable(self._value):
                self._value = self._value()
                if self._value is None:
                    self._value = pandas.Series([])
            elif isinstance(self._value, DtypesDescriptor):
                self._value = self._value.to_series()
            else:
                raise NotImplementedError(type(self._value))
        return self._value

    def __len__(self):
        """
        Redirect the 'len' request to the internal representation.

        Returns
        -------
        int

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return len(self._value)

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        The default implementation generates a recursion error. In a short:
        during the construction of the object, `__getattr__` function is called, which
        is not intended to be used in situations where the object is not initialized.
        """
        return (self.__class__, (self._value,))

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.

        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Series``. The main limitations of the current approach are type
        checking and the use of this object where pandas dtypes are supposed to be used.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute.

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return self._value.__getattribute__(name)

    def copy(self) -> ModinDtypes:
        """
        Copy an object without materializing the internal representation.

        Returns
        -------
        ModinDtypes
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        return ModinDtypes(idx_cache)

    def __getitem__(self, key):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return self._value.__getitem__(key)

    def __setitem__(self, key, item):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        self._value.__setitem__(key, item)

    def __iter__(self):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return iter(self._value)

    def __contains__(self, key):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return key in self._value


class LazyProxyCategoricalDtype(pandas.CategoricalDtype):
    """
    A lazy proxy representing ``pandas.CategoricalDtype``.

    Parameters
    ----------
    categories : list-like, optional
    ordered : bool, default: False

    Notes
    -----
    Important note! One shouldn't use the class' constructor to instantiate a proxy instance,
    it's intended only for compatibility purposes! In order to create a new proxy instance
    use the appropriate class method `._build_proxy(...)`.
    """

    def __init__(self, categories=None, ordered=False):
        # These will be initialized later inside of the `._build_proxy()` method
        self._parent, self._column_name, self._categories_val, self._materializer = (
            None,
            None,
            None,
            None,
        )
        super().__init__(categories, ordered)

    @staticmethod
    def update_dtypes(dtypes, new_parent):
        """
        Update a parent for categorical proxies in a dtype object.

        Parameters
        ----------
        dtypes : dict-like
            A dict-like object describing dtypes. The method will walk through every dtype
            an update parents for categorical proxies inplace.
        new_parent : object
        """
        for key, value in dtypes.items():
            if isinstance(value, LazyProxyCategoricalDtype):
                dtypes[key] = value._update_proxy(new_parent, column_name=key)

    def _update_proxy(self, parent, column_name):
        """
        Create a new proxy, if either parent or column name are different.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._is_materialized:
            # The parent has been materialized, we don't need a proxy anymore.
            return pandas.CategoricalDtype(self.categories, ordered=self._ordered)
        elif parent is self._parent and column_name == self._column_name:
            return self
        else:
            return self._build_proxy(parent, column_name, self._materializer)

    @classmethod
    def _build_proxy(cls, parent, column_name, materializer, dtype=None):
        """
        Construct a lazy proxy.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.
        materializer : callable(parent, column_name) -> pandas.CategoricalDtype
            A function to call in order to extract categorical values.
        dtype : dtype, optional
            The categories dtype.

        Returns
        -------
        LazyProxyCategoricalDtype
        """
        result = cls()
        result._parent = parent
        result._column_name = column_name
        result._materializer = materializer
        result._dtype = dtype
        return result

    def _get_dtype(self):
        """
        Get the categories dtype.

        Returns
        -------
        dtype
        """
        if self._dtype is None:
            self._dtype = self.categories.dtype
        return self._dtype

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        This object is serialized into a ``pandas.CategoricalDtype`` as an actual proxy can't be
        properly serialized because of the references it stores for its potentially distributed parent.
        """
        return (pandas.CategoricalDtype, (self.categories, self.ordered))

    @property
    def _categories(self):
        """
        Get materialized categorical values.

        Returns
        -------
        pandas.Index
        """
        if not self._is_materialized:
            self._materialize_categories()
        return self._categories_val

    @_categories.setter
    def _categories(self, categories):
        """
        Set new categorical values.

        Parameters
        ----------
        categories : list-like
        """
        self._categories_val = categories
        self._parent = None  # The parent is not required any more
        self._materializer = None
        self._dtype = None

    @property
    def _is_materialized(self) -> bool:
        """
        Check whether categorical values were already materialized.

        Returns
        -------
        bool
        """
        return self._categories_val is not None

    def _materialize_categories(self):
        """Materialize actual categorical values."""
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=self._parent is None,
            extra_log="attempted to materialize categories with parent being 'None'",
        )
        categoricals = self._materializer(self._parent, self._column_name)
        self._categories = categoricals.categories
        self._ordered = categoricals.ordered


def get_categories_dtype(
    cdt: Union[LazyProxyCategoricalDtype, pandas.CategoricalDtype]
) -> DtypeObj:
    """
    Get the categories dtype.

    Parameters
    ----------
    cdt : LazyProxyCategoricalDtype or pandas.CategoricalDtype

    Returns
    -------
    dtype
    """
    return (
        cdt._get_dtype()
        if isinstance(cdt, LazyProxyCategoricalDtype)
        else cdt.categories.dtype
    )


def extract_dtype(value) -> DtypeObj | pandas.Series:
    """
    Extract dtype(s) from the passed `value`.

    Parameters
    ----------
    value : object

    Returns
    -------
    DtypeObj or pandas.Series of DtypeObj
    """
    try:
        dtype = pandas.api.types.pandas_dtype(value)
    except (TypeError, ValueError):
        dtype = pandas.Series(value).dtype

    return dtype
