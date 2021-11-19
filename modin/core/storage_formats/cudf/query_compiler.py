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

import numpy as np
import pandas

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class cuDFQueryCompiler(PandasQueryCompiler):

    # Transpose
    # For transpose, we need to check that all the columns are the same type due to cudf limitations.
    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        if len(np.unique(self._modin_frame.dtypes.values)) != 1:
            return self.default_to_pandas(pandas.DataFrame.transpose)
        # Switch the index and columns and transpose the data within the blocks.
        return self.__constructor__(self._modin_frame.transpose())

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            unique_items = np.unique(item)
            # Basically, cudf do not support a nice multi-element assignment when the elements are not equal.
            # This is a nice workaround to use native-cudf element assignment per partition. In the case where
            # the elements and/or the indices are different we need to iterate over all the index combination
            # to assign each element individually.
            if (row_internal_indices == col_internal_indices).all() and len(
                unique_items
            ) == 1:
                partition.iloc[row_internal_indices] = unique_items[0]
            else:
                permutations_col = np.vstack(
                    [col_internal_indices] * len(col_internal_indices)
                ).T.flatten()
                permutations_row = np.hstack(
                    row_internal_indices * len(row_internal_indices)
                )
                for i, j, it in zip(permutations_row, permutations_col, item.flatten()):
                    partition.iloc[i, j] = it
            return partition

        new_modin_frame = self._modin_frame.apply_select_indices(
            axis=None,
            func=iloc_mut,
            row_labels=row_numeric_index,
            col_labels=col_numeric_index,
            new_index=self.index,
            new_columns=self.columns,
            keep_remaining=True,
            item_to_distribute=broadcasted_items,
        )
        return self.__constructor__(new_modin_frame)
