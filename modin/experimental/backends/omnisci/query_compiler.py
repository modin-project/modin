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

from modin.backends.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage

import abc


def DFAlgNotSupported(fn_name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            "{} is not yet suported in DFAlgQueryCompiler".format(fn_name)
        )

    return fn


class DFAlgQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
        with a lazy DataFrame Algebra based backend."""

    def __init__(self, frame):
        self._modin_frame = frame

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    def copy(self):
        return self.__constructor__(self._expr.copy())

    def getitem_column_array(self, key, numeric=False):
        if numeric:
            new_modin_frame = self._modin_frame.mask(col_numeric_idx=key)
        else:
            new_modin_frame = self._modin_frame.mask(col_indices=key)
        return self.__constructor__(new_modin_frame)

    def _get_index(self):
        return self._modin_frame.index

    def _set_index(self, index):
        self._modin_frame.index = index

    def _get_columns(self):
        return self._modin_frame.columns

    def _set_columns(self, columns):
        self._modin_frame.columns = columns

    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    __and__ = DFAlgNotSupported("__and__")
    __or__ = DFAlgNotSupported("__or__")
    __rand__ = DFAlgNotSupported("__rand__")
    __ror__ = DFAlgNotSupported("__ror__")
    __rxor__ = DFAlgNotSupported("__rxor__")
    __xor__ = DFAlgNotSupported("__xor__")
    abs = DFAlgNotSupported("abs")
    add = DFAlgNotSupported("add")
    add_prefix = DFAlgNotSupported("add_prefix")
    add_suffix = DFAlgNotSupported("add_suffix")
    all = DFAlgNotSupported("all")
    any = DFAlgNotSupported("any")
    apply = DFAlgNotSupported("apply")
    applymap = DFAlgNotSupported("applymap")
    astype = DFAlgNotSupported("astype")
    back = DFAlgNotSupported("back")
    clip = DFAlgNotSupported("clip")
    combine = DFAlgNotSupported("combine")
    combine_first = DFAlgNotSupported("combine_first")
    concat = DFAlgNotSupported("concat")
    count = DFAlgNotSupported("count")
    cummax = DFAlgNotSupported("cummax")
    cummin = DFAlgNotSupported("cummin")
    cumprod = DFAlgNotSupported("cumprod")
    cumsum = DFAlgNotSupported("cumsum")
    describe = DFAlgNotSupported("describe")
    diff = DFAlgNotSupported("diff")
    drop = DFAlgNotSupported("drop")
    dropna = DFAlgNotSupported("dropna")
    eq = DFAlgNotSupported("eq")
    eval = DFAlgNotSupported("eval")
    fillna = DFAlgNotSupported("fillna")
    first_valid_index = DFAlgNotSupported("first_valid_index")
    floordiv = DFAlgNotSupported("floordiv")
    free = DFAlgNotSupported("free")
    front = DFAlgNotSupported("front")
    ge = DFAlgNotSupported("ge")
    get_dummies = DFAlgNotSupported("get_dummies")
    getitem_row_array = DFAlgNotSupported("getitem_row_array")
    groupby_agg = DFAlgNotSupported("groupby_agg")
    groupby_reduce = DFAlgNotSupported("groupby_reduce")
    gt = DFAlgNotSupported("gt")
    head = DFAlgNotSupported("head")
    idxmax = DFAlgNotSupported("idxmax")
    idxmin = DFAlgNotSupported("idxmin")
    insert = DFAlgNotSupported("insert")
    isin = DFAlgNotSupported("isin")
    isna = DFAlgNotSupported("isna")
    last_valid_index = DFAlgNotSupported("last_valid_index")
    le = DFAlgNotSupported("le")
    lt = DFAlgNotSupported("lt")
    max = DFAlgNotSupported("max")
    mean = DFAlgNotSupported("mean")
    median = DFAlgNotSupported("median")
    memory_usage = DFAlgNotSupported("memory_usage")
    min = DFAlgNotSupported("min")
    mod = DFAlgNotSupported("mod")
    mode = DFAlgNotSupported("mode")
    mul = DFAlgNotSupported("mul")
    ne = DFAlgNotSupported("ne")
    negative = DFAlgNotSupported("negative")
    notna = DFAlgNotSupported("notna")
    nunique = DFAlgNotSupported("nunique")
    pow = DFAlgNotSupported("pow")
    prod = DFAlgNotSupported("prod")
    quantile_for_list_of_values = DFAlgNotSupported("quantile_for_list_of_values")
    quantile_for_single_value = DFAlgNotSupported("quantile_for_single_value")
    query = DFAlgNotSupported("query")
    rank = DFAlgNotSupported("rank")
    reindex = DFAlgNotSupported("reindex")
    reset_index = DFAlgNotSupported("reset_index")
    rfloordiv = DFAlgNotSupported("rfloordiv")
    rmod = DFAlgNotSupported("rmod")
    round = DFAlgNotSupported("round")
    rpow = DFAlgNotSupported("rpow")
    rsub = DFAlgNotSupported("rsub")
    rtruediv = DFAlgNotSupported("rtruediv")
    skew = DFAlgNotSupported("skew")
    sort_index = DFAlgNotSupported("sort_index")
    std = DFAlgNotSupported("std")
    sub = DFAlgNotSupported("sub")
    sum = DFAlgNotSupported("sum")
    tail = DFAlgNotSupported("tail")
    to_numpy = DFAlgNotSupported("to_numpy")
    transpose = DFAlgNotSupported("transpose")
    truediv = DFAlgNotSupported("truediv")
    update = DFAlgNotSupported("update")
    var = DFAlgNotSupported("var")
    view = DFAlgNotSupported("view")
    where = DFAlgNotSupported("where")
    write_items = DFAlgNotSupported("write_items")
