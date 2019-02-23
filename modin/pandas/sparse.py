import pandas
from modin.error_message import ErrorMessage


class SparseArray(pandas.SparseArray):
    def __init__(
        self,
        data,
        sparse_index=None,
        index=None,
        fill_value=None,
        kind="integer",
        dtype=None,
        copy=False,
    ):
        ErrorMessage.default_to_pandas("`SparseArray`")
        super(SparseArray, self).__init__(
            data,
            sparse_index=sparse_index,
            index=index,
            fill_value=fill_value,
            kind=kind,
            dtype=dtype,
            copy=copy,
        )


class SparseSeries(pandas.SparseSeries):
    def __init__(
        self,
        data=None,
        index=None,
        sparse_index=None,
        kind="block",
        fill_value=None,
        name=None,
        dtype=None,
        copy=False,
        fastpath=False,
    ):
        ErrorMessage.default_to_pandas("`SparseSeries`")
        super(SparseSeries, self).__init__(
            data=data,
            index=index,
            sparse_index=sparse_index,
            kind=kind,
            fill_value=fill_value,
            name=name,
            dtype=dtype,
            copy=copy,
            fastpath=fastpath,
        )


class SparseDataFrame(pandas.SparseDataFrame):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        default_kind=None,
        default_fill_value=None,
        dtype=None,
        copy=False,
    ):
        ErrorMessage.default_to_pandas("`SparseDataFrame`")
        super(SparseDataFrame, self).__init__(
            data=data,
            index=index,
            columns=columns,
            default_kind=default_kind,
            default_fill_value=default_fill_value,
            dtype=dtype,
            copy=copy,
        )
