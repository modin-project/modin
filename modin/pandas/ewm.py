import pandas

from modin.utils import _inherit_docstrings


@_inherit_docstrings(pandas.core.window.rolling.Window)
class ExponentialMovingWindow:
    def _init(self, dataframe, ewm_kwargs):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.ewm_kwargs = ewm_kwargs

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_mean(
                self.ewm_kwargs, *args, **kwargs
            )
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_sum(
                self.ewm_kwargs, *args, **kwargs
            )
        )

    def std(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_std(
                self.ewm_kwargs, *args, **kwargs
            )
        )

    def var(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_var(
                self.ewm_kwargs, *args, **kwargs
            )
        )

    # cor and cov get special treatment because they can take another dataframe or
    # series, from which we have to extract the query compiler.

    def cov(self, other=None, *args, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, (DataFrame, Series)):
            other = other._query_compiler

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_cov(
                self.ewm_kwargs, other, *args, **kwargs
            )
        )

    def corr(self, *args, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, (DataFrame, Series)):
            other = other._query_compiler

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.ewm_corr(
                self.ewm_kwargs, other, *args, **kwargs
            )
        )
