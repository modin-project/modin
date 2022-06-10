from abc import ABC

from modin.logging import LoggerBase


class BaseCompatibilityBasePandasDataset(ABC, LoggerBase):
    def max(self, *args, **kwargs):
        pass

    def min(self, *args, **kwargs):
        pass

    def mean(self, *args, **kwargs):
        pass

    def median(self, *args, **kwargs):
        pass

    def rank(self, *args, **kwargs):
        pass

    def reindex(self, *args, **kwargs):
        pass

    def rolling(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs):
        pass

    def sem(self, *args, **kwargs):
        pass

    def shift(self, *args, **kwargs):
        pass

    def skew(self, *args, **kwargs):
        pass

    def std(self, *args, **kwargs):
        pass

    def to_csv(self, *args, **kwargs):
        pass

    def to_json(self, *args, **kwargs):
        pass

    def to_markdown(self, *args, **kwargs):
        pass

    def to_latex(self, *args, **kwargs):
        pass

    def to_pickle(self, *args, **kwargs):
        pass

    def var(self, *args, **kwargs):
        pass
