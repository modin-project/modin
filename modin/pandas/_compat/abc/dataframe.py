from abc import ABC

from modin.logging import LoggerBase


class BaseCompatibilityDataFrame(ABC, LoggerBase):
    def applymap(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def pivot_table(self, *args, **kwargs):
        pass

    def prod(self, *args, **kwargs):
        pass

    def replace(self, *args, **kwargs):
        pass

    def sum(self, *args, **kwargs):
        pass

    def to_parquet(self, *args, **kwargs):
        pass
