from abc import ABC


class BaseCompatibilityBasePandasDataset(ABC):
    def to_csv(self, *args, **kwargs):
        pass

    def to_json(self, *args, **kwargs):
        pass

    def to_markdown(self, *args, **kwargs):
        pass

    def to_pickle(self, *args, **kwargs):
        pass

    def to_latex(self, *args, **kwargs):
        pass
