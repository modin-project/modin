class Function(object):
    def __init__(self):
        raise ValueError(
            "Please use {}.register instead of the constructor".format(
                type(self).__name__
            )
        )

    @classmethod
    def call(cls, func, **call_kwds):
        raise NotImplementedError("Please implement in child class")

    @classmethod
    def register(cls, func, **kwargs):
        return cls.call(func, **kwargs)
