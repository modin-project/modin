class Function(object):
    def __init__(self):
        raise ValueError(
            "Please use {}.register instead of the constructor".format(
                type(self).__name__
            )
        )

    @classmethod
    def register(cls, func, **kwargs):
        raise NotImplementedError("Implement in children classes")
