from . import *


def get_help():
    for obj in globals().values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Parameter)
            and obj is not EnvironmentVariable
            and obj is not Parameter
        ):
            print(f"{obj.get_help()}\nvalue={obj.get()}")


if __name__ == "__main__":
    get_help()
