from . import *


def print_config_help():
    for objname in sorted(globals()):
        obj = globals()[objname]
        if (
            isinstance(obj, type)
            and issubclass(obj, Parameter)
            and obj is not EnvironmentVariable
            and obj is not Parameter
        ):
            print(f"{obj.get_help()}\n\tCurrent value: {obj.get()}")


if __name__ == "__main__":
    print_config_help()
