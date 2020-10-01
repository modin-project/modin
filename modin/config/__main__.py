from . import *  # noqa: F403, F401
from .pubsub import Parameter


def print_config_help():
    for objname in sorted(globals()):
        obj = globals()[objname]
        if isinstance(obj, type) and issubclass(obj, Parameter) and not obj.is_abstract:
            print(f"{obj.get_help()}\n\tCurrent value: {obj.get()}")


if __name__ == "__main__":
    print_config_help()
