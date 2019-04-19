import sys

if sys.version_info[0] == 3:
    import builtins
else:
    import __builtin__ as builtins


def handle_ray_task_error(e):
    for s in e.traceback_str.split("\n")[::-1]:
        if "Error" in s or "Exception" in s:
            try:
                raise getattr(builtins, s.split(":")[0])("".join(s.split(":")[1:]))
            except AttributeError:
                break
    raise e
