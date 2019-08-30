import builtins


def handle_ray_task_error(e):
    for s in e.traceback_str.split("\n")[::-1]:
        if "Error" in s or "Exception" in s:
            try:
                raise getattr(builtins, s.split(":")[0])("".join(s.split(":")[1:]))
            except AttributeError as att_err:
                if "module" in str(att_err) and builtins.__name__ in str(att_err):
                    pass
                else:
                    raise att_err
    raise e
