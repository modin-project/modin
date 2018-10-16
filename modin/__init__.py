import os
import subprocess


def _git_version():
    def _execute_cmd_in_temp_env(cmd):
        # construct environment
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]

    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Check that this is a git repo first.
    if (
        subprocess.call(
            ["git", "-C", "./", "status"],
            stderr=subprocess.STDOUT,
            stdout=open(os.devnull, "w"),
        )
        == 0
    ):
        try:
            git_revision = _execute_cmd_in_temp_env(["git", "rev-parse", "HEAD"])
            rev_string = git_revision.strip().decode()
        except OSError:
            rev_string = "Unknown"
    else:
        rev_string = "Unknown"
    os.chdir(cwd)
    return rev_string


def get_execution_engine():
    # In the future, when there are multiple engines and different ways of
    # backing the DataFrame, there will have to be some changed logic here to
    # decide these things. In the meantime, we will use the currently supported
    # execution engine + backing (Pandas + Ray).
    return "Ray"


def get_partition_format():
    # See note above about engine + backing.
    return "Pandas"


__git_revision__ = _git_version()
__version__ = "0.2.0"
__execution_engine__ = get_execution_engine()
__partition_format__ = get_partition_format()

# We don't want these used outside of this file.
del _git_version
del get_execution_engine
del get_partition_format
