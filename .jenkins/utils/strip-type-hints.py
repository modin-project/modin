from strip_hints import strip_file_to_string
import os
import fnmatch
import sys
from subprocess import check_output
from shlex import split

py_version = sys.version_info
if py_version[0] >= 3 and py_version[1] >= 5:
    print("No need to strip type hint in python 3.5 or above, exit now")
    sys.exit(0)


# From https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python # noqa: E501
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


# Go up to modin root
file_dir = os.path.split(__file__)[0]
modin_path = check_output(split("git rev-parse --show-toplevel"), cwd=file_dir)

for path in find_files(modin_path, "*.py"):
    string = strip_file_to_string(
        path,
        to_empty=False,
        no_ast=False,
        no_colon_move=False,
        only_assigns_and_defs=False,
    )
    with open(path, "w") as f:
        f.write(string)
