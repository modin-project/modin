import os
from os.path import dirname, abspath


rootdir = dirname(abspath(__file__))
exclude_files = ["_version.py"]

with open("{}{}".format(dirname(rootdir), "/LISCENSE_HEADER"), "r") as f:
    # Lines to check each line individually
    header_lines = f.readlines()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        if file.endswith(".py") and file not in exclude_files:
            with open(filepath, "r+") as f:
                # Lines for line by line comparison
                py_file_lines = f.readlines()
                if len(py_file_lines):
                    first_non_comment_idx = next(
                        i for i in range(len(py_file_lines))
                        if not py_file_lines[i].startswith("# ")
                    )
                    py_file_lines = py_file_lines[first_non_comment_idx:]
                new_contents = "".join(header_lines + py_file_lines)
                f.seek(0)
                f.write(new_contents)
                f.truncate()
