from strip_hints import strip_file_to_string
import os
import fnmatch

# From https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

# Go up to modin root
modin_path = os.path.relpath("../")

for path in find_files(modin_path, '*.py'):
    print("Working on {}".format(path))
    string = strip_file_to_string(path, to_empty=False, no_ast=False,
                                  no_colon_move=False, only_assigns_and_defs=False)
    with open(path, 'w') as f:
        f.write(string)
