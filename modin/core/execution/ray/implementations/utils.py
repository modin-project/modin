import os
import pathlib
import re

S3_ADDRESS_REGEX = re.compile("[sS]3://(.*?)/(.*)")


def is_local_path(path_or_buf) -> bool:
    """
    Return True if the specified path_or_buf is a local path, False otherwise.

    Parameters
    ----------
    path_or_buf : str, path object or file-like object
        The path or buffer to check.

    Returns
    -------
    Whether the `path_or_buf` points to a local file.
    """
    if isinstance(path_or_buf, str):
        if S3_ADDRESS_REGEX.match(path_or_buf) is not None or "://" in path_or_buf:
            return False  # S3 or network path.
    if isinstance(path_or_buf, str) or isinstance(path_or_buf, pathlib.PurePath):
        return os.path.exists(path_or_buf)
    return False
