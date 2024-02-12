from __future__ import annotations

from typing import Any

from modin.dataframe_api_standard.tests.utils import ModinHandler


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        libraries = ["modin"]
        lib_handlers = [ModinHandler("modin")]

        metafunc.parametrize("library", lib_handlers, ids=libraries)
