from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="modin",
    version="0.5.4",
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(),
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==0.24.2", "ray==0.7.1"],
    extras_require={
        # can be installed by pip install modin[dask]
        "dask": ["dask==1.1.0", "distributed==1.25.0"]
    },
)
