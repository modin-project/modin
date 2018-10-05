from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="modin",
    version="0.1.2",
    description="Modin: Pandas on Ray - Make your pandas code run faster with "
                "a single line of code change.",
    packages=find_packages(),
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==0.23.4", "ray==0.5.3"])
