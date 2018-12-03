from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="modin",
    version="0.2.5",
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(),
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==0.23.4", "ray==0.6.0"])
