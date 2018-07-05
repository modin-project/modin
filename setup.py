from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages
import sys
import platform

python_version = platform.python_version()[:3]

ray_whl = "ray"  # Fall back on pip install if no relevant wheel exists
ray_whl_prefix = "https://s3-us-west-2.amazonaws.com/ray-wheels/latest/"
if sys.platform.startswith('linux'):
    if python_version == "2.7":
        ray_whl = ray_whl_prefix + \
            "ray-0.4.0-cp27-cp27mu-manylinux1_x86_64.whl"
    elif python_version == "3.3":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp33-cp33m-manylinux1_x86_64.whl"
    elif python_version == "3.4":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp34-cp34m-manylinux1_x86_64.whl"
    elif python_version == "3.5":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp35-cp35m-manylinux1_x86_64.whl"
    elif python_version == "3.6":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp36-cp36m-manylinux1_x86_64.whl"
elif sys.platform == "darwin":
    if python_version == "2.7":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp27-cp27m-macosx_10_6_intel.whl"
    elif python_version == "3.4":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp34-cp34m-macosx_10_6_intel.whl"
    elif python_version == "3.5":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp35-cp35m-macosx_10_6_intel.whl"
    elif python_version == "3.6":
        ray_whl = ray_whl_prefix + "ray-0.4.0-cp36-cp36m-macosx_10_6_intel.whl"

setup(
    name="modin",
    version="0.0.2",
    description="Modin: Pandas on Ray - Make your pandas code run faster with "
                "a single line of code change.",
    packages=find_packages(),
    url="https://github.com/modin-project/modin",
    install_requires=["pandas==0.22",
                      "ray==0.4.0"],
    dependency_links=[ray_whl])
