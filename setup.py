from setuptools import setup, find_packages
import versioneer
import os
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel

    HAS_WHEEL = True
except ImportError:
    HAS_WHEEL = False

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if HAS_WHEEL:

    class ModinWheel(bdist_wheel):
        def finalize_options(self):
            bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            _, _, plat = bdist_wheel.get_tag(self)
            py = "py3"
            abi = "none"
            return py, abi, plat


class ModinDistribution(Distribution):
    def __init__(self, *attrs):
        Distribution.__init__(self, *attrs)
        if HAS_WHEEL:
            self.cmdclass["bdist_wheel"] = ModinWheel

    def is_pure(self):
        return False


dask_deps = ["dask>=2.12.0,<=2.19.0", "distributed>=2.12.0,<=2.19.0"]
ray_deps = ["ray>=1.0.0", "pyarrow==1.0"]
remote_deps = ["rpyc==4.1.5", "cloudpickle==1.4.1", "boto3==1.4.8"]

all_deps = dask_deps + ray_deps + remote_deps

setup(
    name="modin",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    distclass=ModinDistribution,
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(),
    include_package_data=True,
    license="Apache 2",
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==1.1.5", "packaging"],
    extras_require={
        # can be installed by pip install modin[dask]
        "dask": dask_deps,
        "ray": ray_deps,
        "remote": remote_deps,
        "all": all_deps,
    },
    python_requires=">=3.6.1",
)
