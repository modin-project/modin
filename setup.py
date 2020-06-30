from setuptools import setup, find_packages
import versioneer
import os
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel

    HAS_WHEEL = True
except ImportError:
    HAS_WHEEL = False

with open("README.md", "r") as fh:
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


dask_deps = ["dask>=2.1.0", "distributed>=2.3.2"]
ray_deps = ["ray==0.8.6", "pyarrow<0.17"]
if "SETUP_PLAT_NAME" in os.environ:
    if "win" in os.environ["SETUP_PLAT_NAME"]:
        all_deps = dask_deps
    else:
        all_deps = dask_deps + ray_deps
else:
    all_deps = dask_deps if os.name == "nt" else dask_deps + ray_deps

setup(
    name="modin",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    distclass=ModinDistribution,
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(),
    license="Apache 2",
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==1.0.5", "packaging"],
    extras_require={
        # can be installed by pip install modin[dask]
        "dask": dask_deps,
        "ray": ray_deps,
        "all": all_deps,
    },
    python_requires=">=3.6.1",
)
