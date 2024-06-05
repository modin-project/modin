from setuptools import find_packages, setup

import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dask_deps = ["dask>=2.22.0", "distributed>=2.22.0"]
# ray==2.5.0 broken: https://github.com/conda-forge/ray-packages-feedstock/issues/100
ray_deps = ["ray>=2.1.0,!=2.5.0", "pyarrow>=10.0.1"]
mpi_deps = ["unidist[mpi]>=0.2.1"]
consortium_standard_deps = ["dataframe-api-compat>=0.2.7"]
spreadsheet_deps = ["modin-spreadsheet>=0.1.0"]
# Currently, Modin does not include `mpi` option in `all`.
# Otherwise, installation of modin[all] would fail because
# users need to have a working MPI implementation and
# certain software installed beforehand.
all_deps = dask_deps + ray_deps + spreadsheet_deps + consortium_standard_deps

# Distribute 'modin-autoimport-pandas.pth' along with binary and source distributions.
# This file provides the "import pandas before Ray init" feature if specific
# environment variable is set (see https://github.com/modin-project/modin/issues/4564).
cmdclass = versioneer.get_cmdclass()
extra_files = ["modin-autoimport-pandas.pth"]


class AddPthFileBuild(cmdclass["build_py"]):
    def _get_data_files(self):
        return (super()._get_data_files() or []) + [
            (".", ".", self.build_lib, extra_files)
        ]


class AddPthFileSDist(cmdclass["sdist"]):
    def make_distribution(self):
        self.filelist.extend(extra_files)
        return super().make_distribution()


cmdclass["build_py"] = AddPthFileBuild
cmdclass["sdist"] = AddPthFileSDist

setup(
    name="modin",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(exclude=["scripts", "scripts.*"]),
    include_package_data=True,
    license="Apache 2",
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas>=2.2,<2.3",
        "packaging>=21.0",
        "numpy>=1.22.4",
        "fsspec>=2022.11.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        # can be installed by pip install modin[dask]
        "dask": dask_deps,
        "ray": ray_deps,
        "mpi": mpi_deps,
        "consortium-standard": consortium_standard_deps,
        "spreadsheet": spreadsheet_deps,
        "all": all_deps,
    },
    python_requires=">=3.9",
)
