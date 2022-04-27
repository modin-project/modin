from setuptools import setup, find_packages
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dask_deps = ["dask>=2.22.0,<2022.2.0", "distributed>=2.22.0,<2022.2.0"]
# TODO: remove redis dependency when resolving GH#4398
ray_deps = ["ray[default]>=1.4.0", "pyarrow>=4.0.1", "redis>=3.5.0,<4.0.0"]
remote_deps = ["rpyc==4.1.5", "cloudpickle", "boto3"]
spreadsheet_deps = ["modin-spreadsheet>=0.1.0"]
sql_deps = ["dfsql>=0.4.2", "pyparsing<=2.4.7"]
all_deps = dask_deps + ray_deps + remote_deps + spreadsheet_deps

setup(
    name="modin",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Modin: Make your pandas code run faster by changing one line of code.",
    packages=find_packages(exclude=["scripts", "scripts.*"]),
    include_package_data=True,
    license="Apache 2",
    url="https://github.com/modin-project/modin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pandas==1.4.2", "packaging", "numpy>=1.18.5", "fsspec"],
    extras_require={
        # can be installed by pip install modin[dask]
        "dask": dask_deps,
        "ray": ray_deps,
        "remote": remote_deps,
        "spreadsheet": spreadsheet_deps,
        "sql": sql_deps,
        "all": all_deps,
    },
    python_requires=">=3.8",
)
