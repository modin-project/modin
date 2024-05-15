# Jupyter notebook examples to run with Modin

Currently we provide tutorial notebooks for the following execution backends:

- [PandasOnRay](https://modin.readthedocs.io/en/latest/development/using_pandas_on_ray.html)
- [PandasOnDask](https://modin.readthedocs.io/en/latest/development/using_pandas_on_dask.html)
- [PandasOnMPI through unidist](https://modin.readthedocs.io/en/latest/development/using_pandas_on_mpi.html)

## Creating a development environment

To get required dependencies for `PandasOnRay`, `PandasOnDask` and `PandasOnUnidist` Jupyter Notebooks
you should create a development environment with `pip`
using `requirements.txt` file located in the respective directory:

```bash
pip install -r execution/pandas_on_ray/requirements.txt
```

to install dependencies needed to run notebooks with Modin on `PandasOnRay` execution or

```bash
pip install -r execution/pandas_on_dask/requirements.txt
```

to install dependencies needed to run notebooks with Modin on `PandasOnDask` execution or

```bash
pip install -r execution/pandas_on_unidist/requirements.txt
```

to install dependencies needed to run notebooks with Modin on `PandasOnUnidist` execution.

**Note:** Sometimes pip is installing every version of a package. If you encounter that issue,
please install every package listed in `requirements.txt` file individually with `pip install <package>`.

## Run Jupyter Notebooks

A Jupyter Notebook server can be run from the current directory as follows:

```bash
jupyter notebook
```

Navigate to a concrete notebook (for example, to the `execution/pandas_on_ray/local/exercise_1.ipynb`).

**Note:** Since there are some specifics regarding the run of jupyter notebooks with the `Unidist` engine,
refer to [PandasOnUnidist](https://github.com/modin-project/modin/blob/main/examples/tutorial/jupyter/execution/pandas_on_unidist/README.md) document
to get more information on the matter.