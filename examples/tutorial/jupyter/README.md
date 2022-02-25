# Jupyter notebook examples to run Modin on PandasOnRay or PandasOndask

## Creating a development environment

To get required dependencies for these Jupyter Notebooks
you should create a development environment with `pip`
using `requirements.txt` file located in the respective directory:

```bash
pip install -r execution/pandas_on_ray/requirements.txt
```

to install dependencies needed to run notebooks using Ray engine or

```bash
pip install -r execution/pandas_on_dask/requirements.txt
```

to install dependencies needed to run notebooks using Dask engine

**Note:** Sometimes pip is installing every version of a package. If you encounter that issue,
please install every package listed in `requirements.txt` file individually with `pip install <package>`.

## Run Jupyter Notebooks

A Jupyter Notebook server can be run from the current directory as follows:

```bash
jupyter notebook
```

Navigate to a concrete notebook (for example, to the `execution/pandas_on_ray/local/exercise_1.ipynb`).