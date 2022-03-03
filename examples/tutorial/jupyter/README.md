# Jupyter notebook examples to run with Modin

Currently we provide tutorial notebooks for the following execution backends:

- [PandasOnRay](https://modin.readthedocs.io/en/latest/development/using_pandas_on_ray.html)
- [PandasOnDask](https://modin.readthedocs.io/en/latest/development/using_pandas_on_dask.html)
- [OmnisciOnNative](https://modin.readthedocs.io/en/latest/development/using_omnisci.html)

## Creating a development environment

To get required dependencies for `PandasOnRay` and  `PandasOnDask` Jupyter Notebooks
you should create a development environment with `pip`
using `requirements.txt` file located in the respective directory:

```bash
pip install -r execution/pandas_on_ray/requirements.txt
```

to install dependencies needed to run notebooks with Modin on `PandasOnRay` execution or

```bash
pip install -r execution/pandas_on_dask/requirements.txt
```

to install dependencies needed to run notebooks with Modin on `PandasOnDask` execution.

**Note:** Sometimes pip is installing every version of a package. If you encounter that issue,
please install every package listed in `requirements.txt` file individually with `pip install <package>`.

To get required dependencies for `OmnisciOnNative` Jupyter Notebooks
you should create a development environment with `conda`
using `jupyter_omnisci_env.yml` file located in the respective directory:

```bash
conda config --set channel_priority strict
conda env create -f execution/omnisci_on_native/jupyter_omnisci_env.yml
```

After the environment is created it needs to be activated:

```bash
conda activate jupyter_modin_on_omnisci
```

**Note:** `Omnisci` engine is available on Linux only for now.

## Run Jupyter Notebooks

A Jupyter Notebook server can be run from the current directory as follows:

```bash
jupyter notebook
```

Navigate to a concrete notebook (for example, to the `execution/pandas_on_ray/local/exercise_1.ipynb`).