# Jupyter notebook examples to run Modin on PandasOnRay

## Creating a development environment

To get required dependencies for these Jupyter Notebooks
you should create a development environment with `pip`
using `requirements.txt` file located in the current directory:

```bash
pip install -r requirements.txt
```

**Note:** Sometimes pip is installing every version of a package. If you encounter that issue,
please install every package listed in `requirements.txt` file individually with `pip install <package>`.

## Run Jupyter Notebooks

A Jupyter Notebook server example can be run from the current directory as follows:

```bash
jupyter notebook
```

And then navigate to the needed notebook (for example to the `execution/pandas_on_ray/local/exercise_1.ipynb`).