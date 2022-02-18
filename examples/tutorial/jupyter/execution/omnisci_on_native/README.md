# Jupyter notebook examples to run Modin on OmniSci

## Creating a development environment

To get required dependencies for these Jupyter Notebooks
you should create a development environment with `conda`
using `jupyter_omnisci_env.yml` file located in the current directory:

```bash
conda config --set channel_priority strict
conda env create -f jupyter_omnisci_env.yml
```

After the environment is created it needs to be activated:

```bash
conda activate jupyter_modin_on_omnisci
```

## Run Jupyter Notebooks

A Jupyter Notebook example can be run as follows:

```bash
jupyter notebook exercise_1.ipynb
```

All subsequent examples can be run with the same way, or simply by switching via the links at the bottom of the examples.
