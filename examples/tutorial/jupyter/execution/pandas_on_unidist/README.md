# Jupyter notebook examples to run with PandasOnUnidist

Currently, Modin supports `PandasOnUnidist` execution only with MPI backend of [unidist](https://github.com/modin-project/unidist).
There are some specifics on how to run a jupyter notebook with MPI, namely, you should use `mpiexec` command.

```bash
mpiexec -n 1 jupyter notebook
```

**Important**

MPI is not reliable yet to work in interactive environment such as jupyter notebooks. Thus, some things may not work.
For example, if you are experiencing the error `The kernel appears to have died. It will restart automatically.`,
you may want to modify `kernel.json` file or create a new one in order to fix the problem.

For simplicity, you can just run `setup_kernel.py` script located in this directory. This will install a new MPI enabled kernel,
which you can then select using the dropdown menu in your browser. Otherwise, you can follow the steps below:

1. First, what you should do is locate `kernel.json` file with `jupyter kernelspec list` command. It should generally be like this.

```bash
jupyter kernelspec list

Available kernels:
  python3    $PREFIX/share/jupyter/kernels/python3
```

`kernel.json` file should be located in `python3` folder.

2. Second, you should make a copy of the `python3` folder, say to `python3mpi` folder.

```bash
cp -r $PREFIX/share/jupyter/kernels/python3 $PREFIX/share/jupyter/kernels/python3mpi
```

3. Third, modify `kernel.json` file in `python3mpi` folder to add `mpiexec -n 1` command
(like "mpiexec", "-n", "1") to the beginning of the launched command (`argv`).

4. Fourth, change `display_name` in `kernel.json` file to something like `Python 3 (ipykernel) with MPI`.
That way you can specifically select the Python kernel with MPI-enabled using the dropdown menu in your browser.

## Run Jupyter Notebooks with PandasOnUnidist

After the `setup_kernel.py` script is run or the steps above are done, you can run a jupyter notebook with `PandasOnUnidist` in a normal way.

```bash
jupyter notebook
```
