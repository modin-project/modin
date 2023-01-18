# Jupyter notebook examples to run with PandasOnUnidist

Currently, Modin supports `PandasOnUnidist` execution only with MPI backend of [unidist](https://github.com/modin-project/unidist).
There are some specifics on how to run a jupyter notebook with MPI, namely, you should use `mpiexec` command.

```bash
mpiexec -n 1 jupyter notebook
```

**Important**

MPI is not reliable yet to work in interactive environment such as jupyter notebooks. Thus, some things may not work.
For example, if you are experiencing the error `The kernel appears to have died. It will restart automatically.`,
you may want to modify `kernel.json` in order to fix the problem.

For simplicity, a Linux user can use the `setup_kernel.sh` script to install kernel with MPI. Otherwise, you need to follow these steps:

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

3. Third, modify `kernel.json` file in `python3mpi` folder to add `mpiexec -n 1` commamd
(like "mpiexec", "-n", "1") to the beginning of the launched command (`argv`).

4. Fourth, but optional, you can change `display_name` in `kernel.json` file to something like `Python 3 (ipykernel) with MPI`.
That way you can specifically select the Python kernel with MPI-enabled using the dropdown menu in your browser.

## Run Jupyter Notebooks with PandasOnUnidist

After the steps above are done, you can run a jupyter notebook with `PandasOnUnidist` in a normal way.

```bash
jupyter notebook
```
