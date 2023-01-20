import sys
from ipykernel import kernelspec


default_make_ipkernel_cmd = kernelspec.make_ipkernel_cmd


def custom_make_ipkernel_cmd(
    mod="ipykernel_launcher", executable=None, extra_arguments=None
):
    mpi_arguments = ["mpiexec", "-n", "1"]
    arguments = default_make_ipkernel_cmd(mod, executable, extra_arguments)
    return mpi_arguments + arguments


kernelspec.make_ipkernel_cmd = custom_make_ipkernel_cmd

if __name__ == "__main__":
    kernel_name = "python3mpi"
    display_name = "Python 3 (ipykernel) with MPI"
    dest = kernelspec.install(
        kernel_name=kernel_name, display_name=display_name, prefix=sys.prefix
    )
    print(f"Installed kernelspec {kernel_name} in {dest}")
