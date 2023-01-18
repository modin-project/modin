# create a new kernel configuration
python -m ipykernel install --user --name python3mpi --display-name "Python 3 (ipykernel) with MPI"
# update created kernel configuration for mpi execution
sed -i 's/\"argv\"\: \[\n/\"argv\"\: \[ \"mpiexec\"\, \"-n\"\, \"1\"\,/' $(jupyter kernelspec list | grep python3mpi | awk '{print $2"/kernel.json"}')