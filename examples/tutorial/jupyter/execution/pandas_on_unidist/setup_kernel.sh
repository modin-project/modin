kernel_prefix=$(which python | dirname $(awk '{print $0}') | dirname $(awk '{print $0}'))
# create a new kernel configuration
python -m ipykernel install --name python3mpi --display-name "Python 3 (ipykernel) with MPI" --prefix $kernel_prefix
# update created kernel configuration for mpi execution
configuration_file=$(jupyter kernelspec list | grep python3mpi | awk '{print $2"/kernel.json"}')
sed -i 's/\"argv\"\: \[/\"argv\"\: \[ \"mpiexec\"\, \"-n\"\, \"1\"\,/' $configuration_file