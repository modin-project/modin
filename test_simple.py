import os
import ray
import sys;
import pyarrow as pa
from pyarrow.csv import read_csv

os.environ["MODIN_BACKEND"] = "Omnisci"
# os.environ["MODIN_BACKEND"] = "Pyarrow"
#os.environ["MODIN_BACKEND"] = "Pandas"
os.environ["MODIN_ENGINE"] = "Ray"
# os.environ["MODIN_ENGINE"] = "Python"
sys.setdlopenflags( 1|256 )    # RTLD_LAZY+RTLD_GLOBAL
import modin.experimental.pandas as pd

#ray.init(num_cpus=1, plasma_directory="/localdisk1/amalakho/plasma.db", object_store_memory=78643200)

a = pd.DataFrame([[1, 10, 100], [2, 20, 200], [3, 30, 300]], columns=['a', 'b', 'c'])
a = a[['a', 'b']]
print(a)

my_df = pd.read_csv("/users/amalakho/projects/modin/examples/data/boston_housing.csv")
a = my_df[['INDUS', 'AGE']]
print(a)
