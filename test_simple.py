import os
import sys;
import pyarrow as pa
from pyarrow.csv import read_csv
import ray
ray.init(num_cpus=1, plasma_directory="/localdisk1/amalakho/plasma.db", object_store_memory=78643200, ignore_reinit_error=True)

os.environ["MODIN_BACKEND"] = "Omnisci"
# os.environ["MODIN_BACKEND"] = "Pyarrow"
#os.environ["MODIN_BACKEND"] = "Pandas"
os.environ["MODIN_ENGINE"] = "Ray"
# os.environ["MODIN_ENGINE"] = "Python"
sys.setdlopenflags( 1|256 )    # RTLD_LAZY+RTLD_GLOBAL
import modin.experimental.pandas as pd

my_df = pd.read_csv("/users/amalakho/projects/modin/examples/data/boston_housing.csv")
a = my_df[['AGE', 'INDUS']]
print(a)

a = pd.DataFrame([[1, 10, 100], [2, 20, 200], [3, 30, 300]], columns=['a', 'b', 'c'])
a = a[['a', 'b']]
print(a)


cols = ["seller_name", "new_name"]
dtypes = {'seller_name':'category', 'new_name':'category'}
df = pd.read_csv("names_small.csv", names=cols, dtype=dtypes, delimiter=",")
#print(df)


data = { "id": ["id1", "id2", "id3", "id1", "id2", "id3", "id1", "id2", "id3", "id1"], }
df = pd.DataFrame(data)
df["id1"] = df["id"].astype("category")
a = pa.Table.from_pandas(df)
pdb.consumeArrowTable('testtable', a)
