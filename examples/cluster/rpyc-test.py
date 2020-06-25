"""
This example demonstrates switching a local context to a remote cloud context with
using `read_csv` function.

NOTE: for using this script you should define next env variables
and manually start ray on local machine:
    ENV:
        LOCAL_MODIN_PATH
        REMOTE_MODIN_PATH
    Ray command:
        ray start --head  --object-store-memory=1000000000 --memory=1000000000 --plasma-directory=$TMPDIR

To get an object from a remote context, we can use `obtain` method:
    ```
    import rpyc
    local_df = rpyc.classic.obtain(remote_df)
    ```

To put an object into a remote context, we can use `deliver` method:
    ```
    import rpyc
    with modin.pandas.cloud_context():
        remote_df = rpyc.classic.deliver(local_df)
    ```

To move an object from one context to another need to fix:
    https://github.com/modin-project/modin/issues/672

"""

import ray
ray.init(address="auto")

import modin.pandas as pd

import os
local_modin_path = os.environ["LOCAL_MODIN_PATH"]
remote_modin_path = os.environ["REMOTE_MODIN_PATH"]

local_df = pd.read_csv(f"{local_modin_path}/modin/modin/pandas/test/data/blah.csv")
print(" type of local_df: \n{}\n local_df: \n{}\n memory_usage of local_df: \n{}\n".format(
    type(local_df), local_df, local_df.memory_usage()
))

with pd._CloudContext():
    test_df = pd.DataFrame([1,2,3,4])
    print(test_df.sum())
    print(type(test_df))

    remote_df = pd.read_csv(f"{remote_modin_path}/modin/modin/pandas/test/data/issue_621.csv")
    print(" type of remote_df: \n{}\n remote_df: \n{}\n memory_usage of remote_df: \n{}\n".format(
        type(remote_df), remote_df, remote_df.memory_usage()
    ))

local_df = pd.read_csv(f"{local_modin_path}/modin/modin/pandas/test/data/test_categories.csv")
print(" type of local_df: \n{}\n local_df: \n{}\n memory_usage of local_df: \n{}\n".format(
    type(local_df), local_df, local_df.memory_usage()
))

test_df = pd.DataFrame([4,3,2,1])
print(test_df.sum())
print(type(test_df))
