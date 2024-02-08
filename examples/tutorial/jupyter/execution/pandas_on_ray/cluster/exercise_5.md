![LOGO](../../../img/MODIN_ver2_hrz.png)

<center>
<h1>Scale your pandas workflows by changing one line of code</h2>
</center>

# Exercise 5: Setting up cluster environment and executing on a cluster environment

**GOAL**: Learn how to set up a Ray cluster for Modin, connect Modin to a Ray cluster and run pandas queries on a cluster.

**NOTE**: This exercise has extra requirements. Read instructions carefully before attempting. 

**This exercise instructs users on how to start a 500+ core Ray cluster,
and it is not shut down until the end of exercise. Read instructions carefully.**

Often in practice we have a need to exceed the capabilities of a single machine.
Modin works and performs well in both local mode and in a cluster environment.
The key advantage of Modin is that your notebook does not change between
local development and cluster execution. Users are not required to think about
how many workers exist or how to distribute and partition their data;
Modin handles all of this seamlessly and transparently.

![Cluster](../../../img/modin_cluster.png)

**Extra requirements for AWS authentication**

First of all, install the necessary dependencies in your environment:

```bash
pip install boto3
```

The next step is to setup your AWS credentials. One can set  `AWS_ACCESS_KEY_ID`, 
`AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` environment variables or
just run the following command:

```bash
aws configure
```

## Starting and connecting to the cluster

This example starts 1 head node (m5.24xlarge) and 5 worker nodes (m5.24xlarge), 576 total CPUs.

Cost of this cluster can be found here: https://aws.amazon.com/ec2/pricing/on-demand/.

You can manually create AWS EC2 instances and configure them or just use the `Ray autoscaler` to create and initialize
a Ray cluster using the configuration file. This file is included in this directory and is called
[`modin-cluster.yaml`](https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/modin-cluster.yaml).
You can read more about how to modify `Ray cluster YAML Configuration file` here:
https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#cluster-yaml-configuration-options

Detailed instructions can be found here: https://docs.ray.io/en/latest/cluster/getting-started.html

To start up the Ray cluster, run the following command in your terminal:

```bash
ray up modin-cluster.yaml
```

Once the head node has completed initialization, you can optionally connect to it by running the following command.

```bash
ray attach modin-cluster.yaml
```

To exit the ssh session and return back into your local shell session, type:

```bash
exit
```

## Executing on a cluster environment

**NOTE**: Be careful when using the [Ray client](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html)
to connect to a remote cluster. We don't recommend this connection mode, beacuse it may not work. Known bugs:
[Ray issue #38713](https://github.com/ray-project/ray/issues/38713), [Modin issue #6641](https://github.com/modin-project/modin/issues/6641).

Modin lets you instantly speed up your workflows with a large data by scaling pandas on a cluster.
In this exercise, we will use a 12.5 GB `big_yellow.csv` file that was created by concatenating a 200MB 
[NYC Taxi dataset](https://modin-datasets.intel.com/testing/yellow_tripdata_2015-01.csv) 
file 64 times. Preparing this file was provided as part of our [configuration file](https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/modin-cluster.yaml).

If you want use another dataset in your own script, you should provide it to each of the cluster nodes in the same path.
We recomnend doing this by customizing the `setup_commands` section of the [configuration file](https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/modin-cluster.yaml).

To run any scripts on a remote cluster, you need to submit it to the ray. In this way, the script file 
is sent to the the remote cluster head node and executed there. 

In this exercise, we provide the `exercise_5.py` script, which read the data from the CSV file and executed 
some pandas Dataframe function such as count, groupby and applymap. As a result of the script, you will see 
the size of the file being read and the execution time of each function.

You can submit this script to the existing remote cluster by running the following command.

```bash
ray submit modin-cluster.yaml exercise_5.py
```

To download or upload files to the cluster head node, use `ray rsync_down` or `ray rsync_up`. It may help you if you want to use
some other Python modules that should be available to execute your own script or download a result file after executing the script.

```bash
# download a file from the cluster to the local computer:
ray rsync_down modin-cluster.yaml '/path/on/cluster' '/local/path'
# upload a file from the local computer to the cluster:
ray rsync_up modin-cluster.yaml '/local/path' '/path/on/cluster'
```

Modin performance scales as the number of nodes and cores increases. The following chart shows
the performance of the read_csv operation with different number of nodes, with improvements in
performance as we increase the number of resources Modin can use.

![ClusterPerf](../../../img/modin_cluster_perf.png)

## Shutting down the cluster

Now that we have finished the computation, we need to shut down the cluster with `ray down` command.

```bash
ray down modin-cluster.yaml
```

### This ends the cluster exercise
