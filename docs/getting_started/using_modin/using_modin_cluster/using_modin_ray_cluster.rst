================================
Using Modin in a AWS Ray Cluster
================================

.. note::
  | *Estimated Reading Time: 15 minutes*

Often in practice we have a need to exceed the capabilities of a single machine.
Modin works and performs well in both local mode and in a cluster environment.
The key advantage of Modin is that your notebook does not change between
local development and cluster execution. Users are not required to think about
how many workers exist or how to distribute and partition their data;
Modin handles all of this seamlessly and transparently.

.. image:: ../../../img/modin_cluster.png
   :alt: Modin cluster
   :align: center
   :scale: 90%

Extra requirements for AWS authentication
-----------------------------------------

First of all, install the necessary dependencies in your environment:

.. code-block:: bash

   pip install boto3

The next step is to setup your AWS credentials. One can set  `AWS_ACCESS_KEY_ID`, 
`AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` environment variables or  
just run the following command:

.. code-block:: bash

   aws configure

Starting and connecting to the cluster
--------------------------------------

This example starts 1 head node (m5.24xlarge) and 5 worker nodes (m5.24xlarge), 576 total CPUs.
You can check the `Amazon EC2 pricing`_ .

You can manually create AWS EC2 instances and configure them or just use the `Ray autoscaler` to 
create and initialize a Ray cluster on Amazon Web Service (AWS) using `Modin's Ray cluster setup config`_ .
You can read more about how to modify `Ray's autoscaler options`_ .

Detailed instructions can be found in `Ray's cluster docs`_.

To start up the Ray cluster, run the following command in your terminal:

.. code-block:: bash

   ray up modin-cluster.yaml

Once the head node has completed initialization, you can optionally connect to it by running the following command.

.. code-block:: bash

   ray attach modin-cluster.yaml

To exit the ssh session and return back into your local shell session, type:

.. code-block:: bash

   exit

Executing on a cluster environment
----------------------------------

.. note::
   Be careful when using the `Ray client`_ to connect to a remote cluster.
   We don't recommend this connection mode, beacuse it may not work. Known bugs:
   - https://github.com/ray-project/ray/issues/38713,
   - https://github.com/modin-project/modin/issues/6641.

Modin lets you instantly speed up your workflows with a large data by scaling pandas
on a cluster. In this tutorial, we will use a 12.5 GB `big_yellow.csv` file that was
created by concatenating a 200MB `NYC Taxi dataset`_ file 64 times. Preparing this
file was provided as part of our `Modin's Ray cluster setup config`_.

If you want use another dataset in your own script, you should provide it to each of
the cluster nodes in the same path. We recomnend doing this by customizing the
`setup_commands` section of the [configuration file](https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/modin-cluster.yaml).

To run any scripts on a remote cluster, you need to submit it to the ray. In this way,
the script file is sent to the the remote cluster head node and executed there. 

In this tutorial, we provide the `exercise_5.py`_ script, which read the data from the
CSV file and executed some pandas Dataframe function such as count, groupby and applymap.
As a result of the script, you will see the size of the file being read and the execution
time of each function.

You can submit this script to the existing remote cluster by running the following command.

.. code-block:: bash

   ray modin-cluster.yaml exercise_5.py

To download or upload files to the cluster head node, use `ray rsync_down` or `ray rsync_up`.
It may help you if you want to use some other Python modules that should be available to
execute your own script or download a result file after executing the script.

.. code-block:: bash

   # download a file from the cluster to the local computer:
   ray rsync_down modin-cluster.yaml '/path/on/cluster' '/local/path'
   # upload a file from the local computer to the cluster:
   ray rsync_up modin-cluster.yaml '/local/path' '/path/on/cluster'

Modin performance scales as the number of nodes and cores increases. The following
chart shows the performance of the read_csv operation with different number of nodes,
with improvements in performance as we increase the number of resources Modin can use.

.. image:: ../../../../examples/tutorial/jupyter/img/modin_cluster_perf.png
   :alt: Cluster Performance
   :align: center
   :scale: 90%

.. _`Ray's autoscaler options`: https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html#cluster-config
.. _`Ray's cluster docs`: https://docs.ray.io/en/latest/cluster/getting-started.html
.. _`NYC Taxi dataset`: https://modin-datasets.intel.com/testing/yellow_tripdata_2015-01.csv
.. _`Modin's Ray cluster setup config`: https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/modin-cluster.yaml
.. _`Amazon EC2 pricing`: https://aws.amazon.com/ec2/pricing/on-demand/
.. _`exercise_5.py`: https://github.com/modin-project/modin/blob/master/examples/tutorial/jupyter/execution/pandas_on_ray/cluster/exercise_5.py
.. _`Ray client`: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html