![LOGO](../../../img/MODIN_ver2_hrz.png)

<center>
<h1>Scale your pandas workflows on a Ray cluster</h2>
</center>

**NOTE**: Before completing the exercise, please read the full instructions in the 
[Modin documenation](https://modin--6872.org.readthedocs.build/en/6872/getting_started/using_modin/using_modin_cluster.html).

The basic steps to run the script on a remote Ray cluster are:

Step 1. Install the necessary dependencies

```bash
pip install boto3
```

Step 2. Setup your AWS credentials.

```bash
aws configure
```

Step 3. Modify configuration file and start up the Ray cluster.

```bash
ray up modin-cluster.yaml
```

Step 4. Submit your script to the remote cluster.

```bash
ray submit modin-cluster.yaml exercise_5.py
```

Step 5. Shut down the Ray remote cluster.

```bash
ray down 
