Batch Pipline API Usage Guide
=============================

Modin provides an experimental batching feature that pipelines row-parallel queries. This feature 
is currently only supported for the ``PandasOnRay`` engine. Please note that this feature is experimental
and behavior or interfaces could be changed.

Usage examples
--------------

In examples below we build and run some pipelines. It is important to note that the queries passed to
the pipeline operate on Modin DataFrame partitions, which are backed by ``pandas``. When using ``pandas``-
module level functions, please make sure to import and use ``pandas`` rather than ``modin.pandas``.

Simple Batch Pipelining
^^^^^^^^^^^^^^^^^^^^^^^

This example walks through a simple batch pipeline in order to familiarize the user with the API.

.. code-block:: python

    from modin.experimental.batch import PandasQueryPipeline
    import modin.pandas as pd
    import numpy as np

    df = pd.DataFrame(
        np.random.randint(0, 100, (100, 100)),
        columns=[f"col {i}" for i in range(1, 101)],
    ) # Build the dataframe we will pipeline.
    pipeline = PandasQueryPipeline(df) # Build the pipeline.
    pipeline.add_query(lambda df: df + 1, is_output=True) # Add the first query and specify that
                                                          # it is an output query.
    pipeline.add_query(
        lambda df: df.rename(columns={f"col {i}":f"col {i-1}" for i in range(1, 101)})
    ) # Add a second query.
    pipeline.add_query(
        lambda df: df.drop(columns=['col 99']),
        is_output=True,
    ) # Add a third query and specify that it is an output query.
    new_df = pd.DataFrame(
        np.ones((100, 100)),
        columns=[f"col {i}" for i in range(1, 101)],
    ) # Build a second dataframe that we will pipeline now instead.
    pipeline.update_df(new_df) # Update the dataframe that we will pipeline to be `new_df`
                               # instead of `df`.
    result_dfs = pipeline.compute_batch() # Begin batch processing.

    # Print pipeline results
    print(f"Result of Query 1:\n{result_dfs[0]}")
    print(f"Result of Query 2:\n{result_dfs[1]}")
    # Output IDs can also be specified
    pipeline = PandasQueryPipeline(df) # Build the pipeline.
    pipeline.add_query(
        lambda df: df + 1,
        is_output=True,
        output_id=1,
    ) # Add the first query, specify that it is an output query, as well as specify an output id.
    pipeline.add_query(
        lambda df: df.rename(columns={f"col {i}":f"col {i-1}" for i in range(1, 101)})
    ) # Add a second query.
    pipeline.add_query(
        lambda df: df.drop(columns=['col 99']),
        is_output=True,
        output_id=2,
    ) # Add a third query, specify that it is an output query, and specify an output_id.
    result_dfs = pipeline.compute_batch() # Begin batch processing.

    # Print pipeline results - should be a dictionary mapping Output IDs to resulting dataframes:
    print(f"Mapping of Output ID to dataframe:\n{result_dfs}")
    # Print results
    for query_id, res_df in result_dfs.items():
        print(f"Query {query_id} resulted in\n{res_df}")

Batch Pipelining with Postprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A postprocessing function can also be provided when calling ``pipeline.compute_batch``. The example
below runs a similar pipeline as above, but the postprocessing function writes the output dfs to 
a parquet file.

.. code-block:: python

    from modin.experimental.batch import PandasQueryPipeline
    import modin.pandas as pd
    import numpy as np
    import os
    import shutil

    df = pd.DataFrame(
        np.random.randint(0, 100, (100, 100)),
        columns=[f"col {i}" for i in range(1, 101)],
    ) # Build the dataframe we will pipeline.
    pipeline = PandasQueryPipeline(df) # Build the pipeline.
    pipeline.add_query(
        lambda df: df + 1,
        is_output=True,
        output_id=1,
    ) # Add the first query, specify that it is an output query, as well as specify an output id.
    pipeline.add_query(
        lambda df: df.rename(columns={f"col {i}":f"col {i-1}" for i in range(1, 101)})
    ) # Add a second query.
    pipeline.add_query(
        lambda df: df.drop(columns=['col 99']),
        is_output=True,
        output_id=2,
    ) # Add a third query, specify that it is an output query, and specify an output_id.
    def postprocessing_func(df, output_id, partition_id):
        filepath = f"query_{output_id}/"
        os.makedirs(filepath, exist_ok=True)
        filepath += f"part-{partition_id:04d}.parquet"
        df.to_parquet(filepath)
        return df
    result_dfs = pipeline.compute_batch(
        postprocessor=postprocessing_func,
        pass_partition_id=True,
        pass_output_id=True,
    ) # Begin computation, pass in a postprocessing function, and specify that partition ID and 
      # output ID should be passed to that postprocessing function.

    print(os.system("ls query_1/")) # Should show `NPartitions.get()` parquet files - which
                                    # correspond to partitions of the output of query 1.
    print(os.system("ls query_2/")) # Should show `NPartitions.get()` parquet files - which
                                    # correspond to partitions of the output of query 2.

    for query_id, res_df in result_dfs.items():
        written_df = pd.read_parquet(f"query_{query_id}/")
        shutil.rmtree(f"query_{query_id}/") # Clean up
        print(f"Written and Computed DF are " +
              f"{'equal' if res_df.equals(written_df) else 'not equal'} for query {query_id}")

Batch Pipelining with Fan Out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the input dataframe to a query is small (consisting of only one partition), it is possible to 
induce additional parallelism using the ``fan_out`` argument. The ``fan_out`` argument replicates
the input partition, applies the query to each replica, and then coalesces all of the replicas back
to one partition using the ``reduce_fn`` that must also be specified when ``fan_out`` is ``True``.

It is possible to control the parallelism via the ``num_partitions`` parameter passed to the
constructor of the ``PandasQueryPipeline``. This parameter designates the desired number of partitions,
and defaults to ``NPartitions.get()`` when not specified. During fan out, the input partition is replicated
``num_partitions`` times. In the previous examples, ``num_partitions`` was not specified, and so defaulted
to ``NPartitions.get()``.

The example below demonstrates the usage of ``fan_out`` and ``num_partitions``. We first demonstrate
an example of a function that would benefit from this computation pattern:

.. code-block:: python

    import glob
    from PIL import Image
    import torchvision.transforms as T
    import torchvision

    transforms = T.Compose([T.ToTensor()])
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def contains_cat(image, model):
        image = transforms(image)
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in model([image])[0]['labels']]
        return 'cat' in labels

    def serial_query(df):
        """
        This function takes as input a dataframe with a single row corresponding to a folder
        containing images to parse. Each image in the folder is passed through a neural network
        that detects whether it contains a cat, in serial, and a new column is computed for the
        dataframe that counts the number of images containing cats.

        Parameters
        ----------
        df : a dataframe
            The dataframe to process
        
        Returns
        -------
        The same dataframe as before, with an additional column containing the count of images 
        containing cats.
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        img_folder = df['images'][0]
        images = sorted(glob.glob(f"{img_folder}/*.jpg"))
        cats = 0
        for img in images:
            cats = cats + 1 if contains_cat(Image.open(img), model) else cats
        df['cat_count'] = cats
        return df
    
To download the image files to test out this code, run the following bash script, which downloads
the images from the fast-ai-coco S3 bucket to a folder called ``images`` in your current working
directory:

.. code-block:: shell

    aws s3 cp s3://fast-ai-coco/coco_tiny.tgz . --no-sign-request; tar -xf coco_tiny.tgz; mkdir \
        images; mv coco_tiny/train/* images/; rm -rf coco_tiny; rm -rf coco_tiny.tgz

We can pipeline that code like so:

.. code-block:: python

    import modin.pandas as pd
    from modin.experimental.batch import PandasQueryPipeline
    from time import time
    df = pd.DataFrame([['images']], columns=['images'])
    pipeline = PandasQueryPipeline(df)
    pipeline.add_query(serial_query, is_output=True)
    serial_start = time()
    df_with_cat_count = pipeline.compute_batch()[0]
    serial_end = time()
    print(f"Result of pipeline:\n{df_with_cat_count}")

We can induce `8x` parallelism into the pipeline above by combining the ``fan_out`` and ``num_partitions`` parameters like so:

.. code-block:: python

    import modin.pandas as pd
    from modin.experimental.batch import PandasQueryPipeline
    import shutil
    from time import time
    df = pd.DataFrame([['images']], columns=['images'])
    desired_num_partitions = 8
    def parallel_query(df, partition_id):
        """
        This function takes as input a dataframe with a single row corresponding to a folder
        containing images to parse. It parses `total_images/desired_num_partitions` images every
        time it is called. A new column is computed for the dataframe that counts the number of
        images containing cats.

        Parameters
        ----------
        df : a dataframe
            The dataframe to process
        partition_id : int
            The partition id of the dataframe that this function runs on.
        
        Returns
        -------
        The same dataframe as before, with an additional column containing the count of images
        containing cats.
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        img_folder = df['images'][0]
        images = sorted(glob.glob(f"{img_folder}/*.jpg"))
        total_images = len(images)
        cats = 0
        start_index = partition_id * (total_images // desired_num_partitions)
        if partition_id == desired_num_partitions - 1: # Last partition must parse to end of list
            images = images[start_index:]
        else:
            end_index = (partition_id + 1) * (total_images // desired_num_partitions)
            images = images[start_index:end_index]
        for img in images:
            cats = cats + 1 if contains_cat(Image.open(img), model) else cats
        df['cat_count'] = cats
        return df

    def reduce_fn(dfs):
        """
        Coalesce the results of fanning out the `parallel_query` query.

        Parameters
        ----------
        dfs : a list of dataframes
            The resulting dataframes from fanning out `parallel_query`
        
        Returns
        -------
        A new dataframe whose `cat_count` column is the sum of the `cat_count` column of all
        dataframes in `dfs`
        """
        df = dfs[0]
        cat_count = df['cat_count'][0]
        for dataframe in dfs[1:]:
            cat_count += dataframe['cat_count'][0]
        df['cat_count'] = cat_count
        return df
    pipeline = PandasQueryPipeline(df, desired_num_partitions)
    pipeline.add_query(
        parallel_query,
        fan_out=True,
        reduce_fn=reduce_fn,
        is_output=True,
        pass_partition_id=True
    )
    parallel_start = time()
    df_with_cat_count = pipeline.compute_batch()[0]
    parallel_end = time()
    print(f"Result of pipeline:\n{df_with_cat_count}")
    print(f"Total Time in Serial: {serial_end - serial_start}")
    print(f"Total time with induced parallelism: {parallel_end - parallel_start}")
    shutil.rmtree("images/") # Clean up

Batch Pipelining with Dynamic Repartitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, it is also possible to hint to the Pipeline API to repartition after a node completes
computation. This is currently only supported if the input dataframe consists of only one partition.
The number of partitions after repartitioning is controlled by the ``num_partitions`` parameter
passed to the constructor of the ``PandasQueryPipeline``.

The following example demonstrates how to use the ``repartition_after`` parameter.

.. code-block:: python

    import modin.pandas as pd
    from modin.experimental.batch import PandasQueryPipeline
    import numpy as np

    small_df = pd.DataFrame([[1, 2, 3]]) # Create a small dataframe
    
    def increase_dataframe_size(df):
        import pandas
        new_df = pandas.concat([df] * 1000)
        new_df = new_df.reset_index(drop=True) # Get a new range index that isn't duplicated
        return new_df
    
    desired_num_partitions = 24 # We will repartition to 24 partitions

    def add_partition_id_to_df(df, partition_id):
        import pandas
        new_col = pandas.Series([partition_id]*len(df), name="partition_id", index=df.index)
        return pandas.concat([df, new_col], axis=1)
    
    pipeline = PandasQueryPipeline(small_df, desired_num_partitions)
    pipeline.add_query(increase_dataframe_size, repartition_after=True)
    pipeline.add_query(add_partition_id_to_df, pass_partition_id=True, is_output=True)
    result_df = pipeline.compute_batch()[0]
    print(f"Number of partitions passed to second query: " + 
          f"{len(np.unique(result_df['partition_id'].values))}")
    print(f"Result of pipeline:\n{result_df}")

