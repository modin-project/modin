# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import glob
import os
import shutil
import uuid

from fuzzydata.clients.modin import ModinWorkflow
from fuzzydata.core.generator import generate_workflow

from modin.config import Engine


def test_fuzzydata_sample_workflow():
    # Workflow Generation Options
    wf_name = str(uuid.uuid4())[:8]  # Unique name for the generated workflow
    num_versions = 10  # Number of unique CSV files to generate
    cols = 33  # Columns in Base Artifact
    rows = 1000  # Rows in Base Artifact
    bfactor = 1.0  # Branching Factor - 0.1 is linear, 10.0 is star-like
    exclude_ops = ["groupby"]  # In-Memory groupby operations cause issue #4287
    matfreq = 2  # How many operations to chain before materialization

    engine = Engine.get().lower()

    # Create Output Directory for Workflow Data
    base_out_directory = (
        f"/tmp/fuzzydata-test-wf-{engine}/"  # Must match corresponding github-action
    )
    if os.path.exists(base_out_directory):
        shutil.rmtree(base_out_directory)
    output_directory = f"{base_out_directory}/{wf_name}/"
    os.makedirs(output_directory, exist_ok=True)

    # Start Workflow Generation
    workflow = generate_workflow(
        workflow_class=ModinWorkflow,
        name=wf_name,
        num_versions=num_versions,
        base_shape=(cols, rows),
        out_directory=output_directory,
        bfactor=bfactor,
        exclude_ops=exclude_ops,
        matfreq=matfreq,
        wf_options={"modin_engine": engine},
    )

    # Assertions that the workflow generation worked correctly
    assert len(workflow) == num_versions
    assert len(list(glob.glob(f"{output_directory}/artifacts/*.csv"))) == len(
        workflow.artifact_dict
    )
    assert os.path.exists(f"{output_directory}/{workflow.name}_operations.json")
    assert os.path.getsize(f"{output_directory}/{workflow.name}_operations.json") > 0
    assert os.path.exists(f"{output_directory}/{workflow.name}_gt_graph.csv")
