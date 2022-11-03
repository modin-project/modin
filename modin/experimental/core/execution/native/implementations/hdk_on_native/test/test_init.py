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


class TestInit:
    def test_num_threads(self):
        import os
        import modin.pandas as pd

        assert "OMP_NUM_THREADS" not in os.environ

        import modin.config as cfg

        cfg.IsExperimental.put(True)
        cfg.Engine.put("Native")
        cfg.StorageFormat.put("Hdk")
        pd.DataFrame()
        assert os.environ["OMP_NUM_THREADS"] == str(cfg.CpuCount.get())
