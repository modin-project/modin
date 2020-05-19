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

import server.server as server
import server_worker.server_worker as worker
import uuid
import os


class OmnisciServer:
    _server = None
    _worker = None

    @classmethod
    def start_server(cls):
        if cls._server is None:
            server_path = os.environ.get("OMNISCI_SERVER")
            if server_path is None:
                raise KeyError(
                    "you should set OMNISCI_SERVER variable to provide path to OmniSci server executable"
                )
            cls._server = server.OmnisciServer(
                server_path, 6001, "modin_db", 6002, 6003,
            )
            cls._server.launch()
            cls._worker = worker.OmnisciServerWorker(cls._server)
            cls._worker.connect_to_server()
            cls._worker.create_database("modin_db")

    @classmethod
    def stop_server(cls):
        if cls._server is not None:
            cls._server.terminate()
            cls._server = None
            cls._worker = None

    def __init__(self):
        self.start_server()

    @classmethod
    def put(cls, df):
        frame_id = "frame_" + str(uuid.uuid4()).replace("-", "")
        cls._worker.import_data_from_pd_df(frame_id, df, df.columns, df.dtypes)
        return frame_id


def put_to_omnisci(df):
    return OmnisciServer().put(df)
