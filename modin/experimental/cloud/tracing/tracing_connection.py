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

import threading
import time
import collections

from rpyc.core import brine, consts, netref

from ..rpyc_proxy import WrappingConnection

_msg_to_name = collections.defaultdict(dict)
for name in dir(consts):
    if name.upper() == name:
        category, _ = name.split("_", 1)
        _msg_to_name[category][getattr(consts, name)] = name
_msg_to_name = dict(_msg_to_name)


class _Logger:
    def __init__(self, conn, logname):
        self.conn = conn
        self.logname = logname

    def __enter__(self):
        with self.conn.logLock:
            self.conn.logfiles.add(self.logname)
            with open(self.logname, "a") as out:
                out.write(f"------------[new trace at {time.asctime()}]----------\n")
        return self

    def __exit__(self, *a, **kw):
        with self.conn.logLock:
            self.conn.logfiles.remove(self.logname)


class TracingWrappingConnection(WrappingConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.logLock = threading.RLock()
        self.timings = {}
        with open("rpyc-trace.log", "a") as out:
            out.write(f"------------[new trace at {time.asctime()}]----------\n")
        self.logfiles = set(["rpyc-trace.log"])

    @classmethod
    def __stringify(cls, args):
        if isinstance(args, (tuple, list)):
            return tuple(cls.__stringify(i) for i in args)
        if isinstance(args, netref.BaseNetref):
            return str(args.____id_pack__)
        return args

    @classmethod
    def __to_text(cls, args):
        return str(cls.__stringify(args))

    def _send(self, msg, seq, args):
        str_args = self.__to_text(args).replace("\r", "").replace("\n", "\tNEWLINE\t")
        if msg == consts.MSG_REQUEST:
            handler, _ = args
            str_handler = f":req={_msg_to_name['HANDLE'][handler]}"
        else:
            str_handler = ""
        with self.logLock:
            for logfile in self.logfiles:
                with open(logfile, "a") as out:
                    out.write(
                        f"send:msg={_msg_to_name['MSG'][msg]}:seq={seq}{str_handler}:args={str_args}\n"
                    )
        self.timings[seq] = time.time()
        return super()._send(msg, seq, args)

    def _dispatch(self, data):
        """tracing only"""
        got1 = time.time()
        try:
            return super()._dispatch(data)
        finally:
            got2 = time.time()
            msg, seq, args = brine.load(data)
            sent = self.timings.pop(seq, got1)
            if msg == consts.MSG_REQUEST:
                handler, args = args
                str_handler = f":req={_msg_to_name['HANDLE'][handler]}"
            else:
                str_handler = ""
            str_args = (
                self.__to_text(args).replace("\r", "").replace("\n", "\tNEWLINE\t")
            )
            with self.logLock:
                for logfile in self.logfiles:
                    with open(logfile, "a") as out:
                        out.write(
                            f"recv:timing={got1 - sent}+{got2 - got1}:msg={_msg_to_name['MSG'][msg]}:seq={seq}{str_handler}:args={str_args}\n"
                        )

    def _log_extra(self, logname):
        return _Logger(self, logname)
