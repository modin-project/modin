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

"""
Simple parser for rpyc traces produced by tracing_connection.py
"""

from rpyc.core import consts
import re
import collections
import sys


def read_log(fname):
    with open(fname, "rb") as inp:
        data = inp.read()
    data = data.decode("utf8", "xmlcharrefreplace")
    # split the last logging chunk
    data = data.rsplit("---[", 1)[1]

    items = []
    for line in data.splitlines():
        if ":args=" not in line:
            continue
        preargs, args = line.split(":args=", 1)
        pieces = ("kind=" + preargs).split(":") + ["args=" + args]
        item = dict(piece.split("=", 1) for piece in pieces)
        item["timing"] = eval(item.get("timing", "0"))
        items.append(item)

    return items


def get_syncs(items):
    # dels are asynchronous, assume others are synchronous
    dels = {
        i["seq"]
        for i in items
        if i["kind"] == "send" and i.get("req", "") == "HANDLE_DEL"
    }
    return [i for i in items if i["seq"] not in dels]


items = read_log((sys.argv[1:] or ["rpyc-trace.log"])[0])
syncs = get_syncs(items)

print(  # noqa: T001
    f"total time={sum(i['timing'] for i in syncs if i['kind'] == 'recv' and i['msg'] == 'MSG_REPLY')}"
)

longs = [i for i in syncs if i["timing"] > 0.5]
print(f'longs ({len(longs)}) time={sum(i["timing"] for i in longs)}')  # noqa: T001

s_sends = [i for i in syncs if i["kind"] == "send"]

buckets = collections.defaultdict(list)

for i in s_sends:
    buckets[i.get("req", "<REVERSE>")].append(i["args"])

print("-------------------")  # noqa: T001
for k, v in buckets.items():
    print(f"{k}={len(v)}")  # noqa: T001
print("-------------------")  # noqa: T001

sends = {
    i["seq"]: i for i in items if i["kind"] == "send" and i["msg"] == "MSG_REQUEST"
}
pairs, responses = [], {}
for i in items:
    if i["kind"] == "recv" and i["msg"] == "MSG_REPLY":
        try:
            pairs.append((sends[i["seq"]], i))
            responses[i["seq"]] = i
        except KeyError:
            pass
getattrs = [p for p in pairs if p[0].get("req", "") == "HANDLE_GETATTR"]


def _unbox(package):  # boxing
    label, value = package
    if label == consts.LABEL_VALUE:
        return value
    if label == consts.LABEL_TUPLE:
        return tuple(_unbox(item) for item in value)
    if label == consts.LABEL_LOCAL_REF:
        id_pack = (str(value[0]), value[1], value[2])  # so value is a id_pack
        return f"[local]{id_pack[0]}(cls={id_pack[1]}:inst={id_pack[2]})"
    if label == consts.LABEL_REMOTE_REF:
        id_pack = (str(value[0]), value[1], value[2])  # so value is a id_pack
        return f"[remote]{id_pack[0]}(cls={id_pack[1]}:inst={id_pack[2]})"
    raise ValueError("invalid label %r" % (label,))


def from_getattr_send(i, s=True):
    _, args = eval(i["args"])
    obj, attr = _unbox(args)
    return f"{obj}::{attr}" if s else (obj, attr)


def from_getattr_recv(i, s=True):
    if not i:
        return ""
    args = eval(i["args"])
    return _unbox(args)


def from_hash_send(i, s=True):
    _, args = eval(i["args"])
    obj = _unbox(args)[0]
    return obj


def _unwrap_obj(obj, remote):
    try:
        obj, attr = remote[obj.replace("[local]", "[remote]")]
    except (KeyError, ValueError):
        obj = "[>_<] " + obj
    else:
        obj = f"{obj.replace('[local]', '[remote]')}.{attr}"
    return obj


def _stringify(obj):
    if not isinstance(obj, str):
        return str(obj)
    if "[local]" in obj or "[remote]" in obj:
        return obj
    return repr(obj)


def _format_args(args, kw):
    fargs = ", ".join(_stringify(x) for x in args)
    fkw = ", ".join(f"{k}={_stringify(v)}" for (k, v) in kw)
    if fargs and fkw:
        fargs += ", "
    return f"({fargs}{fkw})"


def from_callattr_send(i, s=True, remote=None):
    _, args = eval(i["args"])
    obj, name, args, kw = _unbox(args)
    if remote:
        obj = _unwrap_obj(obj, remote)
    return f"{obj}.{name}{_format_args(args, kw)}" if s else (obj, name, args, kw)


def from_call_send(i, s=True, remote=None):
    _, args = eval(i["args"])
    obj, args, kw = _unbox(args)
    if remote:
        obj = _unwrap_obj(obj, remote)
    if s:
        res = f"{obj}{_format_args(args, kw)}"
        return re.sub(r"\(cls=\d+:inst=", "(inst:", res)
    return obj, args, kw


def _parse_msg(m, s=False, **kw):
    if m["kind"] == "send":
        if m.get("req") == "HANDLE_GETATTR":
            return from_getattr_send(m, s, **kw)
        if m.get("req") in ("HANDLE_HASH", "HANDLE_STR"):
            return from_hash_send(m, s, **kw)
        if m.get("req") == "HANDLE_CALLATTR":
            return from_callattr_send(m, s, **kw)
        if m.get("req") == "HANDLE_CALL":
            return from_call_send(m, s, **kw)
        return str(m)
    return from_getattr_recv(m, s, **kw)


remote = {}
for gsend, grecv in pairs:
    got, sent = _parse_msg(grecv, False), _parse_msg(gsend, False)
    if isinstance(got, str):
        remote[got] = sent
    # remote[from_getattr_recv(grecv, False)] = from_getattr_send(gsend, False)

print(f"total time getattrs={sum(x[1]['timing'] for x in getattrs)}")  # noqa: T001

# import pdb; pdb.set_trace()

print("\n\n----[ getattr ]----")  # noqa: T001
for gsend, grecv in getattrs:
    print(f"{from_getattr_send(gsend)} --> {from_getattr_recv(grecv)}")  # noqa: T001


print("\n\n----[ hash ]----")  # noqa: T001
for i in syncs:
    if i.get("req", "") == "HANDLE_HASH" and i["kind"] == "send":
        print(  # noqa: T001
            from_hash_send(i), "-->", from_getattr_recv(responses.get(i["seq"]))
        )

print("\n\n----[ str ]----")  # noqa: T001
for i in syncs:
    if i.get("req", "") == "HANDLE_STR" and i["kind"] == "send":
        print(  # noqa: T001
            from_hash_send(i), "-->", from_getattr_recv(responses.get(i["seq"]))
        )

print("\n\n----[ callattr ]----")  # noqa: T001
for i in syncs:
    if i.get("req", "") == "HANDLE_CALLATTR" and i["kind"] == "send":
        print(  # noqa: T001
            from_callattr_send(i, remote=remote),
            "-->",
            from_getattr_recv(responses.get(i["seq"])),
        )

print("\n\n----[ call ]----")  # noqa: T001
for i in syncs:
    if i.get("req", "") == "HANDLE_CALL" and i["kind"] == "send":
        print(  # noqa: T001
            from_call_send(i, remote=remote),
            "-->",
            from_getattr_recv(responses.get(i["seq"])),
        )
