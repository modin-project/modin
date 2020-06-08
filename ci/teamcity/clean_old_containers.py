import argparse
import docker
import datetime
from dateutil.parser import parse

containers_to_keep = ["nodelete"]
images_to_keep = ["modin-project/modin-base", "ray-project/", "ubuntu", "nodelete"]

parser = argparse.ArgumentParser(
    description="Remove docker containers and images older than specified number of days"
)
parser.add_argument("-days", default=4, type=int, help="Specify number of days")
args = parser.parse_args()

days_duration = datetime.timedelta(days=args.days)

client = docker.from_env()

for co in client.containers.list(all=True):
    started = parse(co.attrs["State"]["StartedAt"]).replace(tzinfo=None)
    name = co.attrs["Name"]
    if datetime.datetime.now() - started > days_duration and not any(
        [name.find(x) >= 0 for x in containers_to_keep]
    ):
        print("Removing container", co.attrs["Id"], name, ": ", end="")
        try:
            co.stop()
            co.remove(v=True, force=False)
            print("success")
        except Exception:
            print("Failed")

for im in client.images.list():
    created = parse(im.attrs["Created"]).replace(tzinfo=None)
    if datetime.datetime.now() - created > days_duration and not any(
        [any([x.find(y) >= 0 for x in im.attrs["RepoTags"]]) for y in images_to_keep]
    ):
        print("Removing image", im.attrs["Id"], im.attrs["RepoTags"], ": ", end="")
        try:
            client.images.remove(image=im.attrs["Id"], force=False)
            print("success")
        except Exception:
            print("Failed")
