from __future__ import absolute_import
from __future__ import print_function

import click

from modin.scripts import cluster


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--config",
    required=True,
    type=str,
    help="the config file for the cluster")
@click.option(
    "--port",
    required=True,
    help="port to which to forward the notebook server")
def notebook(config, port):
    config = cluster.load_config(config)
    cluster.validate_config(config)
    print("\nSetting up cluster\n")
    redis_address = cluster.setup_cluster(config)
    print("\nLaunching notebook\n")
    print("*" * 68)
    print(("To connect to the cluster, run the following commands in the "
           "notebook:\n"
           "\t\timport ray\n"
           "\t\tray.init(redis_address=\"{0}\")\n"
           "\t\timport modin").format(redis_address))
    print("*" * 68)

    cluster.launch_notebook(config, port)


cli.add_command(notebook)


def main():
    return cli()


if __name__ == "__main__":
    main()
