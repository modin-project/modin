from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from modin.experimental.scripts import cluster


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
    default="8890",
    help="port to which to forward the notebook server")
def notebook(config, port):
    config = cluster.load_config(config)
    cluster.validate_config(config)
    execution_framework = config["execution_framework"]
    if execution_framework == "ray":
        cluster.setup_notebook_ray(config, port)
    else:
        raise NotImplementedError(
            "Execution framework '{0}' not supported yet".format(
                execution_framework))


cli.add_command(notebook)


def main():
    return cli()


if __name__ == "__main__":
    main()
