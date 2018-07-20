import os
import subprocess
import yaml


REQUIRED, OPTIONAL = True, False
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_CONFIG_SCHEMA = {
    # Execution engine for the cluster.
    # Possible options: ray.
    "execution_framework": (str, REQUIRED),

    # Default key used to SSH into nodes.
    "key": (str, OPTIONAL),

    # Head node on which notebooks may run
    "head_node": (
        {
            "hostname": (str, REQUIRED),
            "key": (str, OPTIONAL),     # overrides global key
        },
        REQUIRED),

    # Other nodes part of the cluster
    "nodes": (
        [
            {
                "hostname": (str, REQUIRED),
                "key": (str, OPTIONAL),     # overrides global key
            }
        ],
        OPTIONAL),
}


def typename(v):
    if isinstance(v, type):
        return v.__name__
    else:
        return type(v).__name__


def check_required(config, schema):
    """Check required config entries"""
    if type(config) is not dict and type(config) is not list:
        raise ValueError("Config is not a dictionary or a list")
    if type(config) != type(schema):
        raise ValueError("Config is a {0}, but schema is a {1}".format(
            typename(config), typename(schema)))
    if type(config) is list:
        if not len(config):
            return
        item_schema = schema[0]
        for item_config in config:
            check_required(item_config, item_schema)
    elif type(config) is dict:
        for k, (v, kreq) in schema.items():
            if v is None:
                continue
            if kreq is REQUIRED:
                if k not in config:
                    raise ValueError(
                        "Missing required config key {0} of type {1}".format(
                            k, typename(v)))
                if not isinstance(v, type):
                    check_required(config[k], v)


def check_extraneous(config, schema):
    """Check that all items in config are valid in schema"""
    if type(config) is not dict and type(config) is not list:
        raise ValueError("Config is not a dictionary or a list")
    if type(config) != type(schema):
        raise ValueError("Config is a {0}, but schema is a {1}".format(
            typename(config), typename(schema)))
    if type(config) is list:
        if not len(config):
            return
        item_schema = schema[0]
        for item_config in config:
            # Check required keys in the item's schema because check_required
            # does not navigate extraneous schema paths
            check_required(item_config, item_schema)
            check_extraneous(item_config, item_schema)
    elif type(config) is dict:
        for k in config:
            if k not in schema:
                raise ValueError(
                        "Unexpected config key {0} not in {1}".format(
                            k, list(schema.keys())))
            v, kreq = schema[k]
            if v is None:
                continue
            elif isinstance(v, type):
                if not isinstance(config[k], v):
                    raise ValueError(
                            "Expected {0} for config key {1}, but got {2}"
                            .format(typename(v), k, type(config[k]).__name__))
            else:
                check_extraneous(config[k], v)


def validate_config(config, schema=CLUSTER_CONFIG_SCHEMA):
    """Validates a configuration given a schema"""
    check_required(config, schema)
    check_extraneous(config, schema)


def load_config(filename):
    """Loads a YAML file"""
    with open(filename) as f:
        return yaml.load(f.read())


def resolve_script_path(script_basename):
    """Returns the filepath of the script"""
    return os.path.join(SCRIPTS_DIR, script_basename)


def add_key_to_ssh_agent(key_path):
    """Adds a key to SSH agent"""
    subprocess.call(["ssh-add", key_path])


def add_keys_to_ssh_agent(config):
    """Adds keys to SSH agent"""
    if "key" in config:
        add_key_to_ssh_agent(config["key"])
    if "key" in config["head_node"]:
        add_key_to_ssh_agent(config["head_node"]["key"])
    for node_config in config.get("nodes", []):
        if "key" in node_config:
            add_key_to_ssh_agent(node_config["key"])


def setup_notebook_ray(config, port):
    # Create nodes file
    with open("nodes.txt", "w") as f:
        for node_config in config.get("nodes", []):
            f.write(node_config["hostname"] + "\n")

    # Launch cluster
    subprocess.call(
            ["sh", resolve_script_path("configure_ray_cluster.sh"),
             config["head_node"]["hostname"], "nodes.txt", port])
