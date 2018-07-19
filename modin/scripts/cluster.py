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


def setup_head_node(config):
    """Sets up the head node given a valid configuration"""
    hostname = config["head_node"]["hostname"]
    key = config["head_node"].get("key") or config.get("key")
    if not key:
        raise ValueError("Missing key for head_node")

    output = subprocess.check_output(
            ["sh", resolve_script_path("configure_head_node.sh"), hostname,
             key])

    redis_address = subprocess.check_output(
            ["sh", resolve_script_path("get_redis_address.sh"), output])
    redis_address = redis_address.decode("ascii").strip()

    return redis_address


def setup_nodes(config, redis_address):
    """Sets up nodes given the config and the redis address"""
    try:
        from subprocess import DEVNULL
    except ImportError:
        import os
        DEVNULL = open(os.devnull, "wb")

    for node in config.get("nodes", []):
        hostname = node["hostname"]
        key = node.get("key") or config.get("key")
        if not key:
            raise ValueError("Missing key for node {0}".format(hostname))

        subprocess.Popen(
                ["sh", resolve_script_path("configure_node.sh"), hostname, key,
                 redis_address], stdout=DEVNULL, stderr=DEVNULL)


def setup_cluster(config):
    """Sets up a cluster given a valid configuration"""
    if config["execution_framework"] != "ray":
        raise NotImplementedError("Only Ray clusters supported for now")

    redis_address = setup_head_node(config)
    setup_nodes(config, redis_address)

    return redis_address


def launch_notebook(config, port, redis_address="", blocking=True):
    """SSH into the head node, launches a notebook, and forwards port"""
    exec_framework = config["execution_framework"]
    hostname = config["head_node"]["hostname"]
    key = config["head_node"].get("key") or config.get("key")
    if not key:
        raise ValueError("Missing key for head_node")

    if blocking:
        subprocess.call(
                ["sh", resolve_script_path("launch_notebook.sh"), hostname,
                 key, port, exec_framework, redis_address])
    else:
        subprocess.Popen(["sh", resolve_script_path("launch_notebook.sh"),
                          hostname, key, port, exec_framework, redis_address])
