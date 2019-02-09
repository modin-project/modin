"""
A convient python script module to generate Makefile

@author Simon Mo <xmo@berkeley.edu>
"""

import re
from collections import defaultdict, namedtuple
from distutils.version import LooseVersion
from functools import partial
import argparse

# Global action registry, this will be used to generate the final Makefile
global_registry = {}

# Prelude to print in makefile
PRELUDE = ["SHELL=/bin/bash -o pipefail"]


class Action(object):
    """Action object represent a single Makefile target

    Two actions can be related by setting up dependencies:
    ```
        a1 > a2
    ```
    will translate to makefile, meaning a2 depends on a1
    ```
        a2: a1
    ```

    Action can also be used to register group(s) it belongs to:
    ```
        a3 = Action('a3', ...,  tags=['group1'])
        a4 = Action('a4', ..., tags='group1')
    ```
    will translate to makefile:
    ```
    group1: a3 a4
        # empty target
    ```

    All actions have a default target 'all' except IsolatedAction. 
    """

    def __init__(self, name, command="", tags=None):
        self.name = name
        self.command = command
        self.tags = tags

        if not self.tags:
            self.tags = ["all"]
        elif isinstance(self.tags, str):
            self.tags = ["all", self.tags]
        else:
            self.tags = ["all"] + list(self.tags)

        self.dependencies = []

        global global_registry
        global_registry[name] = self

        # This hook will be used to process the command before it
        # gets printed out to Makfile.
        self.post_processing_hooks = [self._sanitize_command]

    @classmethod
    def get_action(cls, name):
        return global_registry[name]

    @classmethod
    def get_all_action(cls):
        return global_registry.values()

    def add_tag(self, tag):
        self.tags.append(tag)

    def _sanitize_command(self):
        self.command = self.command.replace("\n", "\n\t")

    def __str__(self):
        [hook() for hook in reversed(self.post_processing_hooks)]

        return """
{0}: {1}
\t{2}
        """.format(self.name, " ".join(self.dependencies), self.command)

    def __lt__(self, action):
        self.dependencies.append(action.name)

    def __eq__(self, value):
        return self.name == value.name and self.command == value.command


class IsolatedAction(Action):
    """Action that does not belong to group 'all'"""

    def __init__(self, name, command="", tags=None):
        super().__init__(name, command, tags)
        self.post_processing_hooks.append(self._not_included_in_all)

    def _not_included_in_all(self):
        self.tags.remove("all")


class CIPrettyLogAction(Action):
    """Action which the command output will be tagged by the name and colored.
    In addition, a header and footer will be added so the final output will look
    like this:
    ```
    ===== start: target_name ======
    [target_name] color_log_line
    ===== finished: target_name =====
    ```
    """

    def __init__(self, name, command="", tags=None):
        super().__init__(name, command, tags)
        self.post_processing_hooks.append(self._colorize_output)

    def _colorize_output(self):
        whitespace = re.compile("^[\s]*$")

        header = "=" * 5 + " start: {} ".format(self.name) + "=" * 5
        footer = "=" * 5 + " finished: {} ".format(self.name) + "=" * 5

        self.command = "\n".join(
            ["\t@echo {header}\n"]
            + [
                "\t({line}) 2>&1 | python3 ./jenkins/utils/colorize_output.py --tag {name}\n".format(line=line, name=self.name)
                for line in self.command.split("\n")
                if not whitespace.match(line)
            ]
            + ["\t@echo {footer}\n"]
        ).format(**locals())


def print_make_all():
    """Global function to printout the makefile"""

    # register all the tags or groups.
    # for example, group 'all'
    tag_to_action_name = defaultdict(list)
    for action in global_registry.values():
        print(action)
        for t in action.tags:
            tag_to_action_name[t].append(action.name)

    for tag, actions in tag_to_action_name.items():
        print(
            """
{}: {}
""".format(tag, ''.join(actions))
        )


def generate_make_file(config):
    # prevent ppl to make directly
    IsolatedAction("placeholder", 'echo "Do not run make without any target!"')

    with open(config) as f:
        exec(f.read(), globals())

    for l in PRELUDE:
        print(l)

    print_make_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Makefile from config.")
    parser.add_argument("config", type=str, help="path of the config file")
    args = parser.parse_args()

    generate_make_file(args.config)
