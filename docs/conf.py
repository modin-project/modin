# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Project information -----------------------------------------------------
import sys
import os
import types

import ray

# stub ray.remote to be a no-op so it doesn't shadow docstrings
def noop_decorator(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # This is the case where the decorator is just @ray.remote without parameters.
        return args[0]
    return lambda cls_or_func: cls_or_func


ray.remote = noop_decorator

# fake modules if they're missing
for mod_name in ("cudf", "cupy", "pyarrow.gandiva", "omniscidbe"):
    try:
        __import__(mod_name)
    except ImportError:
        sys.modules[mod_name] = types.ModuleType(
            mod_name, f"fake {mod_name} for building docs"
        )
if not hasattr(sys.modules["cudf"], "DataFrame"):
    sys.modules["cudf"].DataFrame = type("DataFrame", (object,), {})
if not hasattr(sys.modules["omniscidbe"], "PyDbEngine"):
    sys.modules["omniscidbe"].PyDbEngine = type("PyDbEngine", (object,), {})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import modin

from modin.config.__main__ import export_config_help

configs_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "flow/modin/configs_help.csv")
)
# Export configs help to create configs table in the docs/flow/modin/config.rst
export_config_help(configs_file_path)

project = "Modin"
copyright = "2018-2022, Modin Developers."
author = "Modin contributors"

# The short X.Y version
version = "{}".format(modin.__version__)
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.graphviz",
    'sphinx.ext.autosummary',
    "sphinxcontrib.plantuml",
    "sphinx_issues",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Turn on sphinx.ext.autosummary
autosummary_generate = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# Maps git branches to Sphinx themes
default_html_theme = "pydata_sphinx_theme"
current_branch = "nature"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_favicon = "img/MODIN_ver2.ico"

html_logo = "img/MODIN_ver2.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "sidebarwidth": 270,
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "github_url": "https://github.com/modin-project/modin",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/modin",
            "icon": "fab fa-python",
        },
        {
            "name": "conda-forge",
            "url": "https://anaconda.org/conda-forge/modin",
            "icon": "fas fa-circle-notch",
        },
        {
            "name": "Join the Slack",
            "url": "https://modin.org/slack.html",
            "icon": "fab fa-slack",
        },
        {
            "name": "Discourse",
            "url": "https://discuss.modin.org/",
            "icon": "fab fa-discourse",
        },
        {
            "name": "Mailing List",
            "url": "https://groups.google.com/forum/#!forum/modin-dev",
            "icon": "fas fa-envelope-square",
        },
    ],
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# The default pydata_sphinx_theme sidebar templates are
# sidebar-nav-bs.html and search-field.html.
html_sidebars = {}

issues_github_path = "modin-project/modin"
