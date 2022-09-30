<p align="center"><a href="https://modin.readthedocs.io"><img width=77% alt="" src="https://github.com/modin-project/modin/blob/3d6368edf311995ad231ec5342a51cd9e4e3dc20/docs/img/MODIN_ver2_hrz.png?raw=true"></a></p>
<h2 align="center">Scale your pandas workflows by changing one line of code</h2>

<div align="center">

| <h3>Dev Community & Support</h3> | <h3>Forums</h3> | <h3>Socials</h3> | <h3>Docs</h3> |
|:---: | :---: | :---: | :---: |
| [![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/modin-project/shared_invite/zt-yvk5hr3b-f08p_ulbuRWsAfg9rMY3uA) | [![Stack Overflow](https://img.shields.io/badge/-Stackoverflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/questions/tagged/modin) | <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/modin_project?style=social" height=28 align="center"> | <a href="https://modin.readthedocs.io/en/latest/?badge=latest"><img alt="" src="https://readthedocs.org/projects/modin/badge/?version=latest" height=28 align="center"></a> |

</div>

<p align="center">
<a href="https://pepy.tech/project/modin"><img src="https://static.pepy.tech/personalized-badge/modin?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads" align="center"></a>
<a href="https://codecov.io/gh/modin-project/modin"><img src="https://codecov.io/gh/modin-project/modin/branch/master/graph/badge.svg" align="center"/></a>
<a href="https://github.com/modin-project/modin/actions"><img src="https://github.com/modin-project/modin/workflows/master/badge.svg" align="center"></a>
<a href="https://pypi.org/project/modin/"><img src="https://badge.fury.io/py/modin.svg" alt="PyPI version" align="center"></a>
<a href="https://modin.org/modin-bench/#/"><img src="https://img.shields.io/badge/benchmarked%20by-asv-blue.svg" align="center"></a>
</p>

### What is Modin?

Modin is a drop-in replacement for [pandas](https://github.com/pandas-dev/pandas). While pandas is
single-threaded, Modin lets you instantly speed up your workflows by scaling pandas so it uses all of your
cores. Modin works especially well on larger datasets, where pandas becomes painfully slow or runs
[out of memory](https://modin.readthedocs.io/en/latest/getting_started/why_modin/out_of_core.html).

By simply replacing the import statement, Modin offers users effortless speed and scale for their pandas workflows:

<img src="https://github.com/modin-project/modin/blob/master/docs/img/Import.gif" style="display: block;margin-left: auto;margin-right: auto;" width="100%"></img>

In the GIFs below, Modin (left) and pandas (right) perform *the same pandas operations* on a 2GB dataset. The only difference between the two notebook examples is the import statement. 

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax" style="text-align: center;"><img src="https://github.com/modin-project/modin/blob/master/docs/img/MODIN_ver2_hrz.png?raw=True" height="35px"></th>
    <th class="tg-0lax" style="text-align: center;"><img src="https://pandas.pydata.org/static/img/pandas.svg" height="50px"></img></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax"><img src="https://github.com/modin-project/modin/blob/master/docs/img/Modin.gif"></img></td>
    <td class="tg-0lax"><img src="https://github.com/modin-project/modin/blob/master/docs/img/Pandas.gif"></img></td>
  </tr>
</tbody>
</table>

The charts below show the speedup you get by replacing pandas with Modin based on the examples above. The example notebooks can be found [here](examples/jupyter). To learn more about the speedups you could get with Modin and try out some examples on your own, check out our [10-minute quickstart guide](https://modin.readthedocs.io/en/latest/getting_started/quickstart.html) to try out some examples on your own!

<img src="https://github.com/modin-project/modin/blob/master/docs/img/Modin_Speedup.svg" style="display: block;margin-left: auto;margin-right: auto;" width="100%"></img>

### Installation

#### From PyPI

Modin can be installed with `pip` on Linux, Windows and MacOS:

```bash
pip install modin[all] # (Recommended) Install Modin with all of Modin's currently supported engines.
```

If you want to install Modin with a specific engine, we recommend:

```bash
pip install modin[ray] # Install Modin dependencies and Ray.
pip install modin[dask] # Install Modin dependencies and Dask.
```

Modin automatically detects which engine(s) you have installed and uses that for scheduling computation.

#### From conda-forge

Installing from [conda forge](https://github.com/conda-forge/modin-feedstock) using `modin-all`
will install Modin and three engines: [Ray](https://github.com/ray-project/ray),
[Dask](https://github.com/dask/dask), and [HDK](https://github.com/intel-ai/hdk).

```bash
conda install -c conda-forge modin-all
```

Each engine can also be installed individually:

```bash
conda install -c conda-forge modin-ray  # Install Modin dependencies and Ray.
conda install -c conda-forge modin-dask # Install Modin dependencies and Dask.
conda install -c conda-forge modin-hdk # Install Modin dependencies and HDK.
```

#### Choosing a Compute Engine

If you want to choose a specific compute engine to run on, you can set the environment
variable `MODIN_ENGINE` and Modin will do computation with that engine:

```bash
export MODIN_ENGINE=ray  # Modin will use Ray
export MODIN_ENGINE=dask  # Modin will use Dask
```

This can also be done within a notebook/interpreter before you import Modin:

```python
from modin.config import Engine

Engine.put("ray")  # Modin will use Ray
Engine.put("dask")  # Modin will use Dask
```

Check [this Modin docs section](https://modin.readthedocs.io/en/latest/development/using_hdk.html) for HDK engine setup.

_Note: You should not change the engine after your first operation with Modin as it will result in undefined behavior._

#### Which engine should I use?

On Linux, MacOS, and Windows you can install and use either Ray or Dask. There is no knowledge required
to use either of these engines as Modin abstracts away all of the complexity, so feel
free to pick either!

On Linux you also can choose [HDK](https://modin.readthedocs.io/en/latest/development/using_hdk.html), which is an experimental
engine based on [HDK](https://github.com/intel-ai/hdk) and included in the
[Intel® Distribution of Modin](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-of-modin.html),
which is a part of [Intel® oneAPI AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html).

### Pandas API Coverage

<p align="center">

| pandas Object     | Modin's Ray Engine Coverage                                                          | Modin's Dask Engine Coverage |
|-------------------|:------------------------------------------------------------------------------------:|:---------------:|
| `pd.DataFrame`    | <img src=https://img.shields.io/badge/api%20coverage-90.8%25-hunter.svg> | <img src=https://img.shields.io/badge/api%20coverage-90.8%25-hunter.svg> |
| `pd.Series`       | <img src=https://img.shields.io/badge/api%20coverage-88.05%25-green.svg> | <img src=https://img.shields.io/badge/api%20coverage-88.05%25-green.svg> |
| `pd.read_csv`     | ✅                                               | ✅ |
| `pd.read_table`   | ✅                                               | ✅ |
| `pd.read_parquet` | ✅                                               | ✅ |
| `pd.read_sql`     | ✅                                               | ✅ |
| `pd.read_feather` | ✅                                               | ✅ |
| `pd.read_excel`   | ✅                                               | ✅ |
| `pd.read_json`    | [✳️](https://github.com/modin-project/modin/issues/554)                                         | [✳️](https://github.com/modin-project/modin/issues/554) |
| `pd.read_<other>` | [✴️](https://modin.readthedocs.io/en/latest/supported_apis/io_supported.html) | [✴️](https://modin.readthedocs.io/en/latest/supported_apis/io_supported.html) |

</p>
Some pandas APIs are easier to implement than others, so if something is missing feel
free to open an issue!

### More about Modin

For the complete documentation on Modin, visit our [ReadTheDocs](https://modin.readthedocs.io/en/latest/index.html) page.

#### Scale your pandas workflow by changing a single line of code.

_Note: In local mode (without a cluster), Modin will create and manage a local (Dask or Ray) cluster for the execution._

To use Modin, you do not need to specify how to distribute the data, or even know how many
cores your system has. In fact, you can continue using your previous
pandas notebooks while experiencing a considerable speedup from Modin, even on a single
machine. Once you've changed your import statement, you're ready to use Modin just like
you would with pandas!

#### Faster pandas, even on your laptop

<img align="right" style="display:inline;" height="350" width="300" src="https://github.com/modin-project/modin/blob/master/docs/img/read_csv_benchmark.png?raw=true"></a>

The `modin.pandas` DataFrame is an extremely light-weight parallel DataFrame.
Modin transparently distributes the data and computation so that you can continue using the same pandas API
while working with more data faster. Because it is so light-weight,
Modin provides speed-ups of up to 4x on a laptop with 4 physical cores.

In pandas, you are only able to use one core at a time when you are doing computation of
any kind. With Modin, you are able to use all of the CPU cores on your machine. Even with a
traditionally synchronous task like `read_csv`, we see large speedups by efficiently
distributing the work across your entire machine.

```python
import modin.pandas as pd

df = pd.read_csv("my_dataset.csv")
```

#### Modin can handle the datasets that pandas can't 

Often data scientists have to switch between different tools
for operating on datasets of different sizes. Processing large dataframes with pandas
is slow, and pandas does not support working with dataframes that are too large to fit
into the available memory. As a result, pandas workflows that work well
for prototyping on a few MBs of data do not scale to tens or hundreds of GBs (depending on the size
of your machine). Modin supports operating on data that does not fit in memory, so that you can comfortably
work with hundreds of GBs without worrying about substantial slowdown or memory errors.
With [cluster](https://modin.readthedocs.io/en/latest/getting_started/using_modin/using_modin_cluster.html)
and [out of core](https://modin.readthedocs.io/en/latest/getting_started/why_modin/out_of_core.html)
support, Modin is a DataFrame library with both great single-node performance and high
scalability in a cluster.

#### Modin Architecture

We designed [Modin's architecture](https://modin.readthedocs.io/en/latest/development/architecture.html)
to be modular so we can plug in different components as they develop and improve:

<img src="docs/img/modin_architecture.png" alt="Modin's architecture" width="75%"></img>

### Other Resources

#### Getting Started with Modin

- [Documentation](https://modin.readthedocs.io/en/latest/)
- [10-min Quickstart Guide](https://modin.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Examples and Tutorials](https://modin.readthedocs.io/en/latest/getting_started/examples.html)
- [Videos and Blogposts](https://modin.readthedocs.io/en/latest/getting_started/examples.html#talks-podcasts)
- [Benchmarking Modin](https://modin.readthedocs.io/en/latest/usage_guide/benchmarking.html)

#### Modin Community

- [Slack](https://join.slack.com/t/modin-project/shared_invite/zt-yvk5hr3b-f08p_ulbuRWsAfg9rMY3uA)
- [Discourse](https://discuss.modin.org)
- [Twitter](https://twitter.com/modin_project)
- [Mailing List](https://groups.google.com/g/modin-dev)
- [GitHub Issues](https://github.com/modin-project/modin/issues)
- [StackOverflow](https://stackoverflow.com/questions/tagged/modin)

#### Learn More about Modin

- [Frequently Asked Questions (FAQs)](https://modin.readthedocs.io/en/latest/getting_started/faq.html)
- [Troubleshooting Guide](https://modin.readthedocs.io/en/latest/getting_started/troubleshooting.html)
- [Development Guide](https://modin.readthedocs.io/en/latest/development/index.html)
- Modin is built on many years of research and development at UC Berkeley. Check out these selected papers to learn more about how Modin works:
  - [Flexible Rule-Based Decomposition and Metadata Independence in Modin](https://people.eecs.berkeley.edu/~totemtang/paper/Modin.pdf) (VLDB 2021)
  - [Dataframe Systems: Theory, Architecture, and Implementation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-193.pdf) (PhD Dissertation 2021)
  - [Towards Scalable Dataframe Systems](https://arxiv.org/pdf/2001.00888.pdf) (VLDB 2020)

#### Getting Involved

***`modin.pandas` is currently under active development. Requests and contributions are welcome!***

For more information on how to contribute to Modin, check out the
[Modin Contribution Guide](https://modin.readthedocs.io/en/latest/development/contributing.html).

### License

[Apache License 2.0](LICENSE)
