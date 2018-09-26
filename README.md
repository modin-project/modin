<h1 align="center">Modin</h1>
<h3 align="center">Scale your pandas workflows by changing one line of code</h3>

<p align="center">
<a href="https://travis-ci.com/modin-project/modin"><img alt="" src="https://travis-ci.com/modin-project/modin.svg?branch=master"></a>
<a href="https://modin.readthedocs.io/en/latest/?badge=latest"><img alt="" src="https://readthedocs.org/projects/modin/badge/?version=latest"></a>
<a href="https://badge.fury.io/py/modin"><img alt="" src="https://badge.fury.io/py/modin.svg"></a>
<a href="https://github.com/ambv/black"><img alt="" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<p align="center"><b>To use Modin, replace the pandas import:</b></p>

```python
# import pandas as pd
import modin.pandas as pd
```

### Installation

Modin can be installed from PyPI:

```bash
pip install modin
```

### `modin.pandas`: Scale your pandas workflow by changing a single line of code.

Modin uses [Ray](https://github.com/ray-project/ray/) to provide an effortless way to 
speeding up your pandas notebooks, scripts, and libraries. Unlike other distributed
DataFrame libraries, **Modin** provides seamless integration and compatibility with
existing pandas code. Even using the DataFrame constructor is identical.

```python
import modin.pandas as pd
import numpy as np

frame_data = np.random.randint(0, 100, size=(2**10, 2**8))
df = pd.DataFrame(frame_data).add_prefix("col_")
```

To use Modin, you not need to know how many cores your system has, nor do you need to 
specify how to distribute the data. In fact, you can continue using your previous pandas
notebooks while experiencing a considerable speedup from Modin, even on a single
machine. Once you’ve changed your import statement, you’re ready to use Modin just like
you would pandas.

#### Faster pandas, even on your laptop

<img align="right" style="display:inline;" height="320" width="275" src="docs/img/read_csv_benchmark.png"></a>

The `modin.pandas` DataFrame is an extremely light-weight parallel DataFrame. Modin 
transparently distributes the data and computation so that all you need to do is
continue using the pandas API as you were before installing Modin. Because it is so
light-weight, Modin can provide speed-ups of up to 4x on a laptop with 4 physical cores.

In pandas, you are only able to use one core at a time when you are doing computation of
any kind. In `modin.pandas`, you are using all of the CPU cores on your machine.

#### Modin is a DataFrame for datasets from 1KB to 1TB+ 

We have focused heavily on bridging the solutions between DataFrames for small data 
(e.g. pandas) and lar

**`modin.pandas` is currently under active development. Requests and contributions are welcome!**

### More information and Getting Involved

- [Documentation](https://modin.readthedocs.io/en/latest/)
- Ask questions on our mailing list [modin-dev@googlegroups.com](https://groups.google.com/forum/#!forum/modin-dev).
- Submit bug reports to our [GitHub Issues Page](https://github.com/modin-project/modin/issues).
- Contributions are welcome! Open a [pull request](https://github.com/modin-project/modin/pulls).
