Third Party Library Integrations
================================

Summary
-------
To benchmark a single Modin function, often turning on the
:doc:`configuration variable </flow/modin/config>` variable
:code:`BenchmarkMode` will suffice.

.. note::
    All examples in this doc use the system specified at the bottom of this page.


Modin's execution and benchmark mode
''''''''''''''''''''''''''''''''''''
.. list-table::
   :widths: 5 5 5 5 5 5
   :header-rows: 1

   * - Library
     - API successes / calls
     - Pandas Interoperability
     - Modin Interoperability
     - Jupyter Notebooks
     - Github Issues
   * - seaborn
     - 73% (11/15)
     - Accepts Pandas DataFrames as inputs for producing plot
     - Mostly accepts Modin DataFrames as inputs for producing plots, but fails in some cases (pairplot, lmplot), and for others (catplot, objects.Plot) works for some parameter combinations  but fails with others
     - 
     - https://github.com/modin-project/modin/issues/5435 https://github.com/modin-project/modin/issues/5433

   * - plotly
     - 78% (7 / 9)
     - Accepts Pandas DataFrames as inputs for producing plots, including specifying X and Y parameters as df columns
     - Mostly accepts Modin DataFrames as inputs for producing plots (the exception is choropleth), but fails when specifying X and Y parameters as df columns
     - 
     - https://github.com/modin-project/modin/issues/5447 https://github.com/modin-project/modin/issues/5445
   * - matplotlib
     - 100% (5 / 5)
     - Accepts Pandas DataFrames as inputs for producing plots like scatter, barh, etc.
     - Accepts Modin DataFrames as inputs for producing plots like scatter, barh, etc.
     - 
     - 
   * - altair
     - 0% (0 / 1)
     - Accepts Pandas DataFrames as inputs for producing charts through Chart
     - Does not accept Modin DataFrames as inputs for producing charts through Chart
     - 
     - https://github.com/modin-project/modin/issues/5438
   * - bokeh
     - 0% (0 / 1)
     - Loads Pandas DataFrames through ColumnDataSource
     - Does not load Modin DataFrames through ColumnDataSource
     - 
     - https://github.com/modin-project/modin/issues/5437
   * - sklearn
     - 100% (6 / 6)
     - Many functions take Pandas DataFrames as inputs
     - Many functions take Modin DataFrames as inputs
     - 
     - 
   * - Hugging Face (Transformers, Datasets)
     - 100% (2 / 2) 
     - Loads Pandas DataFrames into Datasets, and processes Pandas DataFrame rows as inputs using Transformers.InputExample (deprecated)
     - Loads Modin DataFrames into Datasets (though slowly), and processes Modin DataFrame rows as inputs through Transformers.InputExample (deprecated)
     - 
     - 
   * - Tensorflow
     - 75% (3 / 4)
     - Converts Pandas dataframes to tensors
     - Converts Modin DataFrames to tensors, but specialized APIs like Keras might not work yet
     - 
     - https://github.com/modin-project/modin/issues/5439
   * - NLTK
     - 100% (1 / 1)
     - Performs transformations like tokenization on Pandas DataFrames 
     - Performs transformations like tokenization on Modin DataFrames
     - 
     - 
   * - XGBoost
     - 100% (1 / 1)
     - Loads Pandas DataFrames through the DMatrix function
     - Loads Modin DataFrames through the DMatrix function
     - 
     - 
   * - statsmodels
     - 50% (1 / 2)
     - Can accept Pandas DataFrames when fitting models
     - Sometimes accepts Modin DataFrames when fitting models (e.g., formula.api.ols), but does not in others (e.g., api.OLS)
     - 
     - https://github.com/modin-project/modin/issues/5440




Appendix: System Information
----------------------------
The example scripts here were run on the following system:

- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**: macOS Monterey 12.4
- **Modin version**: d6d503ac7c3028d871c34d9e99e925ddb0746df6
- **Ray version**: 2.0.0
- **Python version**: 3.10.4
- **Machine**: MacBook Pro (16-inch, 2019)
- **Processor**: 2.3 GHz 8-core Intel Core i9 processor
- **Memory**: 16 GB 2667 MHz DDR4
