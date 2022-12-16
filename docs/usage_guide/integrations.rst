Third Party Library Integrations
================================

Our goal is that Modin is a drop-in replacement for Pandas — i.e., it won't break your workflow when you try to use Modin 
with a third-party library that works with Pandas. In the table we have tested select APIs from several libraries and taken note of each one's success metric
(the number of successful test calls / total test calls) and compared the overall interoperability between Pandas and Modin. 

In the deeper dive, you can view the Jupyter notebook we have used to test API calls and the corresponding Github issues filed. If you come across other issues/ examples 
in your own workflows we encourage you to file an `issue <https://github.com/modin-project/modin/issues/new/choose>`_ or contribute a `PR <https://github.com/modin-project/modin/pulls>`_!


.. note::
    These interoperability metrics are preliminary and not all APIs for each library have been tested.


Modin Interoperability by Library
''''''''''''''''''''''''''''''''''''
.. list-table::
   :widths: 5 5 20
   :header-rows: 1

   * - Library
     - API successes / calls
     - Interoperability
     
   * - seaborn
     - 73% (11/15)
     - **Pandas**: Accepts Pandas DataFrames as inputs for producing plot |br|
       **Modin**: Mostly accepts Modin DataFrames as inputs for producing plots, but fails in some cases (pairplot, lmplot), and for others (catplot, objects.Plot) works for some parameter combinations  but fails with others

   * - plotly
     - 78% (7 / 9)
     - **Pandas**: Accepts Pandas DataFrames as inputs for producing plots, including specifying X and Y parameters as df columns |br|
       **Modin**: Mostly accepts Modin DataFrames as inputs for producing plots (the exception is choropleth), but fails when specifying X and Y parameters as df columns
   
   * - matplotlib
     - 100% (5 / 5)
     - **Pandas**: Accepts Pandas DataFrames as inputs for producing plots like scatter, barh, etc. |br|
       **Modin**: Accepts Modin DataFrames as inputs for producing plots like scatter, barh, etc.
  
   * - altair
     - 0% (0 / 1)
     - **Pandas**: Accepts Pandas DataFrames as inputs for producing charts through Chart |br|
       **Modin**: Does not accept Modin DataFrames as inputs for producing charts through Chart

   * - bokeh
     - 0% (0 / 1)
     - **Pandas**: Loads Pandas DataFrames through ColumnDataSource |br|
       **Modin**: Does not load Modin DataFrames through ColumnDataSource
     
   * - sklearn
     - 100% (6 / 6)
     - **Pandas**: Many functions take Pandas DataFrames as inputs |br|
       **Modin**: Many functions take Modin DataFrames as inputs
    
   * - Hugging Face (Transformers, Datasets)
     - 100% (2 / 2) 
     - **Pandas**: Loads Pandas DataFrames into Datasets, and processes Pandas DataFrame rows as inputs using Transformers.InputExample (deprecated) |br|
       **Modin**: Loads Modin DataFrames into Datasets (though slowly), and processes Modin DataFrame rows as inputs through Transformers.InputExample (deprecated)
     
   * - Tensorflow
     - 75% (3 / 4)
     - **Panda**s: Converts Pandas dataframes to tensors |br|
       **Modin**: Converts Modin DataFrames to tensors, but specialized APIs like Keras might not work yet
     
   * - NLTK
     - 100% (1 / 1)
     - **Pandas**: Performs transformations like tokenization on Pandas DataFrames |br|
       **Modin**: Performs transformations like tokenization on Modin DataFrames
    
   * - XGBoost
     - 100% (1 / 1)
     - **Pandas**: Loads Pandas DataFrames through the DMatrix function |br|
       **Modin**: Loads Modin DataFrames through the DMatrix function
    
   * - statsmodels
     - 50% (1 / 2)
     - **Pandas**: Can accept Pandas DataFrames when fitting models |br|
       **Modin**: Sometimes accepts Modin DataFrames when fitting models (e.g., formula.api.ols), but does not in others (e.g., api.OLS)
     
.. |br| raw:: html

     <br>

A Deeper Dive
''''''''''''''

**seaborn**
-----------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5435 
    * https://github.com/modin-project/modin/issues/5433

**plotly**
-----------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5447 
    * https://github.com/modin-project/modin/issues/5445

**altair**
----------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5438

**bokeh**
---------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5437

**Tensorflow**
--------------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5439

**statsmodels**
---------------

Jupyter Notebook

Github Issues
    * https://github.com/modin-project/modin/issues/5440

Appendix: System Information
'''''''''''''''''''''''''''''
The example scripts here were run on the following system:

- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**: macOS Big Sur 11.5.2
- **Modin version**: 0.18.0+3.g4114183f
- **Ray version**: 2.0.1
- **Python version**: 3.9.7.final.0
- **Machine**: MacBook Pro (16-inch, 2019)
- **Processor**: 2.3 GHz 8-core Intel Core i9 processor
- **Memory**: 16 GB 2667 MHz DDR4
