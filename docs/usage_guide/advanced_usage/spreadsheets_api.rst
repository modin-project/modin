Modin Spreadsheets API
======================

Getting started
---------------
Install Modin-spreadsheet using pip:

.. code-block:: bash

    pip install "modin[spreadsheet]"


The following code snippet creates a spreadsheet using the FiveThirtyEight dataset on labor force information by college majors (licensed under CC BY 4.0):

.. code-block:: python

    import modin.pandas as pd
    import modin.experimental.spreadsheet as mss
    df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/all-ages.csv')
    spreadsheet = mss.from_dataframe(df)
    spreadsheet


.. figure:: /img/modin_spreadsheets_installation.png
    :align: center

Basic Manipulations through User Interface
------------------------------------------

The Spreadsheet API allows users to manipulate the DataFrame with simple graphical controls for sorting, filtering, and editing. 

Here are the instructions for each operation:
    * **Sort**: Click on the column header of the column to sort on.
    * **Filter**: Click on the filter button on the column header and apply the desired filter to the column. The filter dropdown changes depending on the type of the column. Multiple filters are automatically combined.
    * **Edit Cell**: Double click on a cell and enter the new value.
    * **Add Rows**: Click on the “Add Row” button in the toolbar to duplicate the last row in the DataFrame. The duplicated values provide a convenient default and can be edited as necessary.
    * **Remove Rows**: Select row(s) and click the “Remove Row” button. Select a single row by clicking on it. Multiple rows can be selected with Cmd+Click (Windows: Ctrl+Click) on the desired rows or with Shift+Click to specify a range of rows. 

Some of these operations can also be done through the spreadsheet’s programmatic interface. Sorts and filters can be reset using the toolbar buttons. Edits and adding/removing rows can only be undone manually.

Virtual Rendering
-----------------

The spreadsheet will only render data based on the user’s viewport. This allows for quick rendering 
even on very large DataFrames because only a handful of rows are loaded at any given time. As a result, scrolling and viewing your data is smooth and responsive.

Transformation History and Exporting Code
-----------------------------------------

All operations on the spreadsheet are recorded and are easily exported as code for sharing or reproducibility. 
This history is automatically displayed in the history cell, which is generated below the spreadsheet whenever the spreadsheet widget is displayed. 
The history cell is displayed on default, but this can be turned off. Modin Spreadsheet API provides a few methods for interacting with the history:

    * `SpreadsheetWidget.get_history()`: Retrieves the transformation history in the form of reproducible code. 
    * `SpreadsheetWidget.filter_relevant_history(persist=True)`: Returns the transformation history that contains only code relevant to the final state of the spreadsheet. The `persist` parameter determines whether the internal state and the displayed history is also filtered.
    * `SpreadsheetWidget.reset_history()`: Clears the history of transformation.

Customizable Interface
----------------------

The spreadsheet widget provides a number of options that allows the user to change the appearance and the interactivity of the spreadsheet. Options include:

    * Row height/Column width
    * Preventing edits, sorts, or filters on the whole spreadsheet or on a per-column basis
    * Hiding the toolbar and history cell
    * Float precision
    * Highlighting of cells and rows
    * Viewport size

Converting Spreadsheets To and From Dataframes
----------------------------------------------

.. automodule:: modin.experimental.spreadsheet.general
    :noindex:
    :members: from_dataframe

    
.. automodule:: modin.experimental.spreadsheet.general
    :noindex:
    :members: to_dataframe


Further API Documentation
-------------------------

.. automodule:: modin_spreadsheet.grid
    :noindex:
    :members: SpreadsheetWidget