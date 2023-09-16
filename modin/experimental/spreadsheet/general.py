# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from modin_spreadsheet import SpreadsheetWidget, show_grid

from .. import pandas as pd


def from_dataframe(
    dataframe,
    show_toolbar=None,
    show_history=None,
    precision=None,
    grid_options=None,
    column_options=None,
    column_definitions=None,
    row_edit_callback=None,
):
    """
    Renders a DataFrame or Series as an interactive spreadsheet, represented by
    an instance of the ``SpreadsheetWidget`` class.  The ``SpreadsheetWidget`` instance
    is constructed using the options passed in to this function.  The
    ``dataframe`` argument to this function is used as the ``df`` kwarg in
    call to the SpreadsheetWidget constructor, and the rest of the parameters
    are passed through as is.

    If the ``dataframe`` argument is a Series, it will be converted to a
    DataFrame before being passed in to the SpreadsheetWidget constructor as the
    ``df`` kwarg.

    :rtype: SpreadsheetWidget

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame that will be displayed by this instance of
        SpreadsheetWidget.
    grid_options : dict
        Options to use when creating the SlickGrid control (i.e. the
        interactive grid).  See the Notes section below for more information
        on the available options, as well as the default options that this
        widget uses.
    precision : integer
        The number of digits of precision to display for floating-point
        values.  If unset, we use the value of
        `pandas.get_option('display.precision')`.
    show_toolbar : bool
        Whether to show a toolbar with options for adding/removing rows.
        Adding/removing rows is an experimental feature which only works
        with DataFrames that have an integer index.
    show_history : bool
        Whether to show the cell containing the spreadsheet transformation
        history.
    column_options : dict
        Column options that are to be applied to every column. See the
        Notes section below for more information on the available options,
        as well as the default options that this widget uses.
    column_definitions : dict
        Column options that are to be applied to individual
        columns. The keys of the dict should be the column names, and each
        value should be the column options for a particular column,
        represented as a dict. The available options for each column are the
        same options that are available to be set for all columns via the
        ``column_options`` parameter. See the Notes section below for more
        information on those options.
    row_edit_callback : callable
        A callable that is called to determine whether a particular row
        should be editable or not. Its signature should be
        ``callable(row)``, where ``row`` is a dictionary which contains a
        particular row's values, keyed by column name. The callback should
        return True if the provided row should be editable, and False
        otherwise.


    Notes
    -----
    The following dictionary is used for ``grid_options`` if none are
    provided explicitly::

        {
            # SlickGrid options
            'fullWidthRows': True,
            'syncColumnCellResize': True,
            'forceFitColumns': False,
            'defaultColumnWidth': 150,
            'rowHeight': 28,
            'enableColumnReorder': False,
            'enableTextSelectionOnCells': True,
            'editable': True,
            'autoEdit': False,
            'explicitInitialization': True,

            # Modin-spreadsheet options
            'maxVisibleRows': 15,
            'minVisibleRows': 8,
            'sortable': True,
            'filterable': True,
            'highlightSelectedCell': False,
            'highlightSelectedRow': True
        }

    The first group of options are SlickGrid "grid options" which are
    described in the `SlickGrid documentation
    <https://github.com/mleibman/SlickGrid/wiki/Grid-Options>`__.

    The second group of option are options that were added specifically
    for modin-spreadsheet and therefore are not documented in the SlickGrid documentation.
    The following bullet points describe these options.

    * **maxVisibleRows** The maximum number of rows that modin-spreadsheet will show.
    * **minVisibleRows** The minimum number of rows that modin-spreadsheet will show
    * **sortable** Whether the modin-spreadsheet instance will allow the user to sort
      columns by clicking the column headers. When this is set to ``False``,
      nothing will happen when users click the column headers.
    * **filterable** Whether the modin-spreadsheet instance will allow the user to filter
      the grid. When this is set to ``False`` the filter icons won't be shown
      for any columns.
    * **highlightSelectedCell** If you set this to True, the selected cell
      will be given a light blue border.
    * **highlightSelectedRow** If you set this to False, the light blue
      background that's shown by default for selected rows will be hidden.

    The following dictionary is used for ``column_options`` if none are
    provided explicitly::

        {
            # SlickGrid column options
            'defaultSortAsc': True,
            'maxWidth': None,
            'minWidth': 30,
            'resizable': True,
            'sortable': True,
            'toolTip': "",
            'width': None

            # Modin-spreadsheet column options
            'editable': True,
        }

    The first group of options are SlickGrid "column options" which are
    described in the `SlickGrid documentation
    <https://github.com/mleibman/SlickGrid/wiki/Column-Options>`__.

    The ``editable`` option was added specifically for modin-spreadsheet and therefore is
    not documented in the SlickGrid documentation.  This option specifies
    whether a column should be editable or not.

    See Also
    --------
    set_defaults : Permanently set global defaults for the parameters
                   of ``show_grid``, with the exception of the ``dataframe``
                   and ``column_definitions`` parameters, since those
                   depend on the particular set of data being shown by an
                   instance, and therefore aren't parameters we would want
                   to set for all SpreadsheetWidget instances.
    set_grid_option : Permanently set global defaults for individual
                      grid options.  Does so by changing the defaults
                      that the ``show_grid`` method uses for the
                      ``grid_options`` parameter.
    SpreadsheetWidget : The widget class that is instantiated and returned by this
                  method.

    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be modin.DataFrame, not %s" % type(dataframe))
    return show_grid(
        dataframe,
        show_toolbar,
        show_history,
        precision,
        grid_options,
        column_options,
        column_definitions,
        row_edit_callback,
    )


def to_dataframe(spreadsheet):
    """
    Get a copy of the DataFrame that reflects the current state of the ``spreadsheet`` SpreadsheetWidget instance UI.
    This includes any sorting or filtering changes, as well as edits
    that have been made by double clicking cells.

    :rtype: DataFrame

    Parameters
    ----------
    spreadsheet : SpreadsheetWidget
        The SpreadsheetWidget instance that DataFrame that will be displayed by this instance of
        SpreadsheetWidget.
    """
    if not isinstance(spreadsheet, SpreadsheetWidget):
        raise TypeError(
            "spreadsheet must be modin_spreadsheet.SpreadsheetWidget, not %s"
            % type(spreadsheet)
        )
    return spreadsheet.get_changed_df()
