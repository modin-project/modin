## High-Level Data Import Operation Workflow

Note: in this text for convenience was considered `read_csv` function with Pandas backend and Ray engine, for another import functions, workflow and classes/functions naming convension will be the same.

After user calls high-level `modin.read_csv` function, call is forwarded to the `EngineDispatcher`, which defines which factory from `modin\data_management\factories\factories` and backend/engine specific IO class should be used. If we take Ray engine and Pandas backend - IO class will be named `PandasOnRayIO`. This class defines modin frame and query compiler classes and `read_*` functions from: `RayTask` - class for managing remote tasks, `PandasCSVParser` - class for data parsing on the workers and `CSVDispatcher` - class for files handling of exact file format on the head node.

## Dispatcher Classes Workflow Overview
Call from `read_csv` function of `PandasOnRayIO` is forwarded to the `_read` function of `CSVDispatcher` class, where function parameters are preprocessed to check if they are supported (if they are not supported default pandas implementation is used) and get some common for all partitions metadata. Then file is splitted into chunks (mechanism of splitting is described below) and using this data, tasks are launched on the remote workers. After remote tasks are done, some additional results postprocessing is performed, new query compiler with imported data will be returned.

## Data File Splitting Mechanism

## Modules Description

This module is used for storing utils and dispatcher classes for reading files of different formats.

* io.py  
  Module houses `BaseIO` class, that contains basic utils and default implementation of IO functions.
* file_dispatcher.py  
  Module houses `FileDispatcher` class, that is used for reading data from different kinds of files and handling some common for all files formats util functions. Also this class contains `read` function which is entry point function for all dispatchers `_read` functions.
* text is directory for storing all text file format dispatcher classes
  * text_file_dispatcher.py





