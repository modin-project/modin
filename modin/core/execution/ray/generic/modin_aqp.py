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

"""The module for working with displaying progress bars for Ray engine."""

import ray
import os
import time
import threading
import warnings
import uuid

from modin.config.envvars import MicroPartitions
from datetime import datetime

progress_bars = {}
bar_log_ids = {}
bar_lock = threading.Lock()


def call_progress_bar(result_parts, line_no, func):
    """
    Attach a progress bar to given `result_parts`.

    The progress bar is expected to be shown in a Jupyter Notebook cell.

    Parameters
    ----------
    result_parts : list of list of ray.ObjectRef
        Objects which are being computed for which progress is requested.
    line_no : int
        Line number in the call stack which we're displaying progress for.
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     try:
    #         from tqdm.autonotebook import tqdm as tqdm_notebook
    #     except ImportError:
    #         raise ImportError("Please pip install tqdm to use the progress bar")
    # from IPython import get_ipython

    # try:
    #     cell_no = get_ipython().execution_count
    # # This happens if we are not in ipython or jupyter.
    # # No progress bar is supported in that case.
    # except AttributeError:
    #     cell_no = str(uuid.uuid4())
    #     return

    cell_no = str(uuid.uuid4()) #fix this later, temporary hack
    pbar_id = str(cell_no) + "-" + str(line_no)
    futures = [x.oid for row in result_parts for x in row]
    head_micro_partition_ids = set(futures[:len(result_parts[0])])
    tail_partition_ids = set(futures[-len(result_parts[-1]):])
    bar_format = (
        "{l_bar}{bar}{r_bar}"
        if "DEBUG_PROGRESS_BAR" in os.environ
        and os.environ["DEBUG_PROGRESS_BAR"] == "True"
        else "{desc}: {percentage:3.0f}%{bar} Elapsed time: {elapsed}, estimated remaining time: {remaining}"
    )
    already_in_progress = False
    bar_lock.acquire()
    
    if pbar_id in bar_log_ids:
        curr_id = bar_log_ids[pbar_id]
    else:
        curr_id = str(uuid.uuid4())

    bar_log_ids[pbar_id] = curr_id
    # if pbar_id in progress_bars:
    #     already_in_progress = True
    #     if hasattr(progress_bars[pbar_id], "container"):
    #         if hasattr(progress_bars[pbar_id].container.children[0], "max"):
    #             index = 0
    #         else:
    #             index = 1
    #         progress_bars[pbar_id].container.children[index].max = progress_bars[
    #             pbar_id
    #         ].container.children[index].max + len(futures)
    #     progress_bars[pbar_id].total = progress_bars[pbar_id].total + len(futures)
    #     progress_bars[pbar_id].refresh()
    # else:
    #     progress_bars[pbar_id] = tqdm_notebook(
    #         total=len(futures),
    #         desc="Estimated completion of line " + str(line_no),
    #         bar_format=bar_format,
    #     )
    bar_lock.release()

    log_file_group = os.environ["LOG_GROUP"]
    # log_file_group = "2^23_group_by_2/"

    # if not already_in_progress:
        #start another thread to poll and show updates
        # threading.Thread(target=show_time_updates, args=(progress_bars[pbar_id],)).start()

    log_file_path = "./logs/micro_part_pbar_times/" + log_file_group + curr_id + "===" + str(time.time())
    log_file = open(log_file_path, "w+")

    start_time = time.time()

    log_file.write("S " + str(start_time) + "\n")
    # first_row_micro_parts = ray.wait(futures, num_returns=len(result_parts[0]))
    # end_time = time.time()
    # micro_partition_elapsed_time = end_time - start_time

    final_finished_list = set([])
    for i in range(1, len(futures) + 1):
        finished_partition_ids, _ = ray.wait(futures, num_returns=i)
        for id in finished_partition_ids:
            if id not in final_finished_list:
                final_finished_list.add(id)
                if id in head_micro_partition_ids:
                    log_file.write("H " + str(time.time()) + "\n")
                elif id in tail_partition_ids:
                    log_file.write("T " + str(time.time()) + "\n")
                else:
                    log_file.write("N " + str(time.time()) + "\n")
                break

        # progress_bars[pbar_id].update(1)
        # progress_bars[pbar_id].refresh()
    # if progress_bars[pbar_id].n == progress_bars[pbar_id].total:
    #     progress_bars[pbar_id].close()

    log_file.flush()
    log_file.close()


def display_time_updates(bar):
    """
    Start displaying the progress `bar` in a notebook.

    Parameters
    ----------
    bar : tqdm.tqdm
        The progress bar wrapper to display in a notebook cell.
    """
    threading.Thread(target=show_time_updates, args=(bar,)).start()


def show_time_updates(p_bar):
    """
    Refresh displayed progress bar `p_bar` periodically until it is complete.

    Parameters
    ----------
    p_bar : tqdm.tqdm
        The progress bar wrapper being displayed to refresh.
    """
    while p_bar.total > p_bar.n:
        time.sleep(1)
        if p_bar.total > p_bar.n:
            p_bar.refresh()
