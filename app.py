#!/usr/bin/env python3

import logging
import tkinter as tk
import pandas as pd
import numpy as np

from tkinter.ttk import Treeview
from pathlib import Path
from functools import partial
from typing import Callable, Tuple, Iterable, List, Any
from collections import defaultdict

from nlp import WebSimilarity


logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__package__)


def _resolve_wordlist_path() -> Path:
    # HACK: do not this at home kids
    current_dir = Path(__file__).parent
    resource_path = Path(current_dir, "resources", "wordlist.json")
    return resource_path


def get_command(task_id: int, similarity: WebSimilarity):
    if task_id == 2:
        task = similarity.get_correlation_between_words
    elif task_id == 3:
        task = similarity.construct_result_table
    elif task_id == 5:
        task = similarity.sim_snippet1
    elif task_id == 6:
        task = similarity.execute_sim_snippet2
    elif task_id == 7:
        task = similarity.compare_tasks
    elif task_id == 8:
        task = similarity.compute_correlation_with_annotated_data
    return task


def app() -> None:
    def draw_results(func: Callable, task_id: int) -> None:
        results = func()

        # clean existing entries
        for child in (*result_frame.winfo_children(), *info_frame.winfo_children()):
            child.pack_forget()

        if task_id == 2:
            box = tk.Listbox(result_frame)
            for index, (name, values) in enumerate(results.T.items()):
                box.insert(
                    index, f"Mean {values.mean()} and std. {values.std()} for {name}"
                )

            box.pack(fill=tk.BOTH, expand=True)

        # huge and dirty hack but I don't care as this feature should not exist
        tree = Treeview(result_frame)
        if task_id not in (6,):
            tree["columns"] = list(["Output", *results.keys()])
            tree.column("Output", anchor=tk.CENTER, width=80)
            tree.insert(
                parent="",
                index="end",
                iid=0,
                text="Results",
                values=list(results.index),
            )
        else:
            tree["columns"] = list(results.keys())

        for index, (key, values) in enumerate(results.items()):
            tree.column(key, anchor=tk.CENTER, width=80)
            tree.insert(
                parent="",
                index="end",
                iid=index + 1,
                text=key,
                values=values.to_list()
                if isinstance(values, (pd.Series, pd.DataFrame))
                else values,
            )

        tree.pack(fill=tk.BOTH, expand=True)
        tk.Label(info_frame, text=f"Task {str(task_id)} results").pack(
            fill=tk.BOTH, side=tk.TOP, expand=True
        )

    wordlist_path = _resolve_wordlist_path()
    similarity = WebSimilarity(wordlist=wordlist_path)

    window = tk.Tk()
    main_frame = tk.Frame(master=window)
    main_frame.pack(fill=tk.BOTH, side=tk.TOP)

    info_frame = tk.Frame(master=main_frame)
    info_frame.pack(fill=tk.BOTH, side=tk.TOP)
    result_frame = tk.Frame(master=window, bg="white", width=500, height=300)
    result_frame.pack(fill=tk.BOTH, side=tk.TOP)

    for task_id in (2, 3, 5, 6, 7, 8):
        func = get_command(task_id, similarity)
        tk.Button(
            master=main_frame,
            text=f"Task {str(task_id)}",
            width=25,
            height=5,
            command=partial(draw_results, func, task_id),
        ).pack(side=tk.LEFT)

    window.mainloop()


if __name__ == "__main__":
    app()
