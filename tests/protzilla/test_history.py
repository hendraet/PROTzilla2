from shutil import rmtree

import pandas as pd

from protzilla.constants.paths import RUNS_PATH
from protzilla.history import History
from protzilla.utilities.random import random_string


def test_history_memory_identity():
    name = "test_memory_identity" + random_string()
    history = History(name, df_mode="memory")
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    history.add_step(
        "section1", "step1", "method1", {"param1": 3}, df1, outputs={}, plots=[]
    )
    history.add_step(
        "section2", "step2", "method2", {"param1": 5}, df2, outputs={}, plots=[]
    )
    assert history.steps[0].dataframe is df1
    assert history.steps[0].dataframe is not df2
    assert history.steps[1].dataframe is df2
    assert history.steps[1].dataframe is not df1
    rmtree(RUNS_PATH / name)


def test_history_disk():
    name = "test_disk" + random_string()
    history = History(name, df_mode="disk")
    df1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    history.add_step(
        "section1", "step1", "method1", {"param1": 3}, df1, outputs={}, plots=[]
    )
    # the history should not contain a dataframe in memory when using disk mode
    assert history.steps[0]._dataframe is None
    # the dataframe should be loaded from disk correctly when using .dataframe
    assert history.steps[0].dataframe.equals(df1)
    rmtree(RUNS_PATH / name)


def test_history_disk_delete():
    name = "test_disk_delete" + random_string()
    history = History(name, df_mode="disk")
    df1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    history.add_step(
        "section1", "step1", "method1", {"param1": 3}, df1, outputs={}, plots=[]
    )
    assert history.df_path(0).exists()
    history.remove_step()
    assert not history.df_path(0).exists()
    rmtree(RUNS_PATH / name)


def test_history_save():
    name = "test_history_save" + random_string()
    history = History(name, df_mode="disk")
    df1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    history.add_step(
        "section1", "step1", "method1", {"param1": 3}, df1, outputs={}, plots=[]
    )
    df2 = pd.DataFrame(data={"hellocol": [2, 2], "col2": [4, 4]})
    history.add_step(
        "section2",
        "step2",
        "method2",
        {"param3": 3, "other": "no"},
        df2,
        outputs={"out": 5},
        plots=[],
    )
    history.save()
    steps = history.steps
    del history
    history2 = History.from_disk(name, df_mode="disk")
    assert steps[0].dataframe.equals(history2.steps[0].dataframe)
    assert history2.steps[0]._dataframe is None
    assert steps[1].dataframe.equals(history2.steps[1].dataframe)
    assert history2.steps[1]._dataframe is None
    rmtree(RUNS_PATH / name)


def test_dataframe_in_json():
    name = "test_history_df" + random_string()
    history = History(name, df_mode="disk")
    df1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"asdfs": [3, 2], "aaaa": [99, 1]})
    history.add_step(
        "section1",
        "step1",
        "method1",
        {"param1": 3},
        df1,
        outputs={"another_df": df2},
        plots=[],
    )
    assert (RUNS_PATH / name / "history.json").exists()
    del history
    history2 = History.from_disk(name, df_mode="disk")
    assert df2.equals((history2.steps[0].outputs["another_df"]))
    rmtree(RUNS_PATH / name)
