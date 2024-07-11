import numpy as np
import pandas as pd
import pytest

from protzilla.data_analysis.time_series_plot_peptide import time_series_plot_peptide


@pytest.fixture
def time_series_test_data():
    test_intensity_list = (
        ["Sample1", "Protein1", "Gene1", 20],
        ["Sample1", "Protein2", "Gene1", 16],
        ["Sample1", "Protein3", "Gene1", 1],
        ["Sample1", "Protein4", "Gene1", 14],
        ["Sample2", "Protein1", "Gene1", 20],
        ["Sample2", "Protein2", "Gene1", 15],
        ["Sample2", "Protein3", "Gene1", 2],
        ["Sample2", "Protein4", "Gene1", 15],
        ["Sample3", "Protein1", "Gene1", 22],
        ["Sample3", "Protein2", "Gene1", 14],
        ["Sample3", "Protein3", "Gene1", 3],
        ["Sample3", "Protein4", "Gene1", 16],
        ["Sample4", "Protein1", "Gene1", 8],
        ["Sample4", "Protein2", "Gene1", 15],
        ["Sample4", "Protein3", "Gene1", 1],
        ["Sample4", "Protein4", "Gene1", 9],
        ["Sample5", "Protein1", "Gene1", 10],
        ["Sample5", "Protein2", "Gene1", 14],
        ["Sample5", "Protein3", "Gene1", 2],
        ["Sample5", "Protein4", "Gene1", 10],
        ["Sample6", "Protein1", "Gene1", 12],
        ["Sample6", "Protein2", "Gene1", 13],
        ["Sample6", "Protein3", "Gene1", 3],
        ["Sample6", "Protein4", "Gene1", 11],
        ["Sample7", "Protein1", "Gene1", 12],
        ["Sample7", "Protein2", "Gene1", 13],
        ["Sample7", "Protein3", "Gene1", 3],
        ["Sample7", "Protein4", "Gene1", 11],
    )

    test_intensity_df = pd.DataFrame(
        data=test_intensity_list,
        columns=["Sample", "Protein ID", "Gene", "Intensity"],
    )

    test_metadata_df = (
        ["Sample1", "02:00:00", 1],
        ["Sample2", "06:00:00", 1],
        ["Sample3", "10:00:00", 1],
         ["Sample4", "14:00:00", 1],
    )
    test_metadata_df = pd.DataFrame(
        data=test_metadata_df,
        columns=["Sample", "Time", "Day"],
    )
    return test_intensity_df, test_metadata_df

def test_time_series_plot(show_figures, time_series_test_data):
    test_intensity, test_metadata = time_series_test_data
    outputs = time_series_plot_peptide(test_intensity, test_metadata, "Protein1")
    assert "plots" in outputs
    fig = outputs["plots"][0]
    if show_figures:
        fig.show()
    return