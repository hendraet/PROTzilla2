import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from protzilla.data_analysis.time_series_helper import convert_time_to_datetime
from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_SEQUENCE

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

colors = {
    "plot_bgcolor": "white",
    "gridcolor": "#F1F1F1",
    "linecolor": "#F1F1F1",
    "annotation_text_color": "#ffffff",
    "annotation_proteins_of_interest": "#4A536A",
}


def time_series_linear_regression(
        input_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        protein_group: str,
        test_size: float,
):
    """
    Perform linear regression on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param test_size: The proportion of the dataset to include in the test split

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """

    if test_size < 0 or test_size > 1 :
        raise ValueError("Test size should be between 0 and 1")

    input_df = input_df[input_df['Protein ID'] == protein_group]

    input_df = pd.merge(
        left=input_df,
        right=metadata_df,
        on="Sample",
        copy=False,
    )

    input_df["Time"] = input_df["Time"].apply(convert_time_to_datetime)
    input_df = input_df.interpolate(method='linear', axis=0)
    X = input_df[["Time"]]
    y = input_df["Intensity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_df = pd.DataFrame({'Time': X_train['Time'], 'Intensity': y_train, 'Predicted': y_pred_train, 'Type': 'Train'})
    test_df = pd.DataFrame({'Time': X_test['Time'], 'Intensity': y_test, 'Predicted': y_pred_test, 'Type': 'Test'})
    plot_df = pd.concat([train_df, test_df])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Intensity'],
        mode='markers',
        name='Actual Intensity',
        marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
    ))

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Predicted'],
        mode='lines',
        name='Predicted Intensity',
        line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
    ))

    fig.update_layout(
        title=f"Intensity over Time for {protein_group}",
        plot_bgcolor=colors["plot_bgcolor"],
        xaxis_gridcolor=colors["gridcolor"],
        yaxis_gridcolor=colors["gridcolor"],
        xaxis_linecolor=colors["linecolor"],
        yaxis_linecolor=colors["linecolor"],
        xaxis_title="Time (hours)",
        yaxis_title="Intensity",
        legend_title="Legend",
        autosize=True,
        margin=dict(l=100, r=300, t=100, b=100),
    )

    return dict(
        train_root_mean_squared=train_rmse,
        test_root_mean_squared=test_rmse,
        train_r2_score=train_r2,
        test_r2_score=test_r2,
        plots=[fig],
    )
