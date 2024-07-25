import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from protzilla.data_analysis.time_series_helper import convert_time_to_datetime
from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_SEQUENCE

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from plotly.subplots import make_subplots

colors = {
    "plot_bgcolor": "white",
    "gridcolor": "#F1F1F1",
    "linecolor": "#F1F1F1",
    "annotation_text_color": "#4c4c4c",
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

    fig = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25], vertical_spacing=0.025)

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Intensity'],
        mode='markers',
        name='Actual Intensity',
        marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Predicted'],
        mode='lines',
        name='Predicted Intensity',
        line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
    ), row=1, col=1)

    # Add annotation text as a separate trace in the subplot
    annotation_text = (
        f"Train RMSE: {train_rmse:.3f}<br>"
        f"Test RMSE: {test_rmse:.3f}<br>"
        f"Train R²: {train_r2:.3f}<br>"
        f"Test R²: {test_r2:.3f}"
    )

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0.25],
        text=[annotation_text],
        mode='text',
        textfont=dict(size=12),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=f"Intensity over Time for {protein_group}",
        plot_bgcolor=colors["plot_bgcolor"],
        xaxis_gridcolor=colors["gridcolor"],
        yaxis_gridcolor=colors["gridcolor"],
        xaxis_linecolor=colors["linecolor"],
        yaxis_linecolor=colors["linecolor"],
        xaxis_title="Time",
        yaxis_title="Intensity",
        legend_title="Legend",
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=50),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.825
        )
    )

    # Hide x-axis of the annotation subplot
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)

    return dict(
        train_root_mean_squared=train_rmse,
        test_root_mean_squared=test_rmse,
        train_r2_score=train_r2,
        test_r2_score=test_r2,
        plots=[fig],
    )


def time_series_ransac_regression(
        input_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        protein_group: str,
        test_size: float,
):
    """
    Perform RANSAC regression on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param test_size: The proportion of the dataset to include in the test split

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """

    if test_size < 0 or test_size > 1:
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
    model = RANSACRegressor(base_estimator=LinearRegression())
    model.fit(X_train, y_train)

    inlier_mask = model.inlier_mask_

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train[inlier_mask], y_pred_train[inlier_mask]))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train[inlier_mask], y_pred_train[inlier_mask])
    test_r2 = r2_score(y_test, y_pred_test)

    train_df = pd.DataFrame({'Time': X_train["Time"], 'Intensity': y_train, 'Predicted': y_pred_train, 'Type': 'Train'})
    test_df = pd.DataFrame({'Time': X_test["Time"], 'Intensity': y_test, 'Predicted': y_pred_test, 'Type': 'Test'})
    train_df['Inlier'] = inlier_mask
    test_df['Inlier'] = False
    plot_df = pd.concat([train_df, test_df])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25], vertical_spacing=0.025)

    # Add main plot traces
    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Intensity'],
        mode='markers',
        name='Actual Intensity',
        marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Predicted'],
        mode='lines',
        name='Predicted Intensity',
        line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df[plot_df['Inlier'] == False]['Time'],
        y=plot_df[plot_df['Inlier'] == False]['Intensity'],
        mode='markers',
        name='Outliers',
        marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[1])
    ), row=1, col=1)

    # Add annotation text as a separate trace in the subplot
    annotation_text = (
        f"Train RMSE: {train_rmse:.3f}<br>"
        f"Test RMSE: {test_rmse:.3f}<br>"
        f"Train R²: {train_r2:.3f}<br>"
        f"Test R²: {test_r2:.3f}"
    )

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0.25],
        text=[annotation_text],
        mode='text',
        textfont=dict(size=12),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=f"Intensity over Time for {protein_group}",
        plot_bgcolor=colors["plot_bgcolor"],
        xaxis_gridcolor=colors["gridcolor"],
        yaxis_gridcolor=colors["gridcolor"],
        xaxis_linecolor=colors["linecolor"],
        yaxis_linecolor=colors["linecolor"],
        xaxis_title="Time",
        yaxis_title="Intensity",
        legend_title="Legend",
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=50),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.825
        )
    )

    # Hide x-axis of the annotation subplot
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)

    return dict(
        train_root_mean_squared=train_rmse,
        test_root_mean_squared=test_rmse,
        train_r2_score=train_r2,
        test_r2_score=test_r2,
        plots=[fig],
    )


def adfuller_test(
    input_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    protein_group: str,
    alpha: float = 0.05,
) -> dict:
    """
    Perform the Augmented Dickey-Fuller test to check for stationarity in a time series.
    :param input_df: The dataframe containing the time series data.
    :param metadata_df: The dataframe containing the metadata.
    :param protein_group: The protein group to perform the test on.
    :param alpha: The significance level for the test (default is 0.05).

    :return: A dictionary containing:
        - test_statistic: The test statistic from the ADF test.
        - p_value: The p-value from the ADF test.
        - critical_values: The critical values for different significance levels.
        - is_stationary: A boolean indicating if the series is stationary.
        - messages: A list of messages for the user.
    """

    messages = []
    input_df = input_df[input_df['Protein ID'] == protein_group]

    input_df = pd.merge(
        left=input_df,
        right=metadata_df,
        on="Sample",
        copy=False,
    )

    input_df = input_df["Intensity"].dropna()

    # Perform the ADF test
    result = adfuller(input_df)
    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Determine if the series is stationary
    is_stationary = p_value < alpha

    # Create a message for the user
    if is_stationary:
        messages.append(
            {
                "level": logging.INFO,
                "msg": f"The time series is stationary (p-value: {p_value:.5f}).",
            }
        )
    else:
        messages.append(
            {
                "level": logging.WARNING,
                "msg": f"The time series is not stationary (p-value: {p_value:.5f}).",
            }
        )
    """
    fig = go.Figure()

    annotation_text = (
        f"Test Statistic: {test_statistic:.3f}<br>"
        f"P-Value: {p_value:.3f}<br>"
        f"Critical Values:<br>"
        f"Is Stationary: {is_stationary}"
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0.25],
            text=[annotation_text],
            mode='text',
            textfont=dict(size=12),
            showlegend=False
        )
    )

    fig.update_layout(
        title=f"Augmented Dickey-Fuller Test for {protein_group}",
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=50),
    )

    # Hide x-axis of the annotation subplot
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    fig.update_annotations(font_size=12)
    """
    return dict(
        test_statistic=test_statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        messages=messages,
    )

