import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from protzilla.data_analysis.time_series_helper import convert_time_to_hours
from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_SEQUENCE

from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
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
        train_size: float = 0.2,
        grouping: str = None,
):
    """
    Perform linear regression on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param train_size: The proportion of the dataset to include in the test split
    :param grouping: Option to select whether regression should be performed on the entire dataset or separately on the control and experimental groups

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """
    color_index = 0
    if train_size < 0 or train_size > 1:
        raise ValueError("Test size should be between 0 and 1")

    input_df = input_df[input_df['Protein ID'] == protein_group]

    input_df = pd.merge(
        left=input_df,
        right=metadata_df,
        on="Sample",
        copy=False,
    )

    input_df["Time"] = input_df["Time"].apply(convert_time_to_hours)
    input_df = input_df.interpolate(method='linear', axis=0)

    input_df = input_df.sample(frac=1, random_state = 42).reset_index(drop=True)

    X = input_df[["Time"]]
    y = input_df["Intensity"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25], vertical_spacing=0.025)

    scores = []

    if grouping == "With Grouping" and "Group" in input_df.columns:
        groups = input_df["Group"].unique()
        for group in groups:
            group_df = input_df[input_df["Group"] == group]
            X_group = group_df[["Time"]]
            y_group = group_df["Intensity"]

            X_train, X_test, y_train, y_test = train_test_split(X_group, y_group, test_size=train_size, shuffle=False)
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

            color = PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index % len(PROTZILLA_DISCRETE_COLOR_SEQUENCE)]
            color_index += 3

            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=plot_df['Intensity'],
                mode='markers',
                name=f'Actual Intensity ({group})',
                marker=dict(color=color)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=plot_df['Predicted'],
                mode='lines',
                name=f'Predicted Intensity ({group})',
                line=dict(color=color)
            ), row=1, col=1)

            scores.append({
                'group': group,
                'train_root_mean_squared': train_rmse,
                'test_root_mean_squared': test_rmse,
                'train_r2_score': train_r2,
                'test_r2_score': test_r2,
            })

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
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

        scores.append({
            'group': 'Overall',
            'train_root_mean_squared': train_rmse,
            'test_root_mean_squared': test_rmse,
            'train_r2_score': train_r2,
            'test_r2_score': test_r2,
        })

    # Add annotation text as a separate trace in the subplot
    annotation_text = "<br>".join([
        f"Group: {res['group']} (Train/Test)"
        f"<br>RMSE: {res['train_root_mean_squared']:.3f} / {res['test_root_mean_squared']:.3f}<br>"
        f"Train R²: {res['train_r2_score']:.3f} / {res['test_r2_score']:.3f}<br>"
        for res in scores
    ])

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
            x=0.8
        )
    )

    # Hide x-axis of the annotation subplot
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)

    return dict(
        scores=scores,
        plots=[fig],
    )


def time_series_ransac_regression(
        input_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        protein_group: str,
        max_trials: int,
        stop_probability: float,
        loss: str,
        train_size: float,
        grouping: str,
):
    """
    Perform RANSAC regression on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param train_size: The proportion of the dataset to include in the test split

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """

    color_index = 0
    if train_size < 0 or train_size > 1:
        raise ValueError("Test size should be between 0 and 1")

    input_df = input_df[input_df['Protein ID'] == protein_group]

    input_df = pd.merge(
        left=input_df,
        right=metadata_df,
        on="Sample",
        copy=False,
    )

    input_df["Time"] = input_df["Time"].apply(convert_time_to_hours)
    input_df = input_df.interpolate(method='linear', axis=0)

    input_df = input_df.sample(frac=1, random_state = 42).reset_index(drop=True)

    X = input_df[["Time"]]
    y = input_df["Intensity"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.75, 0.25], vertical_spacing=0.025)

    scores = []

    if grouping == "With Grouping" and "Group" in input_df.columns:
        groups = input_df["Group"].unique()
        for group in groups:
            group_df = input_df[input_df["Group"] == group]
            X_group = group_df[["Time"]]
            y_group = group_df["Intensity"]

            X_train, X_test, y_train, y_test = train_test_split(X_group, y_group, train_size=train_size, shuffle=False)
            model = RANSACRegressor(max_trials = max_trials, stop_probability = stop_probability, loss = loss, base_estimator=LinearRegression())
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

            # Add main plot traces
            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=plot_df['Intensity'],
                mode='markers',
                name='Actual Intensity',
                marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=plot_df['Predicted'],
                mode='lines',
                name='Predicted Intensity',
                line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 1])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=plot_df[plot_df['Inlier'] == False]['Time'],
                y=plot_df[plot_df['Inlier'] == False]['Intensity'],
                mode='markers',
                name='Outliers',
                marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 2])
            ), row=1, col=1)

            color_index += 3

            scores.append({
                'group': group,
                'train_root_mean_squared': train_rmse,
                'test_root_mean_squared': test_rmse,
                'train_r2_score': train_r2,
                'test_r2_score': test_r2,
            })

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size, shuffle=False)
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
            line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=plot_df[plot_df['Inlier'] == False]['Time'],
            y=plot_df[plot_df['Inlier'] == False]['Intensity'],
            mode='markers',
            name='Outliers',
            marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
        ), row=1, col=1)

        scores.append({
            'group': 'Overall',
            'train_root_mean_squared': train_rmse,
            'test_root_mean_squared': test_rmse,
            'train_r2_score': train_r2,
            'test_r2_score': test_r2,
        })

    # Add annotation text as a separate trace in the subplot
    annotation_text = "<br>".join([
        f"Group: {res['group']} (Train/Test)"
        f"<br>RMSE: {res['train_root_mean_squared']:.3f} / {res['test_root_mean_squared']:.3f}<br>"
        f"Train R²: {res['train_r2_score']:.3f} / {res['test_r2_score']:.3f}<br>"
        for res in scores
    ])

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
            x=0.8
        )
    )

    # Hide x-axis of the annotation subplot
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)

    return dict(
        scores=scores,
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

    return dict(
        test_statistic=test_statistic,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        messages=messages,
    )


def time_series_auto_arima(
    input_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    protein_group: str,
    seasonal: str,
    m: int,
    train_size: float,
    grouping: str,
) -> dict:
    """
    Perform an automatic ARIMA model selection on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param seasonal: Whether the ARIMA model should be seasonal
    :param m: The number of time steps for a single seasonal period (ignored if seasonal=False)
    :param train_size: The proportion of the dataset to include in the test split
    :param grouping: Whether to group the data by the 'Group' column

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """

    color_index = 0

    if train_size < 0 or train_size > 1:
        raise ValueError("Train size should be between 0 and 1")
    if seasonal == "Yes":
        seasonal = True
    else:
        seasonal = False

    input_df = input_df[input_df['Protein ID'] == protein_group]
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)

    input_df = pd.merge(
        left=input_df,
        right=metadata_df,
        on="Sample",
        copy=False,
    )

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3])
    scores = []

    if grouping == "With Grouping" and "Group" in input_df.columns:
        groups = input_df["Group"].unique()
        for group in groups:
            group_df = input_df[input_df["Group"] == group]

            group_df["Time"] = group_df["Time"].apply(convert_time_to_hours)
            group_df = group_df.interpolate(method='linear', axis=0)

            train_df_size = int(len(group_df) * train_size)
            train_df, test_df = group_df[:train_df_size], group_df[train_df_size:]

            train_df = train_df.set_index("Time")["Intensity"]
            test_df = test_df.set_index("Time")["Intensity"]

            # Fit the ARIMA model
            model = auto_arima(
                train_df,
                seasonal=seasonal,
                m=m,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
            )

            # Forecast the test set
            forecast = model.predict(n_periods=test_df.shape[0])

            test_rmse = np.sqrt(mean_squared_error(test_df, forecast))
            test_r2 = r2_score(test_df, forecast)
            train_rmse = np.sqrt(mean_squared_error(train_df, model.predict_in_sample()))
            train_r2 = r2_score(train_df, model.predict_in_sample())

            forecast_reset = forecast.reset_index(drop=True)
            forecast_plot = pd.Series(forecast_reset.values, index=test_df.index)
            forecast_plot = forecast_plot.groupby(forecast_plot.index).mean()

            fig.add_trace(go.Scatter(
                x=test_df.index,
                y=test_df,
                mode='markers',
                name='Actual Intensity',
                marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=test_df.index,
                y=forecast,
                mode='markers',
                name='Predicted Intensity',
                line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 2])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = forecast_plot.index,
                y = forecast_plot,
                mode = 'lines',
                name = 'Mean Predicted Intensity',
                line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 2])
            ), row=1, col=1)

            color_index += 3

            scores.append({
                'group': group,
                 'train_root_mean_squared': train_rmse,
                'test_root_mean_squared': test_rmse,
                'train_r2_score': train_r2,
                'test_r2_score': test_r2,
            })

    else:
        input_df["Time"] = input_df["Time"].apply(convert_time_to_hours)
        input_df = input_df.interpolate(method='linear', axis=0)

        train_size = int(len(input_df) * train_size)
        train_df, test_df = input_df[:train_size], input_df[train_size:]

        train_df = train_df.set_index("Time")["Intensity"]
        test_df = test_df.set_index("Time")["Intensity"]

        # Fit the ARIMA model
        model = auto_arima(
            train_df,
            seasonal=seasonal,
            m=m,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
        )

        # Forecast the test set
        forecast = model.predict(n_periods=test_df.shape[0])

        test_rmse = np.sqrt(mean_squared_error(test_df, forecast))
        test_r2 = r2_score(test_df, forecast)
        train_rmse = np.sqrt(mean_squared_error(train_df, model.predict_in_sample()))
        train_r2 = r2_score(train_df, model.predict_in_sample())

        forecast_reset = forecast.reset_index(drop=True)
        forecast_plot = pd.Series(forecast_reset.values, index=test_df.index)
        forecast_plot = forecast_plot.groupby(forecast_plot.index).mean()

        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df,
            mode='markers',
            name='Actual Intensity',
            marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=forecast,
            mode='markers',
            name='Predicted Intensity',
            line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_plot.index,
            y=forecast_plot,
            mode='lines',
            name='Mean Predicted Intensity',
            line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[3])
        ), row=1, col=1)

        scores.append({
            'group': 'Overall',
            'train_root_mean_squared': train_rmse,
            'test_root_mean_squared': test_rmse,
            'train_r2_score': train_r2,
            'test_r2_score': test_r2,
        })

    # Add annotation text as a separate trace in the subplot
    annotation_text = "<br>".join([
        f"Group: {res['group']} (Train/Test)"
        f"<br>RMSE: {res['train_root_mean_squared']:.3f} / {res['test_root_mean_squared']:.3f}<br>"
        f"Train R²: {res['train_r2_score']:.3f} / {res['test_r2_score']:.3f}<br>"
        for res in scores
    ])

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
            x=0.775
        )
    )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)


    return dict(
        scores=scores,
        plots=[fig],
    )


def time_series_arima(
    input_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    protein_group: str,
    seasonal: str,
    p: int,
    d: int,
    q: int,
    train_size: float,
    grouping: str,
) -> dict:

    """
    Perform ARIMA model selection on the time series data for a given protein group.
    :param input_df: Peptide dataframe which contains the intensity of each sample
    :param metadata_df: Metadata dataframe which contains the timestamps
    :param protein_group: Protein group to perform the analysis on
    :param seasonal: Whether the ARIMA model should be seasonal
    :param p: ARIMA p parameter
    :param d: ARIMA d parameter
    :param q: ARIMA q parameter
    :param train_size: The proportion of the dataset to include in the test split
    :param grouping: Whether to group the data by the 'Group' column

    :return: A dictionary containing the root mean squared error and r2 score for the training and test sets
    """

    color_index = 0

    if train_size < 0 or train_size > 1:
        raise ValueError("Train size should be between 0 and 1")

    input_df = input_df[input_df['Protein ID'] == protein_group]
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)

    input_df = pd.merge(left=input_df, right=metadata_df, on="Sample", copy=False)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3])
    scores = []

    if grouping == "With Grouping" and "Group" in input_df.columns:
        groups = input_df["Group"].unique()
        for group in groups:
            group_df = input_df[input_df["Group"] == group]

            group_df["Time"] = group_df["Time"].apply(convert_time_to_hours)
            group_df = group_df.interpolate(method='linear', axis=0)

            train_df_size = int(len(group_df) * train_size)
            train_df, test_df = group_df[:train_df_size], group_df[train_df_size:]

            train_df = train_df.set_index("Time")["Intensity"]
            test_df = test_df.set_index("Time")["Intensity"]

            if seasonal == "Yes":
                model = ARIMA(
                    train_df,
                    order=(p, d, q),
                    #seasonal_order=(P, D, Q, m)
                )
            else:
                model = ARIMA(
                    train_df,
                    order=(p, d, q)
                )

            model_fit = model.fit()

            forecast = model_fit.forecast(steps=len(test_df))

            test_rmse = np.sqrt(mean_squared_error(test_df, forecast))
            test_r2 = r2_score(test_df, forecast)
            train_rmse = np.sqrt(mean_squared_error(train_df, model_fit.fittedvalues))
            train_r2 = r2_score(train_df, model_fit.fittedvalues)

            forecast_reset = forecast.reset_index(drop=True)
            forecast_plot = pd.Series(forecast_reset.values, index=test_df.index)
            forecast_mean_plot = forecast_plot.groupby(forecast_plot.index).mean()

            fig.add_trace(go.Scatter(
                x=test_df.index,
                y=test_df,
                mode='markers',
                name='Actual Intensity',
                marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=forecast_plot.index,
                y=forecast_plot,
                mode='markers',
                name='Predicted Intensity',
                line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 2])
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x = forecast_mean_plot.index,
                y = forecast_mean_plot,
                mode = 'lines',
                name = 'Mean Predicted Intensity',
                line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[color_index + 2])
            ), row=1, col=1)

            color_index += 3

            scores.append({
                'group': group,
                'train_root_mean_squared': train_rmse,
                'test_root_mean_squared': test_rmse,
                'train_r2_score': train_r2,
                'test_r2_score': test_r2,
            })

    else:
        input_df["Time"] = input_df["Time"].apply(convert_time_to_hours)
        input_df = input_df.interpolate(method='linear', axis=0)

        train_size = int(len(input_df) * train_size)
        train_df, test_df = input_df[:train_size], input_df[train_size:]

        train_df = train_df.set_index("Time")["Intensity"]
        test_df = test_df.set_index("Time")["Intensity"]

        if seasonal == "Yes":
            model = ARIMA(train_df, order=(p, d, q))
        else:
            model = ARIMA(train_df, order=(p, d, q))

        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test_df))

        test_rmse = np.sqrt(mean_squared_error(test_df, forecast))
        test_r2 = r2_score(test_df, forecast)
        train_rmse = np.sqrt(mean_squared_error(train_df, model_fit.fittedvalues))
        train_r2 = r2_score(train_df, model_fit.fittedvalues)

        forecast_reset = forecast.reset_index(drop=True)
        forecast_plot = pd.Series(forecast_reset.values, index=test_df.index)
        forecast_plot = forecast_plot.groupby(forecast_plot.index).mean()

        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=test_df,
            mode='markers',
            name='Actual Intensity',
            marker=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[0])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=test_df.index,
            y=forecast,
            mode='markers',
            name='Predicted Intensity',
            line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[2])
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=forecast_plot.index,
            y=forecast_plot,
            mode='lines',
            name='Mean Predicted Intensity',
            line=dict(color=PROTZILLA_DISCRETE_COLOR_SEQUENCE[3])
        ), row=1, col=1)

        scores.append({
            'group': 'Overall',
            'train_root_mean_squared': train_rmse,
            'test_root_mean_squared': test_rmse,
            'train_r2_score': train_r2,
            'test_r2_score': test_r2,
        })

    annotation_text = "<br>".join([
        f"Group: {res['group']} (Train/Test)"
        f"<br>RMSE: {res['train_root_mean_squared']:.3f} / {res['test_root_mean_squared']:.3f}<br>"
        f"Train R²: {res['train_r2_score']:.3f} / {res['test_r2_score']:.3f}<br>"
        for res in scores
    ])

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
            x=0.775
        )
    )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)

    fig.update_annotations(font_size=12)

    return dict(
        scores=scores,
        plots=[fig],
    )
