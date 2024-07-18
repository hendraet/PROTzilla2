import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from  protzilla.data_analysis.time_series_helper import convert_time_to_datetime

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def time_series_linear_regression(
        input_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        test_size: float,
):

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


    """
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        return dict(
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_r2=train_r2,
            test_r2=test_r2,
        )
    """

    train_df = pd.DataFrame({'Time': X_train['Time'], 'Intensity': y_train, 'Predicted': y_pred_train, 'Type': 'Train'})
    test_df = pd.DataFrame({'Time': X_test['Time'], 'Intensity': y_test, 'Predicted': y_pred_test, 'Type': 'Test'})
    plot_df = pd.concat([train_df, test_df])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Intensity'],
        mode='markers',
        name='Actual Intensity',
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Predicted'],
        mode='lines',
        name='Predicted Intensity',
        line=dict(color='red')
    ))

    fig.update_layout(
        title={
            "text": "<b>Intensity over Time</b>",
            "font": dict(size=16),
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Time",
        yaxis_title="Intensity",
        plot_bgcolor="white",
        yaxis={"gridcolor": "lightgrey", "zerolinecolor": "lightgrey"},
        font=dict(size=14, family="Arial")
    )

    return dict(plot=[fig])
