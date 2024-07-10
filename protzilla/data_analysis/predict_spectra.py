import logging
from typing import Optional

import pandas as pd
import plotly.express as px

from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
from protzilla.data_analysis.spectrum_prediction.spectrum import (
    SpectrumExporter,
    SpectrumPredictorFactory,
)
from protzilla.data_analysis.spectrum_prediction.spectrum_prediction_utils import (
    DATA_KEYS,
)


def predict(
    model_name: str,
    peptide_df: pd.DataFrame,
    output_format: str,
    normalized_collision_energy: Optional[float] = None,
    fragmentation_type: Optional[str] = None,
    csv_seperator: Optional[str] = ",",
):
    # First order of business: rename the columns to the expected names
    peptide_df = peptide_df.rename(
        columns={
            "Sequence": DATA_KEYS.PEPTIDE_SEQUENCE,
            "Charge": DATA_KEYS.PRECURSOR_CHARGE,
            "m/z": DATA_KEYS.PEPTIDE_MZ,
        },
        errors="ignore",
    )
    prediction_df = (
        peptide_df[
            [
                DATA_KEYS.PEPTIDE_SEQUENCE,
                DATA_KEYS.PRECURSOR_CHARGE,
                DATA_KEYS.PEPTIDE_MZ,
            ]
        ]
        .drop_duplicates()
        .copy()[::10]
    )  # TODO this is only for testing, remove
    prediction_df[DATA_KEYS.COLLISION_ENERGY] = normalized_collision_energy
    prediction_df[DATA_KEYS.FRAGMENTATION_TYPE] = fragmentation_type
    predictor = SpectrumPredictorFactory.create_predictor(model_name)
    predictor.load_prediction_df(prediction_df)
    predicted_spectra = predictor.predict()
    base_name = "predicted_spectra"
    if output_format == "csv":
        output = SpectrumExporter.export_to_csv(predicted_spectra, base_name)
    elif output_format == "msp":
        output = SpectrumExporter.export_to_msp(predicted_spectra, base_name)

    max_number_of_peaks = max(
        [len(spectrum.spectrum) for spectrum in predicted_spectra]
    )  # TODO make that a constant
    return {
        "predicted_spectra": output,
        "predicted_spectra_df": pd.concat(
            [
                spectrum.to_mergeable_df(max_number_of_peaks)
                for spectrum in predicted_spectra
            ]
            if predicted_spectra
            else pd.DataFrame()
        ),
        "messages": [
            {
                "level": logging.INFO,
                "msg": f"Successfully predicted {len(predicted_spectra)} spectra.",
            }
        ],
    }


def plot_spectrum(
    prediction_df: pd.DataFrame, peptide: str, charge: int, annotation_threshold: float
):
    assert 0 <= annotation_threshold and annotation_threshold <= 1
    b_ion_color, y_ion_color = PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
    spectrum = prediction_df[
        (prediction_df[DATA_KEYS.PEPTIDE_SEQUENCE] == peptide)
        & (prediction_df[DATA_KEYS.PRECURSOR_CHARGE] == charge)
    ]
    plot_df = spectrum[
        [
            DATA_KEYS.MZ,
            DATA_KEYS.INTENSITY,
            DATA_KEYS.FRAGMENT_TYPE,
            DATA_KEYS.FRAGMENTATION_CHARGE,
        ]
    ]
    plot_df["fragment_ion"] = [fragment[0] for fragment in plot_df["fragment_type"]]
    # Plotting the peaks
    fig = px.bar(
        plot_df,
        x=DATA_KEYS.MZ,
        y=DATA_KEYS.INTENSITY,
        hover_data=[DATA_KEYS.FRAGMENT_TYPE, DATA_KEYS.FRAGMENTATION_CHARGE],
        labels={"x": "m/z", "y": "Relative intensity"},
        color="fragment_ion",
        color_discrete_map={"b": b_ion_color, "y": y_ion_color},
        title=f"{peptide} ({charge}+)",
    )
    fig.update_traces(width=3.0)

    # Adding the annotations
    for _, row in plot_df.iterrows():
        if row[DATA_KEYS.INTENSITY] < annotation_threshold:
            continue
        fig.add_annotation(
            x=row[DATA_KEYS.MZ],
            y=row[DATA_KEYS.INTENSITY],
            font=dict(
                color=y_ion_color
                if "y" in row[DATA_KEYS.FRAGMENT_TYPE]
                else b_ion_color
            ),
            text=f"{row[DATA_KEYS.FRAGMENT_TYPE]} ({row[DATA_KEYS.FRAGMENT_CHARGE]}+)",
            showarrow=False,
            yshift=25,
            textangle=-90,
        )

    # Updating the color legend to say "y" and "b" instead of the color codes
    fig.for_each_trace(
        lambda trace: trace.update(
            name=trace.name.replace(b_ion_color, "b-ion").replace(y_ion_color, "y-ion")
        )
    )
    # Replace title of legend with "Fragment type"
    fig.update_layout(legend_title_text="Fragment type")
    return dict(plots=[fig])
