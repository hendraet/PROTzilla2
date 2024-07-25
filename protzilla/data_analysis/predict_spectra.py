import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
from protzilla.data_analysis.spectrum_prediction.spectrum import (
    SpectrumExporter,
    SpectrumPredictorFactory,
)
from protzilla.data_analysis.spectrum_prediction.spectrum_prediction_utils import (
    DataKeys,
    FragmentationType,
    GenericTextSeparator,
    OutputFormats,
    PredictionModels,
)


def predict(
    model_name: PredictionModels,
    peptide_df: pd.DataFrame,
    output_format: OutputFormats,
    normalized_collision_energy: Optional[float],
    fragmentation_type: Optional[FragmentationType],
    column_seperator: Optional[GenericTextSeparator],
    output_dir: Optional[str] = None,
):
    """
    Predicts the spectra for the given peptides using the specified model.
    :param model_name: the model to use
    :param peptide_df: the result of the evidence import, containing the peptide sequences, charges and m/z values
    :param output_format: output format of the spectral predictions
    :param normalized_collision_energy: the normalized collision energy for which to predict the spectra
    :param fragmentation_type: the type of ms fragmentation for which to predict the spectra
    :param column_seperator: the column separator to use in case the output format is generic text
    :param output_dir: the directory to save the output to, this will just be shown to the user in the return message so he knows where to find the output
    :return: a dictionary containing the output file, metadata and peaks dataframes of the predicted spectra and a message
    """
    # First order of business: rename the columns to the expected names
    peptide_df = peptide_df.rename(
        columns={
            "Sequence": DataKeys.PEPTIDE_SEQUENCE,
            "Charge": DataKeys.PRECURSOR_CHARGE,
            "m/z": DataKeys.PEPTIDE_MZ,
        },
        errors="ignore",
    )
    prediction_df = (
        peptide_df[
            [
                DataKeys.PEPTIDE_SEQUENCE,
                DataKeys.PRECURSOR_CHARGE,
                DataKeys.PEPTIDE_MZ,
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    predictor = SpectrumPredictorFactory.create_predictor(model_name)
    if DataKeys.COLLISION_ENERGY in predictor.required_keys:
        assert normalized_collision_energy is not None, "Collision energy is required."
        prediction_df[DataKeys.COLLISION_ENERGY] = normalized_collision_energy
    if DataKeys.FRAGMENTATION_TYPE in predictor.required_keys:
        assert fragmentation_type is not None, "Fragmentation type is required."
        prediction_df[DataKeys.FRAGMENTATION_TYPE] = fragmentation_type
    predictor.load_prediction_df(prediction_df)
    predicted_spectra = predictor.predict()
    base_name = "predicted_spectra"
    if output_format == OutputFormats.CSV_TSV:
        output = SpectrumExporter.export_to_generic_text(
            predicted_spectra, base_name, column_seperator
        )
    elif output_format == OutputFormats.MSP:
        output = SpectrumExporter.export_to_msp(predicted_spectra, base_name)
    elif output_format == OutputFormats.MGF:
        output = SpectrumExporter.export_to_mgf(predicted_spectra, base_name)

    metadata_dfs = []
    peaks_dfs = []
    for spectrum in predicted_spectra:
        metadata_df, peaks_df = spectrum.to_mergeable_df()
        metadata_dfs.append(metadata_df)
        peaks_dfs.append(peaks_df)

    combined_metadata_df = pd.concat(metadata_dfs)
    combined_peaks_df = pd.concat(peaks_dfs)

    return {
        "predicted_spectra": output,
        "predicted_spectra_metadata": combined_metadata_df,
        "predicted_spectra_peaks": combined_peaks_df,
        "messages": [
            {
                "level": logging.INFO,
                "msg": f"Successfully predicted {len(predicted_spectra)} spectra. The output can be found at {output_dir if output_dir else 'the dataframe folder of the run'}.",
            }
        ],
    }


def plot_spectrum(
    metadata_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    peptide: str,
    charge: int,
    annotation_threshold: float,
):
    assert 0 <= annotation_threshold and annotation_threshold <= 1

    # Get the unique_id for the specified peptide and charge
    unique_id = metadata_df[
        (metadata_df[DataKeys.PEPTIDE_SEQUENCE] == peptide)
        & (metadata_df[DataKeys.PRECURSOR_CHARGE] == charge)
    ].index

    # Filter the peaks_df for the specific spectrum
    spectrum = peaks_df.loc[unique_id]

    plot_df = spectrum[
        [
            DataKeys.MZ,
            DataKeys.INTENSITY,
            DataKeys.FRAGMENT_TYPE,
            DataKeys.FRAGMENT_CHARGE,
        ]
    ]
    plot_df["fragment_ion"] = plot_df[DataKeys.FRAGMENT_TYPE].str[0]

    ion_color = PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
    ion_types = plot_df[DataKeys.FRAGMENT_TYPE].str[0].unique()
    if len(ion_types) != 2:
        raise ValueError(
            f"Expected exactly two fragment types, but got {len(ion_types)}: {ion_types}"
        )

    # Plotting the peaks
    fig = px.bar(
        plot_df,
        x=DataKeys.MZ,
        y=DataKeys.INTENSITY,
        hover_data=[DataKeys.FRAGMENT_TYPE, DataKeys.FRAGMENT_CHARGE],
        color="fragment_ion",
        color_discrete_map={ion_types[0]: ion_color[0], ion_types[1]: ion_color[1]},
        title=f"{peptide} ({charge}+)",
    )

    # Updating the layout
    fig.update_layout(
        yaxis=dict(
            title="Relative intensity",
            range=[0, 1.2],
            tickvals=[0, 0.5, 1],
            ticktext=["0.0", "0.5", "1.0"],
            ticks="outside",
            showline=True,
            linewidth=1,
            linecolor="grey",
        ),
        xaxis=dict(
            title="m/z",
            tickmode="linear",
            ticks="outside",
            tick0=0,
            ticklabelstep=2,
            tickangle=-45,
            dtick=50,
            showline=True,
            linewidth=1,
            linecolor="grey",
        ),
    )
    fig.update_traces(width=3.0)

    # Adding the annotations
    for _, row in plot_df.iterrows():
        if row[DataKeys.INTENSITY] < annotation_threshold:
            continue
        fig.add_annotation(
            x=row[DataKeys.MZ],
            y=row[DataKeys.INTENSITY],
            font=dict(
                color=ion_color[0]
                if ion_types[0] in row[DataKeys.FRAGMENT_TYPE]
                else ion_color[1]
            ),
            text=f"{row[DataKeys.FRAGMENT_TYPE]} ({row[DataKeys.FRAGMENT_CHARGE]}+)",
            showarrow=False,
            yshift=30,
            textangle=-90,
        )

    # Updating the color legend to say "y" and "b" instead of the color codes
    fig.for_each_trace(
        lambda trace: trace.update(
            name=trace.name.replace(ion_color[0], f"{ion_types[0]}-ion").replace(
                ion_color[1], f"{ion_color[1]}-ion"
            )
        )
    )
    # Replace title of legend with "Fragment type"
    fig.update_layout(legend_title_text="Fragment type")
    to_be_returned = dict(
        plots=[fig],
        messages=[
            {
                "level": logging.INFO,
                "msg": f"Successfully plotted the spectrum for {peptide} ({charge}+). Tip: You can zoom in by selecting an area on the plot.",
            }
        ],
    )
    to_be_returned[f"spectrum_{peptide}_{charge}"] = plot_df
    return to_be_returned


def plot_mirror_spectrum(
    metadata_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    peptide: str,
    charge: int,
    annotation_threshold: float,
):
    # Get the unique_id for the specified peptide and charge
    unique_id = metadata_df[
        (metadata_df["peptide_sequences"] == peptide)
        & (metadata_df["precursor_charges"] == charge)
    ].index

    # Filter the peaks_df for the specific spectrum
    spectrum = peaks_df.loc[unique_id]

    upper_plot_df = spectrum[
        [
            "m/z",
            "intensity",
            "fragment_type",
            "fragment_charge",
        ]
    ]
    upper_plot_df["fragment_ion"] = upper_plot_df["fragment_type"].str[0]

    cosine_similarity = advanced_cosine_similarity(
        upper_plot_df.reset_index()[[DataKeys.MZ, DataKeys.INTENSITY]],
        plot_df.reset_index()[[DataKeys.MZ, DataKeys.INTENSITY]],
        0.1,
    )

    ion_color = PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
    ion_types = upper_plot_df["fragment_type"].str[0].unique()
    if len(ion_types) != 2:
        raise ValueError(
            f"Expected exactly two fragment types, but got {len(ion_types)}: {ion_types}"
        )

    title = f"{peptide} ({charge}+) - Mirror spectrum (Cosine similarity: {cosine_similarity:.2f})"

    # Create a single plot
    fig = go.Figure()

    # Upper plot (theoretical spectrum)
    for ion_type in ion_types:
        df_ion = upper_plot_df[upper_plot_df["fragment_ion"] == ion_type]
        fig.add_trace(
            go.Bar(
                x=df_ion["m/z"],
                y=df_ion["intensity"],
                name=f"{ion_type}-ion (theoretical)",
                marker_color=ion_color[list(ion_types).index(ion_type)],
                hovertemplate="m/z: %{x}<br>Intensity: %{y}<br>Fragment: %{text}",
                text=df_ion["fragment_type"]
                + " ("
                + df_ion["fragment_charge"].astype(str)
                + "+)",
            )
        )

    # Lower plot (experimental spectrum)
    fig.add_trace(
        go.Bar(
            x=plot_df["m/z"],
            y=-plot_df["intensity"],  # Negative values for mirror effect
            name="Experimental",
            marker_color="rgba(128, 128, 128, 0.7)",
            hovertemplate="m/z: %{x}<br>Intensity: %{customdata}",
            customdata=plot_df["intensity"],
        )
    )

    # Add annotations to the upper plot
    for _, row in upper_plot_df.iterrows():
        if row["intensity"] >= annotation_threshold:
            fig.add_annotation(
                x=row["m/z"],
                y=row["intensity"],
                text=f"{row['fragment_type']} ({row['fragment_charge']}+)",
                showarrow=False,
                yshift=30,
                textangle=-90,
                font=dict(
                    color=ion_color[list(ion_types).index(row["fragment_ion"])],
                ),
            )

    # Update layout
    fig.update_layout(
        title=title,
        barmode="overlay",
        legend_title_text="Fragment type",
        showlegend=True,
        height=600,
        yaxis=dict(
            title="Relative intensity",
            range=[-1.2, 1.2],  # Set y-axis range from -1 to 1
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["1.0", "0.5", "0.0", "0.5", "1.0"],
        ),
    )

    fig.update_traces(width=3)

    to_be_returned = dict(
        plots=[fig],
        messages=[
            {
                "level": logging.INFO,
                "msg": f"Successfully plotted the mirror spectrum for {peptide} ({charge}+). Tip: You can zoom in by selecting an area on the plot.",
            }
        ],
    )
    to_be_returned[f"spectrum_{peptide}_{charge}"] = plot_df
    return to_be_returned


def advanced_cosine_similarity(
    experimental_peaks_df: pd.DataFrame,
    predicted_peaks_df: pd.DataFrame,
    mz_tolerance: float,
) -> float:
    """
    Calculate the cosine similarity between two spectra.
    :param experimental_peaks_df:
    :param predicted_peaks_df:
    :param mz_tolerance:
    :return:
    """
    original_experimental_peaks_df = experimental_peaks_df.copy()
    original_predicted_peaks_df = predicted_peaks_df.copy()
    matches = []
    unmatched_experimental_peaks = []
    unmatched_theoretical_peaks = []
    for mz_a, int_a in predicted_peaks_df[[DataKeys.MZ, DataKeys.INTENSITY]].values:
        candidates = experimental_peaks_df[
            (experimental_peaks_df[DataKeys.MZ] >= mz_a - mz_tolerance)
            & (experimental_peaks_df[DataKeys.MZ] <= mz_a + mz_tolerance)
        ]
        if candidates.empty:
            unmatched_theoretical_peaks.append((mz_a, int_a))
            continue

        index = candidates[DataKeys.INTENSITY].idxmax()
        mz_b, int_b = experimental_peaks_df.loc[
            index, [DataKeys.MZ, DataKeys.INTENSITY]
        ]
        experimental_peaks_df = experimental_peaks_df.drop(index)
        matches.append(((mz_a, int_a), (mz_b, int_b)))

    for mz_b, int_b in experimental_peaks_df[[DataKeys.MZ, DataKeys.INTENSITY]].values:
        unmatched_experimental_peaks.append((mz_b, int_b))
    if not matches:
        return 0.0
    # Calculate the cosine similarity
    squared_sum_exp_intensities = sum(
        [
            intensity**2
            for mz, intensity in original_experimental_peaks_df[
                [DataKeys.MZ, DataKeys.INTENSITY]
            ].values
        ]
    )
    squared_sum_pred_intensities = sum(
        [
            intensity**2
            for mz, intensity in original_predicted_peaks_df[
                [DataKeys.MZ, DataKeys.INTENSITY]
            ].values
        ]
    )
    squared_sum_unmatched_exp_intensities = sum(
        [intensity**2 for mz, intensity in unmatched_experimental_peaks]
    )
    squared_sum_unmatched_pred_intensities = sum(
        [intensity**2 for mz, intensity in unmatched_theoretical_peaks]
    )
    numerator_a = sum([int_a * int_b for (mz_a, int_a), (mz_b, int_b) in matches])
    demoninator_a = (squared_sum_exp_intensities**0.5) * (
        squared_sum_pred_intensities**0.5
    )
    term_a = numerator_a / demoninator_a

    numerator_b = (
        squared_sum_unmatched_exp_intensities * squared_sum_unmatched_pred_intensities
    )
    denominator_b = squared_sum_exp_intensities * squared_sum_pred_intensities
    term_b = numerator_b / denominator_b
    similarity = term_a - term_b
    if similarity > 1 or similarity < -1:
        raise ValueError(f"Invalid cosine similarity: {similarity}")
    return similarity


def compare_single_spectrum(args, experimental_df, predicted_df):
    peptide_sequence, precursor_charge, experiment_name, spectrum_ref, exp_df = args
    print(
        f"Comparing {peptide_sequence} ({precursor_charge}+) with {experiment_name} and spectrum {spectrum_ref}."
    )
    pred_df = predicted_df[
        (predicted_df[DataKeys.PEPTIDE_SEQUENCE] == peptide_sequence)
        & (predicted_df[DataKeys.PRECURSOR_CHARGE] == precursor_charge)
    ]
    similarity = advanced_cosine_similarity(
        exp_df.reset_index()[[DataKeys.MZ, DataKeys.INTENSITY]],
        pred_df.reset_index()[[DataKeys.MZ, DataKeys.INTENSITY]],
        0.05,
    )
    modifications = exp_df["modifications"].unique()[0]
    return {
        "peptide_sequence": peptide_sequence,
        "precursor_charge": precursor_charge,
        "experiment": experiment_name,
        "spectrum_ref": spectrum_ref,
        "similarity": similarity,
        "modifications": modifications if isinstance(modifications, str) else "",
    }


def compare_experimental_with_predicted_spectra(
    experimental_df: pd.DataFrame, predicted_df: pd.DataFrame, threads: int = 16
):
    if threads is None:
        threads = cpu_count()

    predicted_df[DataKeys.PEPTIDE_SEQUENCE] = predicted_df[
        DataKeys.PEPTIDE_SEQUENCE
    ].str.replace("J", "L")
    experimental_df[DataKeys.PEPTIDE_SEQUENCE] = experimental_df[
        DataKeys.PEPTIDE_SEQUENCE
    ].str.replace("J", "L")

    comparison_args = []
    for (peptide_sequence, precursor_charge), exp_group in experimental_df.groupby(
        [DataKeys.PEPTIDE_SEQUENCE, DataKeys.PRECURSOR_CHARGE]
    ):
        if peptide_sequence not in predicted_df[DataKeys.PEPTIDE_SEQUENCE].values:
            continue
        for (experiment_name, spectrum_ref), exp_df in exp_group.groupby(
            ["experiment", "spectra_ref"]
        ):
            comparison_args.append(
                (
                    peptide_sequence,
                    precursor_charge,
                    experiment_name,
                    spectrum_ref,
                    exp_df,
                )
            )

    with Pool(threads) as pool:
        compare_func = partial(
            compare_single_spectrum,
            experimental_df=experimental_df,
            predicted_df=predicted_df,
        )
        results = pool.map(compare_func, comparison_args)

    result_df = pd.DataFrame(results)

    # Plot the distributions of the similarities in a histogram
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=result_df[result_df["modifications"] == ""]["similarity"],
            name="Without modifications",
            histnorm="probability",
            nbinsx=30,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=result_df[result_df["modifications"] != ""]["similarity"],
            name="With modifications",
            histnorm="probability",
            nbinsx=30,
            opacity=0.7,
        )
    )
    # add information about the mean and median of the similarities for both groups
    fig.add_annotation(
        x=0.5,
        y=0.9,
        xref="paper",
        yref="paper",
        text=f"Mean similarity without modifications: {result_df[result_df['modifications'] == '']['similarity'].mean():.2f}",
        showarrow=False,
    )
    fig.add_annotation(
        x=0.5,
        y=0.85,
        xref="paper",
        yref="paper",
        text=f"Mean similarity with modifications: {result_df[result_df['modifications'] != '']['similarity'].mean():.2f}",
        showarrow=False,
    )

    fig.update_layout(
        title="Distribution of Similarities",
        xaxis_title="Adapted Cosine Similarity",
        yaxis_title="Probability",
        barmode="overlay",
    )

    return {
        "plots": [fig],
        "comparison_result_df": result_df,
        "messages": [
            {
                "level": logging.INFO,
                "msg": f"Successfully compared the experimental spectra with the predicted spectra using {threads} threads.",
            }
        ],
    }
