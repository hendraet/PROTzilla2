import asyncio
import json
import logging
import re
from typing import Dict, Optional

import aiohttp
import numpy as np
import pandas as pd
import plotly.express as px
from pyteomics.mass import mass
from tqdm import tqdm

from protzilla.constants.colors import PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
from protzilla.constants.protzilla_logging import logger
from protzilla.data_analysis.spectrum_prediction_utils import (
    AVAILABLE_MODELS,
    DATA_KEYS,
    FRAGMENTATION_TYPE,
    OUTPUT_KEYS,
)
from protzilla.disk_operator import FileOutput


class Spectrum:
    def __init__(
        self,
        peptide_sequence: str,
        charge: int,
        mz_values: np.array,
        intensity_values: np.array,
        metadata: Optional[dict] = None,
        annotations: Optional[dict] = None,
        sanitize: bool = True,
    ):
        self.peptide_sequence = peptide_sequence
        self.peptide_mz = mass.calculate_mass(
            sequence=peptide_sequence, charge=charge, ion_type="M"
        )
        self.metadata = metadata if metadata else {}
        self.metadata[
            "Charge"
        ] = charge  # TODO maybe this can be handled in the exporting functions instead
        self.peptide_charge = charge

        self.spectrum = pd.DataFrame(
            zip(mz_values, intensity_values), columns=["m/z", "Intensity"]
        )
        if annotations:
            for key, value in annotations.items():
                self.spectrum[key] = value

        if sanitize:
            self._sanitize_spectrum()

    def __repr__(self):
        return f"{self.peptide_sequence}: {self.charge}, {self.spectrum.shape[0]} peaks"

    def _sanitize_spectrum(self):
        self.spectrum = self.spectrum.drop_duplicates(subset="m/z")
        self.spectrum = self.spectrum[self.spectrum["Intensity"] > 0]

    def to_mergeable_df(self):
        return self.spectrum.assign(
            Sequence=self.peptide_sequence, Charge=self.metadata["Charge"]
        )


class SpectrumPredictor:
    def __init__(self, prediction_df: Optional[pd.DataFrame]):
        self.prediction_df = prediction_df

    def predict(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def verify_dataframe(self, prediction_df: Optional[pd.DataFrame] = None):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )


class KoinaModel(SpectrumPredictor):
    ptm_regex = re.compile(r"[\[\(]")
    FRAGMENT_ANNOTATION_PATTERN = re.compile(r"((y|b)\d+)\+(\d+)")

    def __init__(
        self,
        required_keys: list[str],
        url: str,
        prediction_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        super().__init__(prediction_df)
        self.required_keys = required_keys
        self.KOINA_URL = url
        if prediction_df is not None:
            self.load_prediction_df(prediction_df)

    def load_prediction_df(self, prediction_df: pd.DataFrame):
        self.prediction_df = prediction_df
        self.preprocess()
        self.verify_dataframe()
        return self.prediction_df

    def verify_dataframe(self, prediction_df: Optional[pd.DataFrame] = None):
        if prediction_df is None:
            prediction_df = self.prediction_df
        for key in self.required_keys:
            if key not in prediction_df.columns:
                raise ValueError(f"Required key '{key}' not found in input DataFrame.")

    def preprocess(self):
        self.prediction_df.rename(
            columns={
                "Sequence": DATA_KEYS.PEPTIDE_SEQUENCE,
                "Charge": DATA_KEYS.PRECURSOR_CHARGES,
                "NCE": DATA_KEYS.COLLISION_ENERGIES,
                "FragmentationType": DATA_KEYS.FRAGMENTATION_TYPES,
            },
            inplace=True,
        )
        self.prediction_df = (
            self.prediction_df[self.required_keys]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self.prediction_df = self.prediction_df[self.required_keys]
        self.prediction_df = self.prediction_df[
            self.prediction_df[DATA_KEYS.PRECURSOR_CHARGES] <= 5
        ]
        # filter all peptides which match a ptm
        self.prediction_df = self.prediction_df[
            ~self.prediction_df[DATA_KEYS.PEPTIDE_SEQUENCE].str.contains(
                self.ptm_regex, regex=True
            )
        ]

    def predict(self):
        predicted_spectra = []
        slice_indices = self.slice_dataframe()
        formatted_data = self.format_dataframes(slice_indices)
        response_data = asyncio.run(self.make_request(formatted_data, slice_indices))
        for response, indices in tqdm(
            response_data,
            desc="Processing predictions",
            total=len(response_data),
        ):
            predicted_spectra.extend(
                self.process_response(self.prediction_df.loc[indices], response)
            )
        return predicted_spectra

    def format_dataframes(self, slice_indices):
        """Format a list of slices of the prediction DataFrame for the request."""
        return [
            self.format_for_request(self.prediction_df.loc[slice])
            for slice in slice_indices
        ]

    def slice_dataframe(self, chunk_size: int = 1000):
        """Slice the prediction DataFrame into chunks of size chunk_size."""
        return [
            self.prediction_df[i : i + chunk_size].index
            for i in range(0, len(self.prediction_df), chunk_size)
        ]

    def format_for_request(self, to_predict: pd.DataFrame) -> dict:
        """Format the DataFrame for the request to the Koina API. Returns the formatted data as a JSON string."""
        inputs = []
        if DATA_KEYS.PEPTIDE_SEQUENCE in to_predict.columns:
            inputs.append(
                {
                    "name": str(DATA_KEYS.PEPTIDE_SEQUENCE),
                    "shape": [len(to_predict), 1],
                    "datatype": "BYTES",
                    "data": to_predict[DATA_KEYS.PEPTIDE_SEQUENCE].to_list(),
                }
            )
        if DATA_KEYS.PRECURSOR_CHARGES in to_predict.columns:
            inputs.append(
                {
                    "name": str(DATA_KEYS.PRECURSOR_CHARGES),
                    "shape": [len(to_predict), 1],
                    "datatype": "INT32",
                    "data": to_predict[DATA_KEYS.PRECURSOR_CHARGES].to_list(),
                }
            )
        if DATA_KEYS.COLLISION_ENERGIES in to_predict.columns:
            inputs.append(
                {
                    "name": str(DATA_KEYS.COLLISION_ENERGIES),
                    "shape": [len(to_predict), 1],
                    "datatype": "FP32",
                    "data": to_predict[DATA_KEYS.COLLISION_ENERGIES].to_list(),
                }
            )
        if DATA_KEYS.FRAGMENTATION_TYPES in to_predict.columns:
            inputs.append(
                {
                    "name": str(DATA_KEYS.FRAGMENTATION_TYPES),
                    "shape": [len(to_predict), 1],
                    "datatype": "BYTES",
                    "data": to_predict[DATA_KEYS.FRAGMENTATION_TYPES].to_list(),
                }
            )
        return {"id": "0", "inputs": inputs}

    async def make_request(
        self, formatted_data: list[dict], slice_indices: list
    ) -> list[dict]:
        """Asynchronously make a POST request to the Koina API. Returns the response data as a list of dictionaries.
        In the case of an error, we will recursively divide and conquer the data until we get a successful response.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for data, indices in zip(formatted_data, slice_indices):
                tasks.append(
                    (session.post(self.KOINA_URL, data=json.dumps(data)), indices)
                )
                logger.info(
                    f"Requesting {len(indices)} spectra {indices[0]}-{indices[-1]}..."
                )
            responses = []
            for task, indices in tasks:
                response = await task
                if response.status != 200:
                    if len(indices) > 1:
                        mid = len(indices) // 2
                        a, b = indices[:mid], indices[mid:]
                        logger.warning(
                            f"Error response received for {indices[0]}-{indices[-1]}, splitting data: {a[0]}-{a[-1]} and {b[0]}-{b[-1]}..."
                        )
                        formatted_data = self.format_dataframes([a, b])
                        responses.extend(
                            await self.make_request(formatted_data, [a, b])
                        )
                    else:
                        logger.error(
                            f"Skipping peptide {self.prediction_df.loc[indices[0]][DATA_KEYS.PEPTIDE_SEQUENCE]} with charge {self.prediction_df.loc[indices[0]][DATA_KEYS.PRECURSOR_CHARGES]}."
                        )
                        logger.debug(f"Error in response: {await response.text()}")
                else:
                    responses.append((await response.json(), indices))
        return responses

    @staticmethod
    def process_response(request: pd.DataFrame, response: dict) -> list[Spectrum]:
        prepared_data = KoinaModel.prepare_data(response)
        return [
            KoinaModel.create_spectrum(row, prepared_data, i)
            for i, (_, row) in enumerate(request.iterrows())
        ]

    @staticmethod
    def prepare_data(response: dict) -> dict:
        results = KoinaModel.extract_data_from_response(response)
        fragment_charges, fragment_types = KoinaModel.extract_fragment_information(
            results[OUTPUT_KEYS.ANNOTATIONS]
        )
        results[OUTPUT_KEYS.FRAGMENT_TYPE] = fragment_types
        results[OUTPUT_KEYS.FRAGMENT_CHARGE] = fragment_charges
        return results

    @staticmethod
    def create_spectrum(
        row: pd.Series, prepared_data: Dict[str, np.ndarray], index: int
    ) -> Spectrum:
        try:
            peptide_sequence = row.get(DATA_KEYS.PEPTIDE_SEQUENCE)
            precursor_charges = row.get(DATA_KEYS.PRECURSOR_CHARGES)
            mz_values = prepared_data.get(OUTPUT_KEYS.MZ_VALUES)[index]
            intensity_values = prepared_data.get(OUTPUT_KEYS.INTENSITY_VALUES)[index]
            annotations = {
                OUTPUT_KEYS.FRAGMENT_TYPE: prepared_data.get(OUTPUT_KEYS.FRAGMENT_TYPE)[
                    index
                ],
                OUTPUT_KEYS.FRAGMENT_CHARGE: prepared_data.get(
                    OUTPUT_KEYS.FRAGMENT_CHARGE
                )[index],
            }
            return Spectrum(
                peptide_sequence,
                precursor_charges,
                mz_values,
                intensity_values,
                annotations=annotations,
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Error while creating spectrum: {e}")

    @staticmethod
    def extract_fragment_information(fragment_annotations: np.array):
        def extract_annotation_information(annotation_str: str):
            if match := KoinaModel.FRAGMENT_ANNOTATION_PATTERN.match(annotation_str):
                return match.group(1), match.group(3)
            return None, None

        vectorized_annotation_extraction = np.vectorize(extract_annotation_information)
        fragment_types, fragment_charges = vectorized_annotation_extraction(
            fragment_annotations
        )
        # TODO find a way to make the None values always represented as ""
        fragment_types, fragment_charges = fragment_types.astype(
            str
        ), fragment_charges.astype(str)
        fragment_types[fragment_types == "None"] = ""
        fragment_charges[fragment_charges == "None"] = ""
        return fragment_charges, fragment_types

    @staticmethod
    def extract_data_from_response(response: dict):
        results = {}
        for output in response["outputs"]:
            results[output["name"]] = np.reshape(output["data"], output["shape"])
        return results


class SpectrumPredictorFactory:
    @staticmethod
    def create_predictor(model_name: str) -> KoinaModel:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' is not available.")
        return KoinaModel(**AVAILABLE_MODELS[model_name])


def predict(
    model_name: str,
    peptide_df: pd.DataFrame,
    output_format: str,
    csv_seperator: str = ",",
):
    predictor = SpectrumPredictorFactory.create_predictor(model_name)
    prediction_df = peptide_df[["Sequence", "Charge"]].copy().drop_duplicates()
    # TODO this is only a temporary fix
    if DATA_KEYS.COLLISION_ENERGIES in predictor.required_keys:
        prediction_df["NCE"] = 30
    if DATA_KEYS.FRAGMENTATION_TYPES in predictor.required_keys:
        prediction_df["FragmentationType"] = FRAGMENTATION_TYPE.HCD

    predictor.load_prediction_df(prediction_df)
    predicted_spectra = predictor.predict()
    # merge df's into big df
    base_name = "predicted_spectra"
    if output_format == "csv":
        output = SpectrumExporter.export_to_csv(predicted_spectra, base_name)
    elif output_format == "msp":
        output = SpectrumExporter.export_to_msp(predicted_spectra, base_name)
    return {
        "predicted_spectra": output,
        "predicted_spectra_df": pd.concat(
            [spectrum.to_mergeable_df() for spectrum in predicted_spectra]
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


class SpectrumExporter:
    @staticmethod
    def export_to_msp(spectra: list[Spectrum], base_file_name: str):
        lines = []
        for spectrum in tqdm(
            spectra, desc="Exporting spectra to .msp format", total=len(spectra)
        ):
            header_dict = {
                "Name": spectrum.peptide_sequence,
                "Comment": "".join([f"{k}={v} " for k, v in spectrum.metadata.items()]),
                "Num Peaks": len(spectrum.spectrum),
            }
            header = "\n".join([f"{k}: {v}" for k, v in header_dict.items()])
            peaks = SpectrumExporter.format_peaks(spectrum.spectrum)
            peaks = "".join(peaks)
            lines.append(f"{header}\n{peaks}\n")

        logger.info(
            f"Exported {len(spectra)} spectra to MSP format, now combining them"
        )
        content = "\n".join(lines)
        logger.info("Export finished!")
        return FileOutput(base_file_name, "msp", content)

    @staticmethod
    def format_peaks(
        spectrum_df: pd.DataFrame,
        prefix='"',
        seperator=" ",
        suffix='"',
        add_newline=True,
    ):
        peaks = [
            f"{mz}\t{intensity}"
            for mz, intensity in spectrum_df[["m/z", "Intensity"]].values
        ]
        annotations = [f"{prefix}" for _ in spectrum_df.values]
        if len(spectrum_df.columns) <= 2:
            pass
        else:
            for column in spectrum_df.columns[2:]:
                if column == OUTPUT_KEYS.FRAGMENT_CHARGE:
                    annotations = [
                        current_annotation_str[:-1] + f"^{fragment_charge}{seperator}"
                        for current_annotation_str, fragment_charge in zip(
                            annotations, spectrum_df[column]
                        )
                    ]
                    continue

                annotations = [
                    current_annotation_str + str(annotation) + seperator
                    for current_annotation_str, annotation in zip(
                        annotations, spectrum_df[column]
                    )
                ]
            annotations = [
                current_annotation_str[:-1] for current_annotation_str in annotations
            ]
            peaks = [
                f"{peak}\t{annotation}{suffix}\n"
                for peak, annotation in zip(peaks, annotations)
            ]

        return peaks

    @staticmethod
    def export_to_csv(
        spectra: list[Spectrum], base_file_name: str, seperator: str = ","
    ):
        # required columns: PrecursorMz, FragmentMz
        # recommended columns: iRT (impossible), RelativeFragmentIntensity (maybe possible, @Chris),
        # - StrippedSequence (peptide sequence without modifications, easy)
        # - PrecursorCharge (maybe possible with evidence)
        # - FragmentType (b or y, maybe possible, depends on model)
        # - FragmentNumber (after which AA in the sequence the cut is, i think)
        #
        if seperator not in [",", ";", "\t"]:
            raise ValueError(r"Invalid seperator, please use one of: ',' , ';' , '\t'")
        if seperator == "\t":
            file_extension = "tsv"
        else:
            file_extension = "csv"

        output_df = pd.DataFrame()
        spectrum_dfs = []
        fragment_pattern = re.compile(r"([yb])(\d+)")
        for spectrum in tqdm(spectra, desc="Preparing spectra"):
            spectrum_df = spectrum.spectrum
            spectrum_df["PrecursorMz"] = spectrum.peptide_mz
            spectrum_df["StrippedSequence"] = spectrum.peptide_sequence
            spectrum_df["PrecursorCharge"] = spectrum.metadata["Charge"]
            spectrum_df["FragmentNumber"] = spectrum_df[
                OUTPUT_KEYS.FRAGMENT_TYPE
            ].apply(lambda x: fragment_pattern.match(x).group(2))
            spectrum_df["FragmentType"] = spectrum_df[OUTPUT_KEYS.FRAGMENT_TYPE].apply(
                lambda x: fragment_pattern.match(x).group(1)
            )
            spectrum_df.rename(
                columns={"m/z": "FragmentMz", "Intensity": "RelativeFragmentIntensity"},
                inplace=True,
            )

            spectrum_df = spectrum_df[
                [
                    "PrecursorMz",
                    "StrippedSequence",
                    "FragmentMz",
                    "PrecursorCharge",
                    "FragmentNumber",
                    "FragmentType",
                    "RelativeFragmentIntensity",
                ]
            ]
            spectrum_dfs.append(spectrum_df)
        output_df = pd.concat(spectrum_dfs, ignore_index=True)
        content = output_df.to_csv(sep=seperator, index=False)
        return FileOutput(base_file_name, file_extension, content)


def plot_spectrum(
    prediction_df: pd.DataFrame, peptide: str, charge: int, annotation_threshold: float
):
    assert 0 <= annotation_threshold and annotation_threshold <= 1
    b_ion_color, y_ion_color = PROTZILLA_DISCRETE_COLOR_OUTLIER_SEQUENCE
    spectrum = prediction_df[
        (prediction_df["Sequence"] == peptide) & (prediction_df["Charge"] == charge)
    ]
    plot_df = spectrum[["m/z", "Intensity", "fragment_type", "fragment_charge"]]
    plot_df["fragment_ion"] = [fragment[0] for fragment in plot_df["fragment_type"]]
    # Plotting the peaks
    fig = px.bar(
        plot_df,
        x="m/z",
        y="Intensity",
        hover_data=["fragment_type", "fragment_charge"],
        labels={"x": "m/z", "y": "Relative intensity"},
        color="fragment_ion",
        color_discrete_map={"b": b_ion_color, "y": y_ion_color},
        title=f"{peptide} ({charge}+)",
    )
    fig.update_traces(width=3.0)

    # Adding the annotations
    for _, row in plot_df.iterrows():
        if row["Intensity"] < annotation_threshold:
            continue
        fig.add_annotation(
            x=row["m/z"],
            y=row["Intensity"],
            font=dict(
                color=y_ion_color if "y" in row["fragment_type"] else b_ion_color
            ),
            text=f"{row['fragment_type']} ({row['fragment_charge']}+)",
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
