from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protzilla.constants.paths import PEPTIDE_TEST_DATA_PATH
from protzilla.data_analysis.spectrum_prediction.spectrum import (
    KoinaModel,
    Spectrum,
    SpectrumExporter,
)
from protzilla.data_analysis.spectrum_prediction.spectrum_prediction_utils import (
    CSV_COLUMNS,
    DATA_KEYS,
    FRAGMENTATION_TYPE,
    OUTPUT_KEYS,
)
from protzilla.methods.data_analysis import PredictSpectra
from protzilla.methods.importing import EvidenceImport

evidence_small = Path(PEPTIDE_TEST_DATA_PATH) / "evidence-vsmall.txt"
evidence_bad = Path(PEPTIDE_TEST_DATA_PATH) / "evidence-vsmall-invalid-peptides.txt"


def evidence_import_run_factory(evidence_file):
    @pytest.fixture
    def evidence_import_run(run_imported):
        run = run_imported
        run.step_add(EvidenceImport())
        run.step_next()
        run.step_calculate(
            {
                "file_path": str(evidence_file),
                "intensity_name": "Intensity",
                "map_to_uniprot": False,
            }
        )
        assert (
            "peptide_df" in run.current_outputs
            and not run.current_outputs["peptide_df"].empty
        )
        return run

    return evidence_import_run


evidence_import_run_bad_evidence = evidence_import_run_factory(evidence_bad)
evidence_import_run = evidence_import_run_factory(evidence_small)


@pytest.fixture
def spectrum_prediction_run_bad_evidence(evidence_import_run_bad_evidence):
    run = evidence_import_run_bad_evidence
    run.step_add(PredictSpectra())
    run.step_next()
    return run


@pytest.fixture
def spectrum_prediction_run(evidence_import_run):
    run = evidence_import_run
    run.step_add(PredictSpectra())
    run.step_next()
    return run


@pytest.fixture
def prediction_df_complete():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2"],
            DATA_KEYS.PRECURSOR_CHARGE: [1, 2],
            DATA_KEYS.COLLISION_ENERGY: [30, 30],
            DATA_KEYS.PEPTIDE_MZ: [100, 200],
            DATA_KEYS.FRAGMENTATION_TYPE: [
                FRAGMENTATION_TYPE.HCD,
                FRAGMENTATION_TYPE.HCD,
            ],
        }
    )


@pytest.fixture
def prediction_df_incomplete():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2"],
            DATA_KEYS.PRECURSOR_CHARGE: [1, 2],
            DATA_KEYS.PEPTIDE_MZ: [100, 200],
        }
    )


@pytest.fixture
def prediction_df_large():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE" + str(i) for i in range(2000)],
            DATA_KEYS.PRECURSOR_CHARGE: [1 for _ in range(2000)],
            DATA_KEYS.PEPTIDE_MZ: [100 for _ in range(2000)],
            DATA_KEYS.COLLISION_ENERGY: [30 for _ in range(2000)],
            DATA_KEYS.FRAGMENTATION_TYPE: [FRAGMENTATION_TYPE.HCD for _ in range(2000)],
        }
    )


@pytest.fixture
def prediction_df_ptms():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTI[DE2", "PEPTI(DE3"],
            DATA_KEYS.PRECURSOR_CHARGE: [1, 2, 3],
            DATA_KEYS.PEPTIDE_MZ: [100, 200, 300],
            DATA_KEYS.COLLISION_ENERGY: [30, 30, 30],
            DATA_KEYS.FRAGMENTATION_TYPE: [
                FRAGMENTATION_TYPE.HCD,
                FRAGMENTATION_TYPE.HCD,
                FRAGMENTATION_TYPE.HCD,
            ],
        }
    )


@pytest.fixture
def prediction_df_high_charge():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
            DATA_KEYS.PRECURSOR_CHARGE: [1, 5, 6],
            DATA_KEYS.PEPTIDE_MZ: [100, 200, 300],
            DATA_KEYS.COLLISION_ENERGY: [30, 30, 30],
            DATA_KEYS.FRAGMENTATION_TYPE: [
                FRAGMENTATION_TYPE.HCD,
                FRAGMENTATION_TYPE.HCD,
                FRAGMENTATION_TYPE.HCD,
            ],
        }
    )


@pytest.fixture
def spectrum_one():
    return Spectrum(
        "sequence1",
        1,
        1.0,
        [100, 200],
        [0.5, 0.7],
        {"Charge": 1, "Parent": 1.0},
        {DATA_KEYS.FRAGMENT_TYPE: ["b1+1", "y1+2"]},
    )


@pytest.fixture
def spectrum_two():
    return Spectrum(
        "sequence2",
        2,
        2.0,
        [150, 250],
        [0.6, 0.8],
        {"Charge": 2, "Parent": 2.0},
        {DATA_KEYS.FRAGMENT_TYPE: ["b1+1", "y1+2"]},
    )


@pytest.fixture
def spectrum_none_metadata():
    return Spectrum(
        "sequence1",
        None,
        None,
        [100, 200],
        [0.5, 0.7],
        {"Charge": None},
        {DATA_KEYS.FRAGMENT_TYPE: ["b1+1", "y1+2"]},
    )


@pytest.fixture
def expected_csv_header():
    return ",".join(CSV_COLUMNS)


@pytest.fixture
def expected_tsv_header():
    return "\t".join(CSV_COLUMNS)


def test_spectrum_prediction(spectrum_prediction_run):
    spectrum_prediction_run.step_calculate(
        {
            "model_name": "PrositIntensityHCD",
            "output_format": "msp",
            "normalized_collision_energy": 30,
            "fragmentation_type": "HCD",
            "csv_seperator": ",",
        }
    )
    assert "predicted_spectra_metadata" in spectrum_prediction_run.current_outputs
    assert "predicted_spectra_peaks" in spectrum_prediction_run.current_outputs
    return


def test_spectrum_prediction_with_invalid_peptides(
    spectrum_prediction_run_bad_evidence,
):
    spectrum_prediction_run_bad_evidence.step_calculate(
        {
            "model_name": "PrositIntensityHCD",
            "output_format": "msp",
            "normalized_collision_energy": 30,
            "fragmentation_type": "HCD",
            "csv_seperator": ",",
        }
    )
    assert (
        "predicted_spectra_peaks"
        in spectrum_prediction_run_bad_evidence.current_outputs
    )
    assert (
        "predicted_spectra_metadata"
        in spectrum_prediction_run_bad_evidence.current_outputs
    )
    return


def test_preprocess_renames_columns_correctly(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.FRAGMENTATION_TYPE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
        url="",
    )
    model.prediction_df = prediction_df_complete
    model.preprocess()
    assert set(model.prediction_df.columns) == set(
        model.required_keys + [DATA_KEYS.PEPTIDE_MZ]
    )


def test_preprocess_removes_not_required_columns(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.FRAGMENTATION_TYPE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
        url="",
    )
    prediction_df_complete["ExtraColumn"] = 42
    model.prediction_df = prediction_df_complete
    model.preprocess()
    assert "ExtraColumn" not in model.prediction_df.columns


def test_preprocess_filters_out_high_charge_peptides(prediction_df_high_charge):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.FRAGMENTATION_TYPE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
        url="",
    )
    model.prediction_df = prediction_df_high_charge
    model.preprocess()
    assert len(model.prediction_df) == 2


def test_preprocess_filters_out_peptides_with_ptms(prediction_df_ptms):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.FRAGMENTATION_TYPE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
        url="",
    )
    model.prediction_df = prediction_df_ptms
    model.preprocess()
    assert len(model.prediction_df) == 1


def test_dataframe_verification_passes_with_required_keys(prediction_df_complete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    model.prediction_df = prediction_df_complete
    model.preprocess()
    model.verify_dataframe(prediction_df_complete)


def test_dataframe_verification_raises_error_when_key_missing(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            "MissingKey",
        ],
        url="",
    )
    with pytest.raises(ValueError):
        model.verify_dataframe(prediction_df_complete)


def test_slice_dataframe_returns_correct_slices(prediction_df_incomplete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    model.load_prediction_df(prediction_df_incomplete)
    slices = model.slice_dataframe()
    assert len(slices) == 1


def test_slice_dataframe_returns_correct_slices_for_large_dataframe(
    prediction_df_large,
):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    model.load_prediction_df(prediction_df_large)
    slices = model.slice_dataframe()
    assert len(slices) == 2
    slices_small = model.slice_dataframe(200)
    assert len(slices_small) == 10


def test_format_dataframes_returns_correct_output(prediction_df_complete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    model.load_prediction_df(prediction_df_complete)
    slices = model.slice_dataframe()
    formatted_data = model.format_dataframes(slices)
    assert len(formatted_data) == len(slices)


def test_format_for_request(prediction_df_complete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    formatted_data = model.format_for_request(prediction_df_complete)
    assert formatted_data is not None
    assert "id" in formatted_data
    assert "inputs" in formatted_data
    for key in model.required_keys:
        assert any(
            key == input_data["name"] for input_data in formatted_data["inputs"]
        ), f"Key {key} not found in formatted request"


def test_load_prediction_df():
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGE], url=""
    )
    df = pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_MZ: [100, 200],
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2"],
            DATA_KEYS.PRECURSOR_CHARGE: [1, 2],
        }
    )
    model.load_prediction_df(df)
    assert model.prediction_df is not None


def test_extract_fragment_information_handles_valid_annotations():
    fragment_annotations = np.array(["b1+1", "y1+2", "b2+1", "y2+2"])
    fragment_charges, fragment_types = KoinaModel.extract_fragment_information(
        fragment_annotations
    )
    expected_charges, expected_types = np.array(["1", "2", "1", "2"]), np.array(
        ["b1", "y1", "b2", "y2"]
    )
    np.testing.assert_array_equal(fragment_charges, expected_charges)
    np.testing.assert_array_equal(fragment_types, expected_types)


def test_extract_fragment_information_handles_invalid_annotations():
    fragment_annotations = np.array(["invalid1", "invalid2", "invalid3", "invalid4"])
    fragment_charges, fragment_types = KoinaModel.extract_fragment_information(
        fragment_annotations
    )
    expected_charges, expected_types = np.array(["", "", "", ""]), np.array(
        ["", "", "", ""]
    )
    np.testing.assert_array_equal(fragment_charges, expected_charges)
    np.testing.assert_array_equal(fragment_types, expected_types)


def test_extract_fragment_information_handles_mixed_annotations():
    fragment_annotations = np.array(["b1+1", "invalid1", "b2+1", "y2+2"])
    fragment_charges, fragment_types = KoinaModel.extract_fragment_information(
        fragment_annotations
    )
    expected_charges, expected_types = np.array(["1", "", "1", "2"]), np.array(
        ["b1", "", "b2", "y2"]
    )
    np.testing.assert_array_equal(fragment_charges, expected_charges)
    np.testing.assert_array_equal(fragment_types, expected_types)


def test_create_spectrum_with_valid_data():
    row = pd.Series(
        {DATA_KEYS.PEPTIDE_SEQUENCE: "PEPTIDE", DATA_KEYS.PRECURSOR_CHARGE: 2}
    )
    prepared_data = {
        OUTPUT_KEYS.MZ_VALUES: np.array([[100, 200, 300]]),
        OUTPUT_KEYS.INTENSITY_VALUES: np.array([[0.1, 0.2, 0.3]]),
        OUTPUT_KEYS.FRAGMENT_TYPE: np.array([["b", "y", "b"]]),
        OUTPUT_KEYS.FRAGMENT_CHARGE: np.array([[1, 2, 1]]),
    }
    index = 0

    spectrum = KoinaModel.create_spectrum(row, prepared_data, index)

    assert isinstance(spectrum, Spectrum)
    assert spectrum.peptide_sequence == "PEPTIDE"
    assert spectrum.precursor_charge == 2
    assert np.array_equal(
        spectrum.spectrum[DATA_KEYS.MZ].values,
        prepared_data[OUTPUT_KEYS.MZ_VALUES][index],
    )
    assert np.array_equal(
        spectrum.spectrum[DATA_KEYS.INTENSITY].values,
        prepared_data[OUTPUT_KEYS.INTENSITY_VALUES][index],
    )
    assert np.array_equal(
        spectrum.spectrum[OUTPUT_KEYS.FRAGMENT_TYPE],
        prepared_data[OUTPUT_KEYS.FRAGMENT_TYPE][index],
    )
    assert np.array_equal(
        spectrum.spectrum[OUTPUT_KEYS.FRAGMENT_CHARGE],
        prepared_data[OUTPUT_KEYS.FRAGMENT_CHARGE][index],
    )


def test_create_spectrum_with_missing_data():
    row = pd.Series(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: "PEPTIDE",
        }
    )
    prepared_data = {
        OUTPUT_KEYS.MZ_VALUES: np.array([100, 200, 300]),
        OUTPUT_KEYS.INTENSITY_VALUES: np.array([0.1, 0.2, 0.3]),
    }
    index = 1

    with pytest.raises(ValueError):
        KoinaModel.create_spectrum(row, prepared_data, index)


def test_peak_annotation_with_valid_data():
    spectrum_df = pd.DataFrame(
        {
            DATA_KEYS.MZ: [100, 200, 300],
            DATA_KEYS.INTENSITY: [0.1, 0.2, 0.3],
            OUTPUT_KEYS.FRAGMENT_TYPE: ["b", "y", "b"],
            OUTPUT_KEYS.FRAGMENT_CHARGE: [1, 2, 1],
        }
    )
    result = SpectrumExporter._format_peaks_for_msp(spectrum_df)
    expected = ['100.0\t0.1\t"b^1"\n', '200.0\t0.2\t"y^2"\n', '300.0\t0.3\t"b^1"\n']
    assert result == expected


def test_peak_annotation_with_empty_data():
    spectrum_df = pd.DataFrame(
        {
            DATA_KEYS.MZ: [],
            DATA_KEYS.INTENSITY: [],
        }
    )
    result = SpectrumExporter._format_peaks_for_msp(spectrum_df)
    assert result == []


def test_peak_annotation_with_custom_prefix_suffix():
    spectrum_df = pd.DataFrame(
        {
            DATA_KEYS.MZ: [100, 200, 300],
            DATA_KEYS.INTENSITY: [0.1, 0.2, 0.3],
            OUTPUT_KEYS.FRAGMENT_TYPE: ["b", "y", "b"],
            OUTPUT_KEYS.FRAGMENT_CHARGE: [1, 2, 1],
        }
    )
    result = SpectrumExporter._format_peaks_for_msp(spectrum_df, prefix="(", suffix=")")
    expected = ["100.0\t0.1\t(b^1)\n", "200.0\t0.2\t(y^2)\n", "300.0\t0.3\t(b^1)\n"]
    assert result == expected


import pytest

from protzilla.data_analysis.spectrum_prediction.spectrum import (
    Spectrum,
    SpectrumExporter,
)


def test_export_to_msp_with_valid_spectra(spectrum_one, spectrum_two):
    spectra = [spectrum_one, spectrum_two]
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_msp(spectra, base_file_name)
    assert "Name: sequence1" in result.content
    assert "Name: sequence2" in result.content
    assert "Charge=1" in result.content
    assert "Charge=2" in result.content
    assert "Parent=1.0" in result.content
    assert "Parent=2.0" in result.content


def test_export_to_msp_with_empty_spectra():
    spectra = []
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_msp(spectra, base_file_name)
    assert result.content == ""


def test_export_to_msp_with_none_metadata(spectrum_none_metadata):
    spectra = [spectrum_none_metadata]
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_msp(spectra, base_file_name)
    assert "Name: sequence1" in result.content
    assert "Charge=" not in result.content
    assert "Parent=" not in result.content


import pytest

from protzilla.data_analysis.spectrum_prediction.spectrum import (
    Spectrum,
    SpectrumExporter,
)


def test_export_to_csv_with_valid_spectra(
    spectrum_one, spectrum_two, expected_csv_header
):
    spectra = [spectrum_one, spectrum_two]
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_csv(spectra, base_file_name)
    assert result.file_extension == "csv"
    assert result.base_file_name == base_file_name
    assert result.filename == "test_file.csv"
    assert expected_csv_header in result.content
    assert "sequence1,1.0,1,100,0.5,b,1" in result.content
    assert "sequence2,2.0,2,150,0.6,b,1" in result.content


def test_export_to_csv_with_empty_spectra(expected_csv_header):
    spectra = []
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_csv(spectra, base_file_name)
    assert result.filename == "test_file.csv"
    assert expected_csv_header in result.content


def test_export_to_csv_with_invalid_separator(spectrum_one):
    spectra = [spectrum_one]
    base_file_name = "test_file"
    with pytest.raises(ValueError):
        SpectrumExporter.export_to_csv(spectra, base_file_name, seperator="*")


def test_export_to_csv_with_tab_separator(spectrum_one, expected_tsv_header):
    spectra = [spectrum_one]
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_csv(spectra, base_file_name, seperator="\t")
    assert result.file_extension == "tsv"
    assert expected_tsv_header in result.content
    assert "sequence1\t1.0\t1\t" in result.content


import pytest

from protzilla.data_analysis.spectrum_prediction.spectrum import (
    Spectrum,
    SpectrumExporter,
)


def test_export_to_mgf_with_valid_spectra(spectrum_one, spectrum_two):
    spectra = [spectrum_one, spectrum_two]
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_mgf(spectra, base_file_name)
    assert "BEGIN IONS" in result.content
    assert "TITLE=sequence1" in result.content
    assert "TITLE=sequence2" in result.content
    assert "CHARGE=1+" in result.content
    assert "CHARGE=2+" in result.content
    assert "END IONS" in result.content


def test_export_to_mgf_with_empty_spectra():
    spectra = []
    base_file_name = "test_file"
    result = SpectrumExporter.export_to_mgf(spectra, base_file_name)
    assert result.content == ""


def test_export_to_mgf_with_none_metadata(spectrum_none_metadata):
    spectra = [spectrum_none_metadata]
    base_file_name = "test_file"
    with pytest.raises(ValueError):
        SpectrumExporter.export_to_mgf(spectra, base_file_name)
