from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protzilla.constants.paths import PEPTIDE_TEST_DATA_PATH
from protzilla.data_analysis.spectrum_prediction import KoinaModel
from protzilla.data_analysis.spectrum_prediction_utils import DATA_KEYS
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
            "Sequence": ["PEPTIDE1", "PEPTIDE2"],
            "Charge": [1, 2],
            "NCE": [30, 30],
            "FragmentationType": ["HCD", "HCD"],
        }
    )


@pytest.fixture
def prediction_df_ready_for_request():
    return pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2"],
            DATA_KEYS.PRECURSOR_CHARGES: [1, 2],
            DATA_KEYS.COLLISION_ENERGIES: [30, 30],
            DATA_KEYS.FRAGMENTATION_TYPES: ["HCD", "HCD"],
        }
    )


@pytest.fixture
def prediction_df_incomplete():
    return pd.DataFrame({"Sequence": ["PEPTIDE1", "PEPTIDE2"], "Charge": [1, 2]})


@pytest.fixture
def prediction_df_large():
    return pd.DataFrame(
        {
            "Sequence": ["PEPTIDE" + str(i) for i in range(2000)],
            "Charge": [1 for _ in range(2000)],
            "NCE": [30 for _ in range(2000)],
            "FragmentationType": ["HCD" for _ in range(2000)],
        }
    )


@pytest.fixture
def prediction_df_ptms():
    return pd.DataFrame(
        {
            "Sequence": ["PEPTIDE1", "PEPTI[DE2", "PEPTI(DE3"],
            "Charge": [1, 2, 3],
            "NCE": [30, 30, 30],
            "FragmentationType": ["HCD", "HCD", "HCD"],
        }
    )


@pytest.fixture
def prediction_df_high_charge():
    return pd.DataFrame(
        {
            "Sequence": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
            "Charge": [1, 5, 6],
            "NCE": [30, 30, 30],
            "FragmentationType": ["HCD", "HCD", "HCD"],
        }
    )


def test_spectrum_prediction(spectrum_prediction_run):
    spectrum_prediction_run.step_calculate(
        {"model_name": "PrositIntensityHCD", "output_format": "msp"}
    )
    assert "predicted_spectra_df" in spectrum_prediction_run.current_outputs
    return


def test_spectrum_prediction_with_invalid_peptides(
    spectrum_prediction_run_bad_evidence,
):
    spectrum_prediction_run_bad_evidence.step_calculate(
        {"model_name": "PrositIntensityHCD", "output_format": "msp"}
    )
    assert (
        "predicted_spectra_df"
        not in spectrum_prediction_run_bad_evidence.current_outputs
    )
    return


def test_preprocess_renames_columns_correctly(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.FRAGMENTATION_TYPES,
            DATA_KEYS.COLLISION_ENERGIES,
        ],
        url="",
    )
    model.prediction_df = prediction_df_complete
    model.preprocess()
    assert set(model.prediction_df.columns) == set(model.required_keys)


def test_preprocess_removes_not_required_columns(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.FRAGMENTATION_TYPES,
            DATA_KEYS.COLLISION_ENERGIES,
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
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.FRAGMENTATION_TYPES,
            DATA_KEYS.COLLISION_ENERGIES,
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
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.FRAGMENTATION_TYPES,
            DATA_KEYS.COLLISION_ENERGIES,
        ],
        url="",
    )
    model.prediction_df = prediction_df_ptms
    model.preprocess()
    assert len(model.prediction_df) == 1


def test_dataframe_verification_passes_with_required_keys(prediction_df_complete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    model.prediction_df = prediction_df_complete
    model.preprocess()
    model.verify_dataframe(prediction_df_complete)


def test_dataframe_verification_raises_error_when_key_missing(prediction_df_complete):
    model = KoinaModel(
        required_keys=[
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            "MissingKey",
        ],
        url="",
    )
    with pytest.raises(ValueError):
        model.verify_dataframe(prediction_df_complete)


def test_slice_dataframe_returns_correct_slices(prediction_df_incomplete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    model.load_prediction_df(prediction_df_incomplete)
    slices = model.slice_dataframe()
    assert len(slices) == 1


def test_slice_dataframe_returns_correct_slices_for_large_dataframe(
    prediction_df_large,
):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    model.load_prediction_df(prediction_df_large)
    slices = model.slice_dataframe()
    assert len(slices) == 2
    slices_small = model.slice_dataframe(200)
    assert len(slices_small) == 10


def test_format_dataframes_returns_correct_output(prediction_df_complete):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    model.load_prediction_df(prediction_df_complete)
    slices = model.slice_dataframe()
    formatted_data = model.format_dataframes(slices)
    assert len(formatted_data) == len(slices)


def test_format_for_request(prediction_df_ready_for_request):
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    formatted_data = model.format_for_request(prediction_df_ready_for_request)
    assert formatted_data is not None
    assert "id" in formatted_data
    assert "inputs" in formatted_data
    for key in model.required_keys:
        assert any(
            key == input_data["name"] for input_data in formatted_data["inputs"]
        ), f"Key {key} not found in formatted request"


def test_load_prediction_df():
    model = KoinaModel(
        required_keys=[DATA_KEYS.PEPTIDE_SEQUENCE, DATA_KEYS.PRECURSOR_CHARGES], url=""
    )
    df = pd.DataFrame(
        {
            DATA_KEYS.PEPTIDE_SEQUENCE: ["PEPTIDE1", "PEPTIDE2"],
            DATA_KEYS.PRECURSOR_CHARGES: [1, 2],
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
