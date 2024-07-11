from enum import StrEnum

from ui.runs.forms.fill_helper import enclose_links_in_html


def format_citation(model_name):
    model_info = MODEL_METADATA[model_name]
    citation_string = (
        f"{model_name}</b>:<br>"
        f'{enclose_links_in_html(model_info.get("citation", ""))}<br>'
        f'{enclose_links_in_html(model_info["github_url"]) if model_info.get("github_url") else ""}<br>'
    )
    return citation_string


class FRAGMENTATION_TYPE(StrEnum):
    HCD = "HCD"
    CID = "CID"


class DATA_KEYS(StrEnum):
    """Commonly used keys in the dataframes."""

    PEPTIDE_SEQUENCE = "peptide_sequences"
    PRECURSOR_CHARGE = "precursor_charges"
    PEPTIDE_MZ = "peptide_m/z"
    MZ = "m/z"
    COLLISION_ENERGY = "collision_energies"
    FRAGMENTATION_TYPE = "fragmentation_type"
    # These are used for the peaks
    INTENSITY = "intensity"
    FRAGMENT_TYPE = "fragment_type"
    FRAGMENT_CHARGE = "fragment_charge"


class CSV_KEYS(StrEnum):
    PEPTIDE_SEQUENCE = "StrippedSequence"
    PRECURSOR_CHARGE = "PrecursorCharge"
    PEPTIDE_MZ = "PrecursorMz"
    MZ = "FragmentMz"
    INTENSITY = "RelativeFragmentIntensity"
    FRAGMENT_TYPE = "FragmentType"
    FRAGMENT_NUMBER = "FragmentNumber"
    FRAGMENT_CHARGE = "FragmentCharge"


CSV_COLUMNS = [
    CSV_KEYS.PEPTIDE_SEQUENCE,
    CSV_KEYS.PEPTIDE_MZ,
    CSV_KEYS.PRECURSOR_CHARGE,
    CSV_KEYS.MZ,
    CSV_KEYS.INTENSITY,
    CSV_KEYS.FRAGMENT_TYPE,
    CSV_KEYS.FRAGMENT_NUMBER,
]


class OUTPUT_KEYS(StrEnum):
    """These are the keys that are that are returned from the API"""

    MZ_VALUES = "mz"
    INTENSITY_VALUES = "intensities"
    ANNOTATIONS = "annotation"
    FRAGMENT_TYPE = "fragment_type"
    FRAGMENT_CHARGE = "fragment_charge"


class AVAILABLE_MODELS(StrEnum):
    PrositIntensityHCD = "PrositIntensityHCD"
    PrositIntensityCID = "PrositIntensityCID"
    PrositIntensityTimsTOF = "PrositIntensityTimsTOF"
    PrositIntensityTMT = "PrositIntensityTMT"


MODEL_METADATA = {
    AVAILABLE_MODELS.PrositIntensityHCD: {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_HCD/infer",
        "citation": "Wilhelm, M., Zolg, D.P., Graber, M. et al.  Nat Commun 12, 3346 (2021). https://doi.org/10.1038/s41467-021-23713-9",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
    },
    AVAILABLE_MODELS.PrositIntensityCID: {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_CID/infer",
        "citation": "Wilhelm, M., Zolg, D.P., Graber, M. et al. Nat Commun 12, 3346 (2021). https://doi.org/10.1038/s41467-021-23713-9",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
        ],
    },
    AVAILABLE_MODELS.PrositIntensityTimsTOF: {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2023_intensity_timsTOF/infer",
        "citation": "C., Gabriel, W., Laukens, K. et al. Nat Commun 15, 3956 (2024). https://doi.org/10.1038/s41467-024-48322-0",
        "github_url": None,
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.COLLISION_ENERGY,
        ],
    },
    AVAILABLE_MODELS.PrositIntensityTMT: {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_TMT/infer",
        "citation": " Wassim Gabriel, Matthew The, Daniel P. Zolg, Florian P. Bayer, et al. Analytical Chemistry 2022 94 (20), 7181-7190 https://doi.org/10.1021/acs.analchem.1c05435",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGE,
            DATA_KEYS.COLLISION_ENERGY,
            DATA_KEYS.FRAGMENTATION_TYPE,
        ],
    },
}
formatted_citation_dict = {
    model_name: format_citation(model_name) for model_name in MODEL_METADATA
}


class AVAILABLE_FORMATS(StrEnum):
    CSV_TSV = "csv/tsv"
    MSP = "msp"
    MGF = "mgf"
