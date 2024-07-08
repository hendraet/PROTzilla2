from enum import StrEnum

from ui.runs.forms.fill_helper import enclose_links_in_html


def format_citation(model_name):
    model_info = AVAILABLE_MODELS[model_name]
    citation_string = (
        f"Info about {model_name}:<br>"
        f'{enclose_links_in_html(model_info.get("citation", ""))}<br>'
        f'{enclose_links_in_html(model_info["github_url"]) if model_info.get("github_url") else ""}<br>'
    )
    return citation_string


class FRAGMENTATION_TYPE(StrEnum):
    HCD = "HCD"
    CID = "CID"


class DATA_KEYS(StrEnum):
    PEPTIDE_SEQUENCE = "peptide_sequences"
    PRECURSOR_CHARGES = "precursor_charges"
    COLLISION_ENERGIES = "collision_energies"
    FRAGMENTATION_TYPES = "fragmentation_types"


class OUTPUT_KEYS(StrEnum):
    MZ_VALUES = "mz"
    INTENSITY_VALUES = "intensities"
    ANNOTATIONS = "annotation"
    FRAGMENT_TYPE = "fragment_type"
    FRAGMENT_CHARGE = "fragment_charge"


AVAILABLE_MODELS = {
    "PrositIntensityHCD": {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_HCD/infer",
        "citation": "Wilhelm, M., Zolg, D.P., Graber, M. et al.  Nat Commun 12, 3346 (2021). https://doi.org/10.1038/s41467-021-23713-9",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.COLLISION_ENERGIES,
        ],
    },
    "PrositIntensityCID": {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_CID/infer",
        "citation": "Wilhelm, M., Zolg, D.P., Graber, M. et al. Nat Commun 12, 3346 (2021). https://doi.org/10.1038/s41467-021-23713-9",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
        ],
    },
    "PrositIntensityTimsTOF": {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2023_intensity_timsTOF/infer",
        "citation": "C., Gabriel, W., Laukens, K. et al. Nat Commun 15, 3956 (2024). https://doi.org/10.1038/s41467-024-48322-0",
        "github_url": None,
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.COLLISION_ENERGIES,
        ],
    },
    "PrositIntensityTMT": {
        "url": "https://koina.wilhelmlab.org/v2/models/Prosit_2020_intensity_TMT/infer",
        "citation": " Wassim Gabriel, Matthew The, Daniel P. Zolg, Florian P. Bayer, et al. Analytical Chemistry 2022 94 (20), 7181-7190 https://doi.org/10.1021/acs.analchem.1c05435",
        "github_url": "https://github.com/kusterlab/prosit",
        "required_keys": [
            DATA_KEYS.PEPTIDE_SEQUENCE,
            DATA_KEYS.PRECURSOR_CHARGES,
            DATA_KEYS.COLLISION_ENERGIES,
            DATA_KEYS.FRAGMENTATION_TYPES,
        ],
    },
}
AVAILABLE_FORMATS = ["msp", "csv"]
