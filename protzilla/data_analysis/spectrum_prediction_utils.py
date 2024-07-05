from protzilla.data_analysis import spectrum_prediction
from ui.runs.forms.fill_helper import enclose_links_in_html


def format_citation(model_name):
    model_info = spectrum_prediction.AVAILABLE_MODELS[model_name]
    citation_string = (
        f"Info about {model_name}:<br>"
        f'{enclose_links_in_html(model_info.get("citation", ""))}<br>'
        f'{enclose_links_in_html(model_info["github_url"]) if model_info.get("github_url") else ""}<br>'
    )
    return citation_string
