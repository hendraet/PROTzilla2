import sys

from django.template.loader import render_to_string
from main.settings import BASE_DIR

sys.path.append(f"{BASE_DIR}/..")
from protzilla.workflow_helper import (
    get_workflow_default_param_value,
)
from ui.runs.views_helper import insert_special_params, get_displayed_steps


def make_current_fields(run, section, step, method):
    parameters = run.workflow_meta[section][step][method]["parameters"]
    current_fields = []
    for key, param_dict in parameters.items():
        # todo 59 - restructure current_parameters
        param_dict = param_dict.copy()  # to not change workflow_meta
        workflow_default = get_workflow_default_param_value(
            run.workflow_config, section, step, method, key
        )
        if workflow_default is not None:
            param_dict["default"] = workflow_default
        elif run.current_parameters is not None:
            param_dict["default"] = run.current_parameters[key]

        insert_special_params(param_dict, run)
        current_fields.append(make_parameter_input(key, param_dict, disabled=False))

    return current_fields


def make_parameter_input(key, param_dict, disabled):
    if param_dict["type"] == "numeric":
        template = "runs/field_number.html"
        if "step" not in param_dict:
            param_dict["step"] = "any"
    elif param_dict["type"] == "categorical":
        param_dict["multiple"] = param_dict.get("multiple", False)
        template = "runs/field_select.html"
    elif param_dict["type"] == "file":
        template = "runs/field_file.html"
    elif param_dict["type"] == "named_output":
        template = "runs/field_named.html"
    elif param_dict["type"] == "metadata_df":
        template = "runs/field_empty.html"
    else:
        raise ValueError(f"cannot match parameter type {param_dict['type']}")

    return render_to_string(
        template,
        context=dict(
            **param_dict,
            disabled=disabled,
            key=key,
        ),
    )


def make_sidebar(request, run, run_name):
    csrf_token = request.META["CSRF_COOKIE"]
    template = "runs/sidebar.html"
    return render_to_string(
        template,
        context=dict(
            csrf_token=csrf_token,
            workflow_steps=get_displayed_steps(
                run.workflow_config, run.workflow_meta, run.step_index
            ),
            run_name=run_name,
        ),
    )


def make_plot_fields(run, section, step, method):
    plots = run.workflow_meta[section][step][method].get("graphs", [])
    plot_fields = []
    for plot in plots:
        for key, param_dict in plot.items():
            if run.current_plot_parameters is not None:
                param_dict["default"] = run.current_plot_parameters[key]
            plot_fields.append(make_parameter_input(key, param_dict, disabled=False))
    return plot_fields


def make_method_dropdown(run, section, step, method):
    return render_to_string(
        "runs/field_select.html",
        context=dict(
            disabled=False,
            key="chosen_method",
            name=f"{step.replace('_', ' ').title()} Method:",
            default=method,
            categories=run.workflow_meta[section][step].keys(),
        ),
    )


def make_displayed_history(run):
    displayed_history = []
    for i, history_step in enumerate(run.history.steps):
        fields = []
        # should parameters be copied, so workflow_meta won't change?
        parameters = run.workflow_meta[history_step.section][history_step.step][
            history_step.method
        ]["parameters"]
        name = f"{history_step.step.replace('_', ' ').title()}: {history_step.method.replace('_', ' ').title()}"
        section_heading = (
            history_step.section.replace("_", " ").title()
            if run.history.steps[i - 1].section != history_step.section
            else None
        )
        if history_step.section == "importing":
            fields = [""]
        else:
            for key, param_dict in parameters.items():
                param_dict["default"] = history_step.parameters[key]
                if param_dict["type"] == "named_output":
                    param_dict["steps"] = [param_dict["default"][0]]
                    param_dict["outputs"] = [param_dict["default"][1]]
                fields.append(make_parameter_input(key, param_dict, disabled=True))
        displayed_history.append(
            dict(
                display_name=name,
                fields=fields,
                plots=[p.to_html() for p in history_step.plots],
                section_heading=section_heading,
                name=run.history.step_names[i],
                index=i,
            )
        )
    return displayed_history


def make_name_field(allow_next, form):
    return render_to_string(
        "runs/field_text.html",
        context=dict(disabled=not allow_next, key="name", name="Name:", form=form),
    )
