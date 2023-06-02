import sys

import pandas
from django.template.loader import render_to_string
from django.urls import reverse
from main.settings import BASE_DIR

sys.path.append(f"{BASE_DIR}/..")
from protzilla.run_helper import get_parameters, insert_special_params
from protzilla.workflow_helper import get_workflow_default_param_value
from ui.runs.views_helper import get_displayed_steps


def make_current_fields(run, section, step, method):
    if not step:
        return []
    parameters = get_parameters(run, section, step, method)
    current_fields = []
    for key, param_dict in parameters.items():
        if "dynamic" in param_dict:
            continue
        current_fields.append(
            make_parameter_input(key, param_dict, parameters, disabled=False)
        )

    return current_fields


def make_parameter_input(key, param_dict, all_parameters_dict, disabled):
    # In this method param_dict refers to the dictionary that contains all
    # meta information about a specific parameter e.g. type, default value. The
    # all_parameters_dict refers to the dictionary that contains all parameters for
    # a method with its corresponding meta information
    if param_dict["type"] == "numeric":
        template = "runs/field_number.html"
        if "step" not in param_dict:
            param_dict["step"] = "any"
    elif param_dict["type"] == "categorical":
        param_dict["multiple"] = param_dict.get("multiple", False)
        template = "runs/field_select.html"
    elif param_dict["type"] == "categorical_dynamic":
        template = "runs/field_select_dynamic.html"
        selected_category = param_dict["default"]
        dynamic_fields = make_dynamic_fields(
            param_dict, selected_category, all_parameters_dict, disabled
        )
        param_dict["dynamic_fields"] = dynamic_fields
    elif param_dict["type"] == "file":
        template = "runs/field_file.html"
    elif param_dict["type"] == "named_output":
        template = "runs/field_named.html"
    elif param_dict["type"] == "empty":
        template = "runs/field_empty.html"
    elif param_dict["type"] == "text":
        template = "runs/field_text.html"
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


def make_dynamic_fields(param_dict, selected_category, all_parameters_dict, disabled):
    dynamic_fields = []
    if selected_category in param_dict["dynamic_parameters"]:
        dynamic_parameters_list = param_dict["dynamic_parameters"][selected_category]
        for field_key in dynamic_parameters_list:
            field_dict = all_parameters_dict[field_key]
            dynamic_fields.append(
                make_parameter_input(
                    field_key, field_dict, all_parameters_dict, disabled
                )
            )
    return dynamic_fields


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
    if not step:
        return
    plots = run.workflow_meta[section][step][method].get("graphs", [])
    plot_fields = []
    for plot in plots:
        for key, param_dict in plot.items():
            if method in run.current_plot_parameters:
                param_dict["default"] = run.current_plot_parameters[method][key]
            plot_fields.append(
                make_parameter_input(key, param_dict, plot, disabled=False)
            )
    return plot_fields


def make_method_dropdown(run, section, step, method):
    if not step:
        return ""
    methods = run.workflow_meta[section][step].keys()
    method_names = [run.workflow_meta[section][step][key]["name"] for key in methods]

    return render_to_string(
        "runs/field_select_with_label.html",
        context=dict(
            disabled=False,
            key="chosen_method",
            name=f"{step.replace('_', ' ').title()} Method:",
            default=method,
            categories=list(zip(methods, method_names)),
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
                if key.endswith("_wrapper"):
                    key = key[:-8]
                if key == "proteins_of_interest" and key not in history_step.parameters:
                    history_step.parameters[key] = ["", ""]
                param_dict["default"] = history_step.parameters[key]
                if param_dict["type"] == "named_output":
                    param_dict["steps"] = [param_dict["default"][0]]
                    param_dict["outputs"] = [param_dict["default"][1]]
                fields.append(
                    make_parameter_input(key, param_dict, parameters, disabled=True)
                )

        plots = []
        for plot in history_step.plots:
            if isinstance(plot, bytes):
                # Base64 encoded image
                plots.append(
                    '<div class="row d-flex justify-content-between align-items-center mb-4"><img src="data:image/png;base64, {}"></div>'.format(
                        plot.decode("utf-8")
                    )
                )
            elif isinstance(plot, dict):
                plots.append(None)
            else:
                plots.append(plot.to_html(include_plotlyjs=False, full_html=False))

        has_df = any(
            isinstance(v, pandas.DataFrame) for v in history_step.outputs.values()
        )
        table_url = reverse("runs:tables_nokey", args=(run.run_name, i))

        displayed_history.append(
            dict(
                display_name=name,
                fields=fields,
                plots=plots,
                section_heading=section_heading,
                name=run.history.step_names[i],
                index=i,
                table_link=table_url if has_df else "",
            )
        )
    return displayed_history


def make_name_field(allow_next, form, run, end_of_run):
    if end_of_run:
        return ""
    default = get_workflow_default_param_value(
        run.workflow_config, *run.current_run_location(), "output_name"
    )
    if not default:
        default = ""

    return render_to_string(
        "runs/field_name_output_text.html",
        context=dict(
            disabled=not allow_next,
            key="name",
            name="Name:",
            form=form,
            default=default,
        ),
    )
