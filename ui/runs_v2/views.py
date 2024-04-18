from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from django.http import (
    HttpRequest,
    HttpResponseBadRequest,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import render
from django.urls import reverse

from protzilla.run_helper import log_messages
from protzilla.run_v2 import Run, get_available_run_names
from protzilla.stepfactory import StepFactory
from protzilla.utilities.utilities import get_memory_usage, name_to_title
from protzilla.workflow import get_available_workflow_names
from ui.runs_v2.fields import make_displayed_history, make_method_dropdown, make_sidebar
from ui.runs_v2.views_helper import display_messages, parameters_from_post
from .form_mapping import (
    get_empty_form_by_method,
    get_empty_plot_form_by_method,
    get_filled_form_by_request,
)

active_runs: dict[str, Run] = {}


def detail(request: HttpRequest, run_name: str):
    """
    Renders the details page of a specific run.
    For rendering a context dict is created that contains all the dynamic information
    that is needed to display the page. This wraps other methods that provide subparts
    for the page e.g. make_displayed_history() to show the history.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered details page
    :rtype: HttpResponse
    """
    if run_name not in active_runs:
        active_runs[run_name] = Run(run_name)
    run: Run = active_runs[run_name]

    # section, step, method = run.current_run_location()
    # end_of_run = not step

    if request.POST:
        method_form = get_filled_form_by_request(request, run)
        if method_form.is_valid():
            method_form.submit(run)
        plot_form = get_empty_plot_form_by_method(run.current_step, run)
    else:
        method_form = get_empty_form_by_method(run.current_step, run)
        plot_form = get_empty_plot_form_by_method(run.current_step, run)

    description = run.current_step.method_description

    log_messages(run.current_step.messages)
    display_messages(run.current_messages, request)

    current_plots = []
    for plot in run.current_plots:
        if isinstance(plot, bytes):
            # Base64 encoded image
            current_plots.append(
                '<div class="row d-flex justify-content-center mb-4"><img src="data:image/png;base64, {}"></div>'.format(
                    plot.decode("utf-8")
                )
            )
        elif isinstance(plot, dict):
            if "plot_base64" in plot:
                current_plots.append(
                    '<div class="row d-flex justify-content-center mb-4"><img id="{}" src="data:image/png;base64, {}"></div>'.format(
                        plot["key"], plot["plot_base64"].decode("utf-8")
                    )
                )
            else:
                current_plots.append(None)
        else:
            current_plots.append(plot.to_html(include_plotlyjs=False, full_html=False))

    show_table = not run.current_outputs.is_empty and any(
        isinstance(v, pd.DataFrame) for _, v in run.current_outputs
    )

    show_protein_graph = (
        run.current_outputs
        and "graph_path" in run.current_outputs
        and run.current_outputs["graph_path"] is not None
        and Path(run.current_outputs["graph_path"]).exists()
    )

    return render(
        request,
        "runs_v2/details.html",
        context=dict(
            run_name=run_name,
            section=run.current_step.section,
            step=run.current_step,
            display_name=f"{name_to_title(run.current_step.section)} - {run.current_step.display_name}",
            displayed_history=make_displayed_history(
                run
            ),  # TODO: make NewRun compatible
            method_dropdown=make_method_dropdown(
                run.run_name,
                run.current_step.section,
                run.current_step.operation,
                type(run.current_step).__name__,
            ),
            name_field="",
            current_plots=current_plots,
            results_exist=not run.current_step.output.is_empty,
            show_back=run.steps.current_step_index > 0,
            show_plot_button=not run.current_step.output.is_empty,
            # TODO include plot exists and plot parameters match current plot or remove this and replace with results exist
            sidebar=make_sidebar(request, run),
            last_step=run.steps.current_step_index == len(run.steps.all_steps) - 1,
            end_of_run=False,  # TODO?
            show_table=show_table,
            used_memory=get_memory_usage(),
            show_protein_graph=show_protein_graph,
            description=description,
            method_form=method_form,
            plot_form=plot_form,
        ),
    )


def index(request: HttpRequest):
    """
    Renders the main index page of the PROTzilla application.

    :param request: the request object
    :type request: HttpRequest

    :return: the rendered index page
    :rtype: HttpResponse
    """
    return render(
        request,
        "runs_v2/index.html",
        context={
            "available_workflows": get_available_workflow_names(),
            "available_runs": get_available_run_names(),
        },
    )


# TODO: make NewRun compatible
def create(request: HttpRequest):
    """
    Creates a new run. The user is then redirected to the detail page of the run.

    :param request: the request object
    :type request: HttpRequest

    :return: the rendered details page of the new run
    :rtype: HttpResponse
    """
    run_name = request.POST["run_name"]
    run = Run(
        run_name,
        request.POST["workflow_config_name"],
        df_mode=request.POST["df_mode"],
    )
    active_runs[run_name] = run
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def continue_(request: HttpRequest):
    """
    Continues an existing run. The user is redirected to the detail page of the run and
    can resume working on the run.

    :param request: the request object
    :type request: HttpRequest

    :return: the rendered details page of the run
    :rtype: HttpResponse
    """
    run_name = request.POST["run_name"]
    active_runs[run_name] = Run(run_name)

    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


# this function is no longer used, as the Output instance of a step has a utility method not_empty
def results_exist(run: Run) -> bool:
    """
    Checks if the last step has produced valid results.

    :param run: the run to check

    :return: True if the results are valid, False otherwise
    """
    if run.section == "importing":
        return run.result_df is not None or (run.step == "plot" and run.plots)
    if run.section == "data_preprocessing":
        return run.result_df is not None or (run.step == "plot" and run.plots)
    if run.section == "data_analysis" or run.section == "data_integration":
        return run.calculated_method is not None or (run.step == "plot" and run.plots)
    return True


def next_(request, run_name):
    """
    Skips to and renders the next step/method of the run.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run with the next step/method
    :rtype: HttpResponse
    """
    run = active_runs[run_name]
    run.step_next()
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def back(request, run_name):
    """
    Goes back to and renders the previous step/method of the run.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run with the previous step/method
    :rtype: HttpResponse
    """
    run = active_runs[run_name]
    run.step_previous()
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def plot(request, run_name):
    """
    Creates a plot from the current step/method of the run.
    This is only called by the plot button in the data preprocessing section aka when a plot is
    simultaneously a step on its own.
    Django messages are used to display additional information, warnings and errors to the user.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run, now with the plot
    :rtype: HttpResponse
    """
    run = active_runs[run_name]
    parameters = parameters_from_post(request.POST)

    if run.current_step.display_name == "plot":
        del parameters["chosen_method"]
        run.step_calculate(parameters)
    else:
        run.current_step.plot(parameters)

    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def tables(request, run_name, index, key=None):
    if run_name not in active_runs:
        active_runs[run_name] = Run(run_name)
    run = active_runs[run_name]

    # TODO this will change with the update to df_mode
    # use current output when applicable (not yet in history)
    if index < len(run.steps.previous_steps):
        outputs = run.steps.previous_steps[index].output
        section = run.steps.previous_steps[index].section
        step = run.steps.previous_steps[index].operation
        method = run.steps.previous_steps[index].display_name
    else:
        outputs = run.current_outputs
        section = run.current_step.section
        step = run.current_step.operation
        method = run.current_step.display_name

    options = []
    for k, value in outputs:
        if isinstance(value, pd.DataFrame) and k != key:
            options.append(k)

    if key is None and options:
        # choose an option if url without key is used
        return HttpResponseRedirect(
            reverse("runs_v2:tables", args=(run_name, index, options[0]))
        )

    return render(
        request,
        "runs_v2/tables.html",
        context=dict(
            run_name=run_name,
            index=index,
            # put key as first option to make selected
            options=[(opt, opt) for opt in [key] + options],
            key=key,
            section=section,
            step=step,
            method=method,
            clean_ids="clean-ids" if "clean-ids" in request.GET else "",
        ),
    )


def add(request: HttpRequest, run_name: str):
    """
    Adds a new method to the run. The method is added as the next step.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run, new method visible in sidebar
    :rtype: HttpResponse
    """
    run = active_runs[run_name]
    method = dict(request.POST)["method"][0]

    step = StepFactory.create_step(method)
    run.step_add(step)
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def export_workflow(request: HttpRequest, run_name: str):
    """
    Exports the workflow of the run as a JSON file.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run
    :rtype: HttpResponse
    """
    run = active_runs[run_name]
    requested_workflow_name = request.POST["name"]
    run._workflow_export(requested_workflow_name)
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def download_plots(request: HttpRequest, run_name: str):
    # TODO: implement
    raise NotImplementedError("Downloading workflows is not yet implemented.")


def delete_step(request: HttpRequest, run_name: str):
    """
    Deletes a step/method from the run.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: the rendered detail page of the run, deleted method no longer visible in sidebar
    :rtype: HttpResponse
    """
    run = active_runs[run_name]

    post = dict(request.POST)
    index = int(post["index"][0])

    run.step_remove(step_index=index)
    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def navigate(request, run_name):
    raise NotImplementedError("Navigating to specific steps is not yet implemented.")


def tables_content(request, run_name, index, key):
    run = active_runs[run_name]
    # TODO this will change with df_mode implementation
    if index < len(run.steps.previous_steps):
        outputs = run.steps.previous_steps[index].output[key]
    else:
        outputs = run.current_outputs[key]
    out = outputs.replace(np.nan, None)

    if "clean-ids" in request.GET:
        for column in out.columns:
            if "protein" in column.lower():
                out[column] = out[column].map(
                    lambda group: ";".join(
                        unique_justseen(map(clean_uniprot_id, group.split(";")))
                    )
                )
    return JsonResponse(
        dict(columns=out.to_dict("split")["columns"], data=out.to_dict("split")["data"])
    )


def change_method(request, run_name):
    """
    Changes the method during a step of a run.
    This is called when the user selects a new method in the first dropdown of a step.

    :param request: the request object
    :type request: HttpRequest
    :param run_name: the name of the run
    :type run_name: str

    :return: a JSON response object containing the new fields for the selected method
    :rtype: JsonResponse
    """

    if run_name not in active_runs:
        active_runs[run_name] = Run(run_name)
    run = active_runs[run_name]
    chosen_method = request.POST["chosen_method"]
    run.step_change_method(chosen_method)

    return HttpResponseRedirect(reverse("runs_v2:detail", args=(run_name,)))


def protein_graph(request, run_name, index: int):
    if run_name not in active_runs:
        active_runs[run_name] = Run(run_name)
    run = active_runs[run_name]

    outputs = run.current_outputs

    if "graph_path" not in outputs:
        return HttpResponseBadRequest(
            f"No Graph Path found in output of step with index {index}"
        )

    graph_path = outputs["graph_path"]
    peptide_matches = outputs.get("peptide_matches", [])
    peptide_mismatches = outputs.get("peptide_mismatches", [])
    protein_id = outputs.get("protein_id", "")

    if not Path(graph_path).exists():
        return HttpResponseBadRequest(f"Graph file {graph_path} does not exist")

    graph = nx.read_graphml(graph_path)

    max_peptides = 0
    min_peptides = 0
    nodes = []

    # count number of peptides for each node, set max_peptides and min_peptides
    for node in graph.nodes():
        peptides = graph.nodes[node].get("peptides", "")
        if peptides != "":
            peptides = peptides.split(";")
            if len(peptides) > max_peptides:
                max_peptides = len(peptides)
            elif len(peptides) < min_peptides:
                min_peptides = len(peptides)
        nodes.append(
            {
                "data": {
                    "id": node,
                    "label": graph.nodes[node].get(
                        "aminoacid", "####### ERROR #######"
                    ),
                    "match": graph.nodes[node].get("match", "false"),
                    "peptides": graph.nodes[node].get("peptides", ""),
                    "peptides_count": len(peptides),
                }
            }
        )

    edges = [{"data": {"source": u, "target": v}} for u, v in graph.edges()]
    elements = nodes + edges

    return render(
        request,
        "runs_v2/protein_graph.html",
        context={
            "elements": elements,
            "peptide_matches": peptide_matches,
            "peptide_mismatches": peptide_mismatches,
            "protein_id": protein_id,
            "filtered_blocks": outputs.get("filtered_blocks", []),
            "max_peptides": max_peptides,
            "min_peptides": min_peptides,
            "run_name": run_name,
            "used_memory": get_memory_usage(),
        },
    )
