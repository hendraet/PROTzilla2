from __future__ import annotations

import base64
import logging
from io import BytesIO

import pandas as pd
import plotly
from PIL import Image


class Step:
    def __init__(self):
        self.inputs: dict = {}
        self.messages: Messages = Messages([])
        self.output: Output = Output()
        self.plots = []
        self.parameter_names = []
        self.output_names = []
        self.finished = False

    def __repr__(self):
        return self.__class__.__name__

    def calculate(self, steps: StepManager, inputs: dict = None):
        if inputs is not None:
            self.inputs = inputs

        # validate the inputs for the step
        self.validate_inputs(self.parameter_names)

        # calculate the step
        output_dict = self.method(self.get_input_dataframe(steps), **self.inputs)

        # store the output and messages
        messages = output_dict.pop("messages", [])
        self.messages = Messages(messages)
        plots = output_dict.pop("plots", [])
        self.plots = Plots(plots)
        self.handle_outputs(output_dict)

        # validate the output
        self.finished = self.valid_outputs(self.output_names)

    def method(self, dataframe: pd.DataFrame, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass.")

    def get_input_dataframe(self, steps: StepManager) -> pd.DataFrame | None:
        return None

    def handle_outputs(self, output_dict: dict):
        self.output = Output(output_dict)

    def validate_inputs(self, required_keys: list[str]):
        for key in required_keys:
            if key not in self.inputs:
                raise ValueError(f"Missing input {key} in inputs")

    def valid_outputs(self, required_keys: list[str]) -> bool:
        for key in required_keys:
            if key not in self.output.output:
                return False
        return True


class DataAnalysisStep(Step):
    section = "data_analysis"

    def get_input_dataframe(self, steps: StepManager):
        protein_df
        return NotImplementedError("This method must be implemented in a subclass.")


class Output:
    def __init__(self, output: dict = None):
        if output is None:
            output = {}

        self.output = output

    def __iter__(self):
        return iter(self.output.items())

    def __getitem__(self, key):
        return self.output[key]

    @property
    def intensity_df(self):
        if "intensity_df" in self.output:
            return self.output["intensity_df"]
        else:
            return None

    @property
    def is_empty(self):
        return len(self.output) == 0 or all(
            value is None for value in self.output.values()
        )


class Messages:
    def __init__(self, messages: list[dict] = None):
        if messages is None:
            messages = []
        self.messages = messages

        def __iter__(self):
            return iter(self.messages)


class Plots:
    def __init__(self, plots: list = None):
        if plots is None:
            plots = []
        self.plots = plots

    def __iter__(self):
        return iter(self.plots)

    def export(self, format_):
        exports = []
        for plot in self.plots:
            if isinstance(plot, plotly.graph_objs.Figure):
                if format_ in ["eps", "tiff"]:
                    png_binary = plotly.io.to_image(plot, format="png", scale=4)
                    img = Image.open(BytesIO(png_binary)).convert("RGB")
                    binary = BytesIO()
                    if format_ == "tiff":
                        img.save(binary, format="tiff", compression="tiff_lzw")
                    else:
                        img.save(binary, format=format_)
                    exports.append(binary)
                else:
                    binary_string = plotly.io.to_image(plot, format=format_, scale=4)
                    exports.append(BytesIO(binary_string))
            elif isinstance(plot, dict) and "plot_base64" in plot:
                plot = plot["plot_base64"]

            if isinstance(plot, bytes):  # base64 encoded plots
                if format_ in ["eps", "tiff"]:
                    img = Image.open(BytesIO(base64.b64decode(plot))).convert("RGB")
                    binary = BytesIO()
                    if format_ == "tiff":
                        img.save(binary, format="tiff", compression="tiff_lzw")
                    else:
                        img.save(binary, format=format_)
                    binary.seek(0)
                    exports.append(binary)
                elif format_ in ["png", "jpg"]:
                    exports.append(BytesIO(base64.b64decode(plot)))
        return exports


class StepManager:
    def __repr__(self):
        return f"Importing: {self.importing}\nData Preprocessing: {self.data_preprocessing}\nData Analysis: {self.data_analysis}\nData Integration: {self.data_integration}"

    def __init__(self, steps: list[Step] = None):
        self.importing = []
        self.data_preprocessing = []
        self.data_analysis = []
        self.data_integration = []
        self.current_step_index = 0

        if steps is not None:
            for step in steps:
                self.add_step(step)

    @property
    def all_steps(self):
        return (
            self.importing
            + self.data_preprocessing
            + self.data_analysis
            + self.data_integration
        )

    def all_steps_in_section(self, section: str):
        if section == "importing":
            return self.importing
        elif section == "data_preprocessing":
            return self.data_preprocessing
        elif section == "data_analysis":
            return self.data_analysis
        elif section == "data_integration":
            return self.data_integration
        else:
            raise ValueError(f"Unknown section {section}")

    @property
    def previous_steps(self):
        return self.all_steps[: self.current_step_index]

    @property
    def current_step(self) -> Step:
        return self.all_steps[self.current_step_index]

    def current_section(self) -> str:
        return self.current_step.section

    @property
    def protein_df(self):
        # find the last step that has an intensity_df
        for step in reversed(self.all_steps):
            if step.output.protein_df is not None:
                return step.output.protein_df
        logging.warning("No intensity_df found in steps")

    @property
    def metadata_df(self):
        # find the last step that has a metadata_df
        for step in reversed(self.all_steps):
            if hasattr(step.output, "metadata_df"):
                return step.output.metadata_df
        logging.warning("No metadata_df found in steps")

    @property
    def is_at_last_step(self):
        return self.current_step_index == len(self.all_steps) - 1

    def add_step(self, step, index: int | None = None):
        # TODO add support for index
        if step.section == "importing":
            self.importing.append(step)
        elif step.section == "data_preprocessing":
            self.data_preprocessing.append(step)
        elif step.section == "data_analysis":
            self.data_analysis.append(step)
        elif step.section == "data_integration":
            self.data_integration.append(step)
        else:
            raise ValueError(f"Unknown section {step.section}")

    def remove_step(self, step: Step, step_index: int = None):
        if step_index is not None:
            if step_index < self.current_step_index:
                self.current_step_index -= 1
            step = self.all_steps[step_index]

        for section in [
            self.importing,
            self.data_preprocessing,
            self.data_analysis,
            self.data_integration,
        ]:
            try:
                section.remove(step)
                return
            except ValueError:
                pass

        raise ValueError(f"Step {step} not found in steps")
