from __future__ import annotations

import logging

from protzilla.data_preprocessing.imputation import by_min_per_protein
from protzilla.importing.metadata_import import metadata_import_method
from protzilla.importing.ms_data_import import max_quant_import


class Step:
    def __init__(self):
        self.inputs: dict = {}
        self.messages: Messages = Messages([])
        self.output: Output = Output()
        self.plots = []

    def __repr__(self):
        return self.__class__.__name__

    def calculate(self):
        raise NotImplementedError

    def validate_inputs(self, required_keys: list[str]):
        for key in required_keys:
            if key not in self.inputs:
                raise ValueError(f"Missing input {key} in inputs")

    def validate_outputs(self, required_keys: list[str]):
        for key in required_keys:
            if key not in self.output.output:
                raise ValueError(f"Missing output {key} in output")


class Output:
    def __iter__(self):
        return iter(self.output.items())

    def __init__(self, output: dict = None):
        if output is None:
            output = {}

        self.output = output

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
    def __iter__(self):
        return iter(self.messages)

    def __init__(self, messages: list[dict] = None):
        if messages is None:
            messages = []
        self.messages = messages


class MaxQuantImport(Step):
    name = "MaxQuant"
    section = "importing"
    step = "msdataimport"
    method = "max_quant_import"
    method_description = "Import MaxQuant data"

    def calculate(self, steps: StepManager, inputs: dict = None):
        if inputs is not None:
            self.inputs = inputs

        # validate the inputs for the step
        self.validate_inputs(["file_path", "map_to_uniprot", "intensity_name"])

        # calculate the step
        dataframe, output_dict = max_quant_import(None, **self.inputs)
        messages = output_dict.pop("messages")

        # store the output and messages
        self.output = Output({"intensity_df": dataframe} | output_dict)
        self.messages = Messages(messages)

        # validate the output
        self.validate_outputs(["intensity_df"])


class MetadataImport(Step):
    name = "Metadata import"
    section = "importing"
    step = "metadataimport"
    method = "metadata_import_method"
    method_description = "Import metadata"

    def calculate(self, steps: StepManager, inputs: dict = None):
        if inputs is not None:
            self.inputs = inputs

        # validate the inputs for the step
        self.validate_inputs(["file_path", "feature_orientation"])

        # calculate the step
        _, output_dict = metadata_import_method(steps.intensity_df, **self.inputs)

        # store the output and messages
        self.output = Output({"metadata_df": output_dict["metadata"]})
        self.messages = Messages(output_dict["messages"])

        # validate the output
        self.validate_outputs(["metadata_df"])


class ImputationMinPerProtein(Step):
    name = "Min per dataset"
    section = "data_preprocessing"
    step = "imputation"
    method = "by_min_per_protein"
    method_description = "Impute missing values by the minimum per protein"

    def calculate(self, steps: StepManager, inputs: dict = None):
        if inputs is not None:
            self.inputs = inputs

        # validate the inputs for the step
        self.validate_inputs(["shrinking_value"])
        if steps.intensity_df is None:
            raise ValueError("No data to impute")
        # calculate the step
        output, messages = by_min_per_protein(steps.intensity_df, **self.inputs)

        # store the output and messages
        self.output = Output({"intensity_df": output})
        self.messages = Messages(messages)

        # validate the output
        self.validate_outputs(["intensity_df"])


class StepFactory:
    # TODO this could be done with the new mapping class, iterating and checking whether there exists a step with that specific name
    @staticmethod
    def create_step(step_type: str) -> Step:
        if step_type == "MaxQuantImport" or step_type == "max_quant_import":
            return MaxQuantImport()
        elif step_type == "ImputationMinPerProtein":
            return ImputationMinPerProtein()
        elif step_type == "MetadataImport" or step_type == "metadata_import_method":
            return MetadataImport()
        else:
            raise ValueError(f"Unknown step type {step_type}")


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
    def intensity_df(self):
        # find the last step that has an intensity_df
        for step in reversed(self.all_steps):
            if step.output.intensity_df is not None:
                return step.output.intensity_df
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
            self.all_steps.pop(step_index)
        else:
            if step in self.all_steps:
                self.all_steps.remove(step)
            else:
                raise ValueError(f"Step {step} not found in steps")
