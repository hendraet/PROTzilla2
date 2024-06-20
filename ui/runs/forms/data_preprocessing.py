from enum import Enum

from protzilla.run_v2 import Run

from . import fill_helper
from .base import MethodForm, PlotForm
from .custom_fields import (
    CustomCharField,
    CustomChoiceField,
    CustomFloatField,
    CustomMultipleChoiceField,
    CustomNumberField,
)


class EmptyEnum(Enum):
    pass


class LogTransformationBaseType(Enum):
    log2 = "log2"
    log10 = "log10"


class SimpleImputerStrategyType(Enum):
    mean = "mean"
    median = "median"
    most_frequent = "most_frequent"


class ImputationByNormalDistributionSamplingStrategyType(Enum):
    per_protein = "perProtein"
    per_dataset = "perDataset"


class BarAndPieChart(Enum):
    bar_plot = "Bar chart"
    pie_chart = "Pie chart"


class BoxAndHistogramGraph(Enum):
    boxplot = "Boxplot"
    histogram = "Histogram"


class GroupBy(Enum):
    no_grouping = "None"
    sample = "Sample"
    protein_id = "Protein ID"


class VisualTrasformations(Enum):
    log10 = "log10"
    linear = "linear"


class VisulaTransformations(Enum):
    linear = "linear"
    log10 = "log10"


class FilterProteinsBySamplesMissingForm(MethodForm):
    percentage = CustomFloatField(
        label="Percentage of minimum non-missing samples per protein",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class FilterProteinsBySamplesMissingPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type",
        initial=BarAndPieChart.pie_chart,
    )


class FilterByProteinsCountForm(MethodForm):
    deviation_threshold = CustomNumberField(
        label="Number of standard deviations from the median",
        min_value=0,
        initial=2,
    )


class FilterByProteinsCountPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type",
        initial=BarAndPieChart.pie_chart,
    )


class FilterSamplesByProteinsMissingForm(MethodForm):
    percentage = CustomFloatField(
        label="Percentage of minimum non-missing proteins per sample",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class FilterSamplesByProteinsMissingPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type",
        initial=BarAndPieChart.pie_chart,
    )


class FilterSamplesByProteinIntensitiesSumForm(MethodForm):
    deviation_threshold = CustomFloatField(
        label="Number of standard deviations from the median:",
        min_value=0,
        initial=2,
    )


class FilterSamplesByProteinIntensitiesSumPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type",
        initial=BarAndPieChart.pie_chart,
    )


class OutlierDetectionByPCAForm(MethodForm):
    threshold = CustomFloatField(
        label="Threshold for number of standard deviations from the median:",
        min_value=0,
        initial=2,
    )
    number_of_components = CustomNumberField(
        label="Number of components",
        min_value=2,
        max_value=3,
        step_size=1,
        initial=3,
    )


class OutlierDetectionByPCAPlotForm(PlotForm):
    pass


class OutlierDetectionByIsolationForestForm(MethodForm):
    n_estimators = CustomNumberField(
        label="Number of estimators",
        min_value=1,
        step_size=1,
        initial=100,
    )


class OutlierDetectionByIsolationForestPlotForm(PlotForm):
    pass


class OutlierDetectionByLocalOutlierFactorForm(MethodForm):
    number_of_neighbors = CustomNumberField(
        label="Number of neighbours",
        min_value=1,
        step_size=1,
        initial=20,
    )


class OutlierDetectionByLocalOutlierFactorPlotForm(PlotForm):
    pass


class TransformationLogForm(MethodForm):
    log_base = CustomChoiceField(
        choices=LogTransformationBaseType,
        label="Log transformation base:",
        initial=LogTransformationBaseType.log2,
    )


class TransformationLogPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class NormalisationByZScoreForm(MethodForm):
    pass


class NormalisationByZscorePlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class NormalisationByTotalSumForm(MethodForm):
    pass


class NormalisationByTotalSumPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class NormalisationByMedianForm(MethodForm):
    percentile = CustomFloatField(
        label="Percentile for normalisation:",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class NormalisationByMedianPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class NormalisationByReferenceProteinForms(MethodForm):
    reference_protein = CustomCharField(
        label="A function to perform protein-intensity normalisation in reference to "
        "a selected protein on your dataframe. Normalises the data on the level "
        "of each sample. Divides each intensity by the intensity of the chosen "
        "reference protein in each sample. Samples where this value is zero "
        "will be removed and returned separately.A function to perform "
        "protein-intensity normalisation in reference to a selected protein on "
        "your dataframe. Normalises the data on the level of each sample. "
        "Divides each intensity by the intensity of the chosen reference "
        "protein in each sample. Samples where this value is zero will be "
        "removed and returned separately."
    )


class NormalisationByReferenceProteinPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class ImputationByMinPerDatasetForm(MethodForm):
    shrinking_value = CustomNumberField(
        label="A function to impute missing values for each protein by taking into account "
        "data from the entire dataframe. Sets missing value to the smallest measured "
        "value in the dataframe. The user can also assign a shrinking factor to take a "
        "fraction of that minimum value for imputation.",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class ImputationByMinPerDatasetPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class ImputationByMinPerProteinForm(MethodForm):
    shrinking_value = CustomFloatField(
        label="A function to impute missing values for each protein by taking into account data from each protein. "
        "Sets missing value to the smallest measured value for each protein column. The user can also assign a "
        "shrinking factor to take a fraction of that minimum value for imputation. CAVE: All proteins without "
        "any values will be filtered out.",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class ImputationByMinPerProteinPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)

class ImputationByMinPerSampleForms(MethodForm):
    shrinking_value = CustomFloatField(
        label="Sets missing intensity values to the smallest measured value for each sample",
        min_value=0,
        max_value=1,
        step_size=0.1,
        initial=0.5,
    )


class ImputationByMinPerSamplePlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class SimpleImputationPerProteinForm(MethodForm):
    strategy = CustomChoiceField(
        choices=SimpleImputerStrategyType,
        label="Strategy",
        initial=SimpleImputerStrategyType.mean,
    )


class SimpleImputationPerProteinPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class ImputationByKNNForms(MethodForm):
    number_of_neighbours = CustomNumberField(
        label="Number of neighbours",
        min_value=1,
        step_size=1,
        initial=5,
    )


class ImputationByKNNPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)


class ImputationByNormalDistributionSamplingForm(MethodForm):
    strategy = CustomChoiceField(
        choices=ImputationByNormalDistributionSamplingStrategyType,
        label="Strategy",
        initial=ImputationByNormalDistributionSamplingStrategyType.per_protein,
    )
    down_shift = CustomNumberField(
        label="Downshift", min_value=-10, max_value=10, initial=-1
    )
    scaling_factor = CustomFloatField(
        label="Scaling factor", min_value=0, max_value=1, step_size=0.1, initial=0.5
    )


class ImputationByNormalDistributionSamplingPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BoxAndHistogramGraph,
        label="Graph type",
        initial=BoxAndHistogramGraph.boxplot,
    )
    group_by = CustomChoiceField(
        choices=GroupBy, label="Group by", initial=GroupBy.no_grouping
    )
    visual_transformation = CustomChoiceField(
        choices=VisualTrasformations,
        label="Visual transformation",
        initial=VisualTrasformations.log10,
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (will be highlighted)",
    )
    proteins_of_interest = CustomMultipleChoiceField(
        choices=[],
        label="Proteins of interest (By default all proteins are selected)",
    )
    graph_type_quantities = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type - quantity of imputed values",
        initial=BarAndPieChart.pie_chart,
    )

    def fill_form(self, run: Run) -> None:
        proteins = run.steps.protein_df["Protein ID"].unique()

        self.fields["proteins_of_interest"].choices = fill_helper.to_choices(proteins)

class FilterPeptidesByPEPThresholdForm(MethodForm):
    threshold = CustomFloatField(
        label="Threshold value for PEP", min_value=0, initial=0
    )
    peptide_df = CustomChoiceField(choices=EmptyEnum, label="peptide_df")


class FilterPeptidesByPEPThresholdPlotForm(PlotForm):
    graph_type = CustomChoiceField(
        choices=BarAndPieChart,
        label="Graph type",
        initial=BarAndPieChart.pie_chart,
    )
