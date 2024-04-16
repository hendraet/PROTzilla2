from __future__ import annotations

from protzilla.data_preprocessing import imputation
from protzilla.steps import Step, StepManager


class DataPreprocessingStep(Step):
    section = "data_preprocessing"
    output_names = ["protein_df"]

    def get_input_dataframe(self, steps: StepManager, kwargs: dict) -> dict:
        kwargs["protein_df"] = steps.protein_df
        return kwargs


class FilterProteinsBySamplesMissing(DataPreprocessingStep):
    name = "By samples missing"
    step = "filter_proteins"
    method_description = (
        "Filter proteins based on the amount of samples with nan values"
    )

    parameter_names = ["percentage"]

    def method(self, **kwargs):
        return filter_proteins.by_samples_missing(**kwargs)


class FilterSamplesByProteinsMissing(DataPreprocessingStep):
    name = "By proteins missing"
    step = "filter_samples"
    method_description = (
        "Filter samples based on the amount of proteins with nan values"
    )

    parameter_names = ["percentage"]

    def method(self, **kwargs):
        return filter_samples.by_proteins_missing(**kwargs)


class OutlierDetectionByPCA(DataPreprocessingStep):
    name = "PCA"
    step = "outlier_detection"
    method_description = "Detect outliers using PCA"

    parameter_names = ["number_of_components", "threshold"]

    def method(self, **kwargs):
        return outlier_detection.by_pca(**kwargs)


class OutlierDetectionByLocalOutlierFactor(DataPreprocessingStep):
    name = "LOF"
    step = "outlier_detection"
    method_description = "Detect outliers using LOF"

    parameter_names = ["number_of_neighbors", "n_jobs"]

    def method(self, **kwargs):
        return outlier_detection.by_lof(**kwargs)


class OutlierDetectionByIsolationForest(DataPreprocessingStep):
    name = "Isolation Forest"
    step = "outlier_detection"
    method_description = "Detect outliers using Isolation Forest"

    parameter_names = ["n_estimators", "n_jobs"]

    def method(self, **kwargs):
        return outlier_detection.by_isolation_forest(**kwargs)


class TransformationLog(DataPreprocessingStep):
    name = "Log"
    step = "transformation"
    method_description = "Transform data by log"

    parameter_names = ["log_base"]

    def method(self, **kwargs):
        return transformation.by_log(**kwargs)


class NormalisationByZScore(DataPreprocessingStep):
    name = "Z-Score"
    step = "normalisation"
    method_description = "Normalise data by Z-Score"

    parameter_names = []

    def method(self, **kwargs):
        return normalisation.by_z_score(**kwargs)


class NormalisationByTotalSum(DataPreprocessingStep):
    name = "Total sum"
    step = "normalisation"
    method_description = "Normalise data by total sum"

    parameter_names = []

    def method(self, **kwargs):
        return normalisation.by_totalsum(**kwargs)


class NormalisationByMedian(DataPreprocessingStep):
    name = "Median"
    step = "normalisation"
    method_description = "Normalise data by median"

    parameter_names = ["percentile"]

    def method(self, **kwargs):
        return normalisation.by_median(**kwargs)


class NormalisationByReferenceProtein(DataPreprocessingStep):
    name = "Reference protein"
    step = "normalisation"
    method_description = "Normalise data by reference protein"

    parameter_names = ["reference_protein"]

    def method(self, **kwargs):
        return normalisation.by_reference_protein(**kwargs)


class ImputationByMinPerDataset(DataPreprocessingStep):
    name = "Min per dataset"
    step = "imputation"
    method_description = "Impute missing values by the minimum per dataset"

    parameter_names = ["shrinking_value"]

    def method(self, **kwargs):
        return imputation.by_min_per_dataset(**kwargs)


class ImputationByMinPerProtein(DataPreprocessingStep):
    name = "Min per protein"
    step = "imputation"
    method_description = "Impute missing values by the minimum per protein"

    parameter_names = ["shrinking_value"]

    def method(self, **kwargs):
        return imputation.by_min_per_protein(**kwargs)


class ImputationByMinPerSample(DataPreprocessingStep):
    name = "Min per sample"
    step = "imputation"
    method_description = "Impute missing values by the minimum per sample"

    parameter_names = ["shrinking_value"]

    def method(self, **kwargs):
        return by_min_per_protein(**kwargs)
