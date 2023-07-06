import pandas as pd
import pytest
from matplotlib import pyplot as plt

from protzilla.data_analysis.model_evaluation import (
    evaluate_classification_model,
    permutation_testing,
)
from protzilla.data_analysis.classification import random_forest
from protzilla.data_analysis.model_evaluation import evaluate_classification_model
from protzilla.data_analysis.model_evaluation_plots import (
    precision_recall_curve_plot,
    roc_curve_plot,
    permutation_testing_plot,
)
from protzilla.data_analysis.model_selection import (
    compute_learning_curve,
    random_sampling,
    generate_stratified_subsets,
)
from protzilla.data_analysis.model_selection_plots import learning_curve_plot
from protzilla.utilities.transform_dfs import long_to_wide


@pytest.fixture
def classification_df():
    classification_list = (
        ["Sample1", "Protein1", "Gene1", 18],
        ["Sample1", "Protein2", "Gene1", 16],
        ["Sample1", "Protein3", "Gene1", 1],
        ["Sample2", "Protein1", "Gene1", 20],
        ["Sample2", "Protein2", "Gene1", 18],
        ["Sample2", "Protein3", "Gene1", 2],
        ["Sample3", "Protein1", "Gene1", 22],
        ["Sample3", "Protein2", "Gene1", 19],
        ["Sample3", "Protein3", "Gene1", 3],
        ["Sample4", "Protein1", "Gene1", 8],
        ["Sample4", "Protein2", "Gene1", 15],
        ["Sample4", "Protein3", "Gene1", 1],
        ["Sample5", "Protein1", "Gene1", 10],
        ["Sample5", "Protein2", "Gene1", 14],
        ["Sample5", "Protein3", "Gene1", 2],
        ["Sample6", "Protein1", "Gene1", 12],
        ["Sample6", "Protein2", "Gene1", 13],
        ["Sample6", "Protein3", "Gene1", 3],
        ["Sample7", "Protein1", "Gene1", 12],
        ["Sample7", "Protein2", "Gene1", 13],
        ["Sample7", "Protein3", "Gene1", 3],
        ["Sample8", "Protein1", "Gene1", 42],
        ["Sample8", "Protein2", "Gene1", 33],
        ["Sample8", "Protein3", "Gene1", 3],
        ["Sample9", "Protein1", "Gene1", 19],
        ["Sample9", "Protein2", "Gene1", 1],
        ["Sample9", "Protein3", "Gene1", 4],
    )

    classification_df = pd.DataFrame(
        data=classification_list,
        columns=["Sample", "Protein ID", "Gene", "Intensity"],
    )

    return classification_df


@pytest.fixture
def meta_df():
    meta_list = (
        ["Sample1", "AD"],
        ["Sample2", "AD"],
        ["Sample3", "AD"],
        ["Sample4", "CTR"],
        ["Sample5", "CTR"],
        ["Sample6", "CTR"],
        ["Sample7", "CTR"],
        ["Sample8", "AD"],
        ["Sample9", "CTR"],
    )
    meta_df = pd.DataFrame(
        data=meta_list,
        columns=["Sample", "Group"],
    )

    return meta_df


@pytest.fixture
def meta_numeric_df():
    meta_list = (
        ["Sample1", "1"],
        ["Sample2", "1"],
        ["Sample3", "1"],
        ["Sample4", "0"],
        ["Sample5", "0"],
        ["Sample6", "0"],
        ["Sample7", "0"],
        ["Sample8", "1"],
        ["Sample9", "0"],
    )
    meta_df = pd.DataFrame(
        data=meta_list,
        columns=["Sample", "Group"],
    )

    return meta_df


@pytest.fixture
def random_forest_out(
    classification_df,
    meta_df,
    validation_strategy="K-Fold",
    model_selection="Grid search",
):
    return random_forest(
        classification_df,
        meta_df,
        "Group",
        n_estimators=3,
        test_validate_split=0.20,
        model_selection=model_selection,
        validation_strategy=validation_strategy,
        random_state=42,
    )


@pytest.mark.parametrize(
    "validation_strategy,model_selection",
    [
        ("Manual", "Manual"),
        ("K-Fold", "Manual"),
        ("K-Fold", "Grid search"),
        ("K-Fold", "Randomized search"),
    ],
)
def test_random_forest_score(random_forest_out, validation_strategy, model_selection):
    model_evaluation_df = random_forest_out["model_evaluation_df"]
    assert (
        model_evaluation_df["mean_test_accuracy"].values[0] >= 0.8
    ), f"Failed with validation strategy {validation_strategy} and model selection strategy {model_selection}"


def test_model_evaluation_plots(show_figures, random_forest_out, helpers):
    recall_curve_base64 = precision_recall_curve_plot(
        random_forest_out["model"],
        random_forest_out["X_test_df"],
        random_forest_out["y_test_df"],
    )
    roc_curve_base64 = roc_curve_plot(
        random_forest_out["model"],
        random_forest_out["X_test_df"],
        random_forest_out["y_test_df"],
    )

    if show_figures:
        helpers.open_graph_from_base64(recall_curve_base64[0])
        helpers.open_graph_from_base64(roc_curve_base64[0])


def test_model_selection_plots(show_figures, classification_df, meta_df, helpers):
    lc_out = compute_learning_curve(
        clf_str="Random Forest",
        input_df=classification_df,
        metadata_df=meta_df,
        labels_column="Group",
        positive_label="AD",
        train_sizes=[8, 9, 10, 11],
        cross_validation_strategy="Nested CV",
        inner_cv="Repeated Stratified K-Fold",
        outer_cv="Repeated Stratified K-Fold",
        n_splits=3,
        n_repeats=2,
        shuffle="yes",
        scoring="accuracy",
        random_state=42,
    )
    curve_base64 = learning_curve_plot(
        train_sizes=lc_out["train_sizes"],
        train_scores=lc_out["train_scores"],
        test_scores=lc_out["test_scores"],
        score_name="Accuracy",
        minimum_viable_sample_size=lc_out["minimum_viable_sample_size"],
    )
    if True:
        helpers.open_graph_from_base64(curve_base64[0])
        helpers.open_graph_from_base64(curve_base64[1])


def test_evaluate_classification_model(show_figures, random_forest_out):
    evaluation_out = evaluate_classification_model(
        random_forest_out["model"],
        random_forest_out["X_test_df"],
        random_forest_out["y_test_df"],
        ["accuracy", "precision", "recall", "matthews_corrcoef"],
    )
    scores_df = evaluation_out["scores_df"]
    assert (scores_df["Score"] == 1).all()


def test_permutation_testing_plot(show_figures, random_forest_out, helpers):
    current_out = permutation_testing(
        random_forest_out["model"],
        random_forest_out["X_train_df"],
        random_forest_out["y_train_df"],
        "K-Fold",
        "accuracy",
        100,
        42,
    )
    hist_base64 = permutation_testing_plot(
        score=current_out["score"],
        permutation_scores=current_out["permutation_scores"],
        pvalue=current_out["pvalue"],
        score_name="Accuracy",
    )
    if show_figures:
        helpers.open_graph_from_base64(hist_base64[0])


def test_random_sampling(classification_df, meta_df):
    current_out = random_sampling(
        input_df=classification_df,
        metadata_df=meta_df,
        labels_column="Group",
        n_samples=5,
    )
    input_df_len = len(current_out["input_df"])
    labels_df_len = len(current_out["labels_df"])
    assert (
        input_df_len == labels_df_len
    ), f"There is a dimension mismatch between the input dataframe={input_df_len} and the labels={labels_df_len} dataframe"
    assert (
        input_df_len == 5
    ), f"The input dataframe should be reduced to a subset of 5 random samples, but only {input_df_len} were found."
    assert (
        labels_df_len == 5
    ), f"The labels dataframe should be reduced to a subset of 5 random samples, but only {labels_df_len} were found."


def test_generate_stratified_subsets(classification_df, meta_df):
    classification_df = long_to_wide(classification_df)
    train_sizes = [4, 6, 9]
    X_subsets, y_subsets = generate_stratified_subsets(
        input_df=classification_df,
        labels_df=meta_df.set_index("Sample"),
        train_sizes=train_sizes,
        random_state=6,
    )
    label_counts9 = y_subsets[0]["Group"].value_counts()
    label_counts4 = y_subsets[1]["Group"].value_counts()
    label_counts6 = y_subsets[2]["Group"].value_counts()
    assert label_counts9.values.tolist() == [5, 4]
    assert label_counts4.values.tolist() == [2, 2]
    assert label_counts6.values.tolist() == [3, 3]
