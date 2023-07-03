import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.svm import SVC
from joblib import Parallel, delayed
from protzilla.constants.logging import logger

from protzilla.data_analysis.classification_helper import (
    perform_cross_validation,
    encode_labels,
    perform_nested_cross_validation,
)
from protzilla.utilities.transform_dfs import is_long_format, long_to_wide

estimator_mapping = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
}


def compute_learning_curve(
    clf_str,
    input_df,
    metadata_df: pd.DataFrame,
    labels_column: str,
    positive_label: str,
    train_sizes,
    cross_validation_strategy,
    scoring,
    random_state,
    shuffle="yes",
    nested_cv_params=None,
    n_jobs=1,
    **cv_params,
):
    input_df_wide = long_to_wide(input_df) if is_long_format(input_df) else input_df
    input_df_wide.sort_values(by="Sample", inplace=True)
    labels_df = (
        metadata_df[["Sample", labels_column]]
        .set_index("Sample")
        .sort_values(by="Sample")
    )
    common_indices = input_df_wide.index.intersection(labels_df.index)
    labels_df = labels_df.loc[common_indices]
    encoding_mapping, labels_df = encode_labels(
        labels_df, labels_column, positive_label
    )
    clf = estimator_mapping[clf_str]

    if "Nested" in cross_validation_strategy:
        if shuffle == "yes":
            random_indices = np.random.RandomState(random_state).permutation(
                input_df_wide.index
            )
            input_df_wide = input_df_wide.loc[random_indices]
            labels_df = labels_df.loc[random_indices]

        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(perform_nested_cross_validation)(
                input_df=input_df_wide.iloc[:size],
                labels_df=labels_df["Encoded Label"].iloc[:size],
                clf=clf,
                scoring=scoring,
                random_state=random_state,
                **cv_params,
            )
            for size in train_sizes
        )
        test_scores, train_scores = zip(*results)
        test_scores = np.array(test_scores)
        train_scores = np.array(train_scores)

    else:
        cv_callable = perform_cross_validation(
            cross_validation_strategy, random_state_cv=random_state, **cv_params
        )
        _, train_scores, test_scores = learning_curve(
            clf,
            input_df_wide,
            labels_df["Encoded Label"],
            train_sizes=train_sizes,
            scoring=scoring,
            cv=cv_callable,
            random_state=random_state,
        )

    # create df train_sizes, train_scores, test_scores
    kneedle = KneeLocator(
        train_sizes,
        test_scores.mean(axis=1),
        S=1.0,
        curve="concave",
        direction="increasing",
    )
    minimum_viable_sample_size = round(kneedle.elbow)

    return dict(
        train_sizes=train_sizes,
        test_scores=test_scores,
        train_scores=train_scores,
        minimum_viable_sample_size=minimum_viable_sample_size,
    )


def random_sampling(input_df, metadata_df, labels_column, n_samples, random_state=6):
    # prepare X and y dataframes for classification
    input_df_wide = long_to_wide(input_df) if is_long_format(input_df) else input_df
    input_df_wide.sort_values(by="Sample", inplace=True)
    labels_df = (
        metadata_df[["Sample", labels_column]]
        .set_index("Sample")
        .sort_values(by="Sample")
    )
    common_indices = input_df_wide.index.intersection(labels_df.index)
    labels_df = labels_df.loc[common_indices]
    input_df_n_samples, _, labels_df_n_samples, _ = train_test_split(
        input_df_wide, labels_df, train_size=n_samples, random_state=random_state
    )
    return dict(input_df=input_df_n_samples, labels_df=labels_df_n_samples)
