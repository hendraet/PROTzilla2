import concurrent
import logging

import numpy as np
import pandas as pd
from django.contrib import messages
from scipy import stats

from .differential_expression_helper import apply_multiple_testing_correction
from ..utilities import chunks, flatten


def t_test(
        intensity_df,
        metadata_df,
        grouping,
        group1,
        group2,
        multiple_testing_correction_method,
        alpha,
        fc_threshold,
        log_base,
):
    """
    A function to conduct a two sample t-test between groups defined in the
    clinical data. The t-test is conducted on the level of each protein.
    The p-values are corrected for multiple testing.

    :param intensity_df: the dataframe that should be tested in long
        format
    :type intensity_df: pandas DataFrame
    :param metadata_df: the dataframe that contains the clinical data
    :type grouping: pandas DataFrame
    :param grouping: the column name of the grouping variable in the
        metadata_df
    :type grouping: str
    :param group1: the name of the first group for the t-test
    :type group1: str
    :param group2: the name of the second group for the t-test
    :type group2: str
    :param multiple_testing_correction_method: the method for multiple
        testing correction
    :type multiple_testing_correction_method: str
    :param alpha: the alpha value for the t-test
    :type alpha: float

    :return: a dict containing a dataframe de_proteins_df in typical protzilla long format containing the differentially expressed proteins,
    a df corrected_p_values, containing the p_values after application of multiple testing correction,
    a df log2_fold_change, containing the log2 fold changes per protein,
    a float fc_threshold, containing the absolute threshold for the log fold change, above which a protein is considered differentially expressed,
    a float corrected_alpha, containing the alpha value after application of multiple testing correction (depending on the selected multiple testing correction method corrected_alpha may be equal to alpha),
    a df filtered_proteins, containing the filtered out proteins (proteins where the mean of a group was 0),
    a df fold_change_df, containing the fold_changes per protein,
    a df t_statistic_df, containing the t-statistic per protein,
    a df significant_proteins_df, containing the proteins where the p-values are smaller than alpha (if fc_threshold = 0, the significant proteins equal the differentially expressed ones)

    :rtype: dict
    """
    assert grouping in metadata_df.columns

    if not group1:
        group1 = metadata_df[grouping].unique()[0]
        logging.warning("auto-selected first group in t-test")
    if not group2:
        group2 = metadata_df[grouping].unique()[1]
        logging.warning("auto-selected second group in t-test")

    proteins = intensity_df.loc[:, "Protein ID"].unique().tolist()
    intensity_name = intensity_df.columns.values.tolist()[3]
    intensity_df = pd.merge(
        left=intensity_df,
        right=metadata_df[["Sample", grouping]],
        on="Sample",
        copy=False,
    )
    p_values = []
    fold_change = []
    log2_fold_change = []
    filtered_proteins = []
    t_statistic = []

    # split into 4 threads
    proteins_chunks = list(chunks(proteins, 8))
    params = (intensity_df, grouping, group1, group2, intensity_name, log_base)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(t_test_worker, chunk, *params)
            for chunk in proteins_chunks
        ]
        success, results = zip(*[f.result() for f in futures])
        results = flatten(results)

    if not all(success):
        msg = "There are Proteins with NaN values present in your data. \
                        Please filter them out before running the differential expression analysis."
        return dict(
            de_proteins_df=None,
            corrected_p_values=None,
            log2_fold_change=None,
            fc_threshold=None,
            corrected_alpha=None,
            messages=[dict(level=messages.ERROR, msg=msg)],
        )
    for result, protein in zip(results, proteins):
        if result["filtered"]:
            filtered_proteins.append(protein)
            continue
        p_values.append(result["p"])
        t_statistic.append(result["t"])

        fold_change.append(result["fc"])
        log2_fold_change.append(np.log2(result["fc"]))

    (corrected_p_values, corrected_alpha) = apply_multiple_testing_correction(
        p_values=p_values,
        method=multiple_testing_correction_method,
        alpha=alpha,
    )

    p_values_mask = corrected_p_values < corrected_alpha
    fold_change_mask = np.abs(log2_fold_change) > fc_threshold

    remaining_proteins = [
        protein for protein in proteins if protein not in filtered_proteins
    ]

    de_proteins = [
        protein
        for protein, has_p, has_fc in zip(
            remaining_proteins, p_values_mask, fold_change_mask
        )
        if has_p and has_fc
    ]
    de_proteins_df = intensity_df.loc[intensity_df["Protein ID"].isin(de_proteins)]

    significant_proteins = [
        protein for i, protein in enumerate(remaining_proteins) if p_values_mask[i]
    ]
    significant_proteins_df = intensity_df.loc[
        intensity_df["Protein ID"].isin(significant_proteins)
    ]

    corrected_p_values_df = pd.DataFrame(
        list(zip(proteins, corrected_p_values)),
        columns=["Protein ID", "corrected_p_value"],
    )

    log2_fold_change_df = pd.DataFrame(
        list(zip(proteins, log2_fold_change)),
        columns=["Protein ID", "log2_fold_change"],
    )
    fold_change_df = pd.DataFrame(
        list(zip(proteins, fold_change)),
        columns=["Protein ID", "fold_change"],
    )

    t_statistic_df = pd.DataFrame(
        list(zip(proteins, t_statistic)),
        columns=["Protein ID", "t_statistic"],
    )

    proteins_filtered = len(filtered_proteins) > 0
    proteins_filtered_warning_msg = f"Some proteins were filtered out because they had a mean intensity of 0 in one of the groups."

    return dict(
        de_proteins_df=de_proteins_df,
        corrected_p_values_df=corrected_p_values_df,
        log2_fold_change_df=log2_fold_change_df,
        fc_threshold=fc_threshold,
        corrected_alpha=corrected_alpha,
        filtered_proteins=filtered_proteins,
        fold_change_df=fold_change_df,
        t_statistic_df=t_statistic_df,
        significant_proteins_df=significant_proteins_df,
        messages=[dict(level=messages.WARNING, msg=proteins_filtered_warning_msg)]
        if proteins_filtered
        else [],
    )


def t_test_worker(
        proteins_chunk, intensity_df, grouping, group1, group2, intensity_name, log_base
):
    """
    :returns: a Tuple of a boolean if successfully executed t-test and a list of dicts with results
     of t-test
    """
    results = []
    for protein in proteins_chunk:
        protein_df = intensity_df.loc[intensity_df["Protein ID"] == protein]

        group1_intensities = protein_df.loc[
            protein_df.loc[:, grouping] == group1, intensity_name
        ].to_numpy()
        group2_intensities = protein_df.loc[
            protein_df.loc[:, grouping] == group2, intensity_name
        ].to_numpy()

        # if a protein has a NaN value in a sample, user should remove it
        group1_is_nan = np.isnan(group1_intensities)
        group2_is_nan = np.isnan(group2_intensities)
        if group1_is_nan.any() or group2_is_nan.any():
            return False, None
        # if the intensity of a group for a protein is 0, it should be filtered out
        if np.mean(group1_intensities) == 0 or np.mean(group2_intensities) == 0:
            results.append(dict(filtered=True))
            continue

        t, p = stats.ttest_ind(group1_intensities, group2_intensities)
        if log_base == "":
            fc = np.mean(group2_intensities) / np.mean(group1_intensities)
        else:
            fc = log_base ** (np.mean(group2_intensities) - np.mean(group1_intensities))
        results.append(dict(t=t, p=p, fc=fc, filtered=False))

    return True, results
