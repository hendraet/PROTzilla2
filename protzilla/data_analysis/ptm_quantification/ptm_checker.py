import logging
import re

import pandas as pd


def check_ptm_quantification(
    peptide_df: pd.DataFrame,
    diff_modified: pd.DataFrame,
    protein_id: str,
    included_modifications: list[str],
):
    if "Modifications" not in peptide_df.columns:
        return dict(
            messages=[
                {
                    "level": logging.ERROR,
                    "msg": """No 'Modifications' column found in peptide_df. Either your file is not in the correct format or data about modifications is missing. 
                Make sure to use the evidence import if you want to use this method.""",
                }
            ],
        )

    diff_modified.drop(diff_modified.columns[1], axis=1, inplace=True)

    # Creating a dataframe with the expected and found modifications
    result_df = pd.DataFrame(
        columns=["Sample", "Sequence", "Modification found", "Modification expected"]
    )

    peptides = diff_modified.columns[1:]
    protein_df = peptide_df[peptide_df["Protein ID"] == protein_id]
    for _, row in diff_modified.iterrows():
        sample = row[0]
        sample_df = protein_df[protein_df["Sample"] == sample]
        for idx, peptide in enumerate(peptides):

            def clean_peptide_string(s):
                # Remove content within parentheses including the parentheses
                s = re.sub(r"\([^)]*\)", "", s)
                # Remove all underscores and digits
                s = re.sub(r"[_\d]", "", s)
                return s

            found = row[idx + 1]

            included = set(included_modifications + ["Unmodified"])

            def has_more(sequence):
                sequence_set = set(sequence.split(","))
                return len(sequence_set - included) > 0

            df = sample_df[sample_df["Sequence"] == clean_peptide_string(peptide)]
            expected = (df["Modifications"].apply(has_more)).any()

            result_df = result_df.append(
                {
                    "Sample": sample,
                    "Sequence": peptide,
                    "Modification found": found,
                    "Modification expected": expected,
                },
                ignore_index=True,
            )

    # Calculating True Positives, False Positives, True Negatives, and False Negatives
    TP = (
        (result_df["Modification found"] == True)
        & (result_df["Modification expected"] == True)
    ).sum()
    FP = (
        (result_df["Modification found"] == True)
        & (result_df["Modification expected"] == False)
    ).sum()
    TN = (
        (result_df["Modification found"] == False)
        & (result_df["Modification expected"] == False)
    ).sum()
    FN = (
        (result_df["Modification found"] == False)
        & (result_df["Modification expected"] == True)
    ).sum()

    # Calculating accuracy, sensitivity, and specificity
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
    specificity = TN / (TN + FP) if TN + FP != 0 else 0

    return dict(
        ptm_coverage=result_df,
        true_positives=result_df[
            (result_df["Modification found"] == True)
            & (result_df["Modification expected"] == True)
        ],
        false_positives=result_df[
            (result_df["Modification found"] == True)
            & (result_df["Modification expected"] == False)
        ],
        true_negatives=result_df[
            (result_df["Modification found"] == False)
            & (result_df["Modification expected"] == False)
        ],
        false_negatives=result_df[
            (result_df["Modification found"] == False)
            & (result_df["Modification expected"] == True)
        ],
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        messages=[
            {
                "level": logging.INFO,
                "msg": f"Checked {len(result_df)} peptides. Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}",
            }
        ],
    )
