from pathlib import Path

import numpy as np
import pandas as pd
import re
from django.contrib import messages
from collections import defaultdict

from protzilla.utilities import clean_uniprot_id
from protzilla.data_integration.database_query import biomart_query


def max_quant_import(_, file_path, intensity_name):
    assert intensity_name in ["Intensity", "iBAQ", "LFQ intensity"]
    if not Path(file_path).is_file():
        msg = "The file upload is empty. Please provide a Max Quant file."
        return None, dict(
            messages=[dict(level=messages.ERROR, msg=msg)],
        )
    selected_columns = ["Protein IDs", "Gene names"]
    read = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
        na_values=["", 0],
        keep_default_na=True,
    )
    df = read.drop(columns=["Intensity", "iBAQ", "iBAQ peptides"], errors="ignore")
    # df["Protein IDs"] = map_groups_to_uniprot(df["Protein IDs"].tolist())
    # df["Protein IDs"] = df["Protein IDs"].map(handle_protein_ids)
    # df = df[df["Protein IDs"].map(bool)]  # remove rows without valid protein id
    if "Gene names" not in df.columns:  # genes column should be removed eventually
        df["Gene names"] = np.nan
    id_df = df[selected_columns]
    id_df = id_df.rename(columns={"Protein IDs": "Protein ID", "Gene names": "Gene"})
    intensity_df = df.filter(regex=f"^{intensity_name} ", axis=1)

    if intensity_df.empty:
        msg = f"{intensity_name} was not found in the provided file, please use another intensity and try again"
        return None, dict(
            messages=[dict(level=messages.ERROR, msg=msg)],
        )
    intensity_df.columns = [c[len(intensity_name) + 1 :] for c in intensity_df.columns]

    df = pd.concat([id_df, intensity_df], axis=1)
    # sum intensities if id appears multiple times
    # df = df.groupby(["Protein IDs"]).sum(numeric_only=True).reset_index()

    # molten = pd.melt(
    #     df,
    #     id_vars=selected_columns,
    #     var_name="Sample",
    #     value_name=intensity_name,
    # )
    # ordered = molten[["Sample", "Protein ID", "Gene", intensity_name]]
    # ordered.sort_values(by=["Sample", "Protein ID"], ignore_index=True, inplace=True)
    # return ordered, {}
    return handle_df(df, intensity_name)


def ms_fragger_import(_, file_path, intensity_name):
    assert intensity_name in [
        "Intensity",
        "MaxLFQ Total Intensity",
        "MaxLFQ Intensity",
        "Total Intensity",
        "MaxLFQ Unique Intensity",
        "Unique Spectral Count",
        "Unique Intensity",
        "Spectral Count",
        "Total Spectral Count",
    ]
    if not Path(file_path).is_file():
        msg = "The file upload is empty. Please provide a MS Fragger file."
        return None, dict(
            messages=[dict(level=messages.ERROR, msg=msg)],
        )
    selected_columns = ["Protein ID", "Gene"]
    read = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
        na_values=["", 0],
        keep_default_na=True,
    )
    df = read.drop(
        columns=[
            "Combined Spectral Count",
            "Combined Unique Spectral Count",
            "Combined Total Spectral Count",
        ]
    )
    df["Protein ID"] = df["Protein ID"].map(handle_protein_ids)
    df = df[df["Protein ID"].map(bool)]  # remove rows without valid protein id
    id_df = df[selected_columns]
    intensity_df = df.filter(regex=f"{intensity_name}$", axis=1)
    intensity_df.columns = [
        c[: -(len(intensity_name) + 1)] for c in intensity_df.columns
    ]
    intensity_df = intensity_df.drop(
        columns=intensity_df.filter(
            regex="MaxLFQ Total$|MaxLFQ$|Total$|MaxLFQ Unique$|Unique$", axis=1
        ).columns
    )
    molten = pd.melt(
        pd.concat([id_df, intensity_df], axis=1),
        id_vars=selected_columns,
        var_name="Sample",
        value_name=intensity_name,
    )
    ordered = molten[["Sample", "Protein ID", "Gene", intensity_name]]
    ordered.sort_values(by=["Sample", "Protein ID"], ignore_index=True, inplace=True)
    return ordered, {}


def handle_df(df, intensity_name):
    non_contaminant_groups = df["Protein ID"].map(
        lambda group: not any(id_.startswith("CON__") for id_ in group.split(";"))
    )
    df = df[non_contaminant_groups]

    # REV__ and XXX__ proteins get removed here as well
    df["Protein ID"] = map_groups_to_uniprot(df["Protein ID"].tolist())

    df = df[df["Protein ID"].map(bool)]  # remove rows without valid protein id

    df = df.groupby(["Protein ID", "Gene"]).sum().reset_index()
    molten = pd.melt(
        df,
        id_vars=["Protein ID", "Gene"],
        var_name="Sample",
        value_name=intensity_name,
    )

    ordered = molten[["Sample", "Protein ID", "Gene", intensity_name]]
    ordered.sort_values(by=["Sample", "Protein ID"], ignore_index=True, inplace=True)

    return ordered, {}


def map_ids(extracted_ids):
    id_to_uniprot = defaultdict(list)
    all_count = 0
    for identifier, matching_ids in extracted_ids.items():
        all_count += len(matching_ids)
        if not matching_ids:
            continue
        result = list(
            biomart_query(
                matching_ids,
                identifier,
                [identifier, "uniprotswissprot"],
            )
        )
        for query, swiss in result:
            if swiss:
                id_to_uniprot[query].append(swiss)

        # we trust reviewed results more, so we don't look up ids we found in swiss
        # in trembl again
        left = matching_ids - set(id_to_uniprot.keys())
        result = list(
            biomart_query(
                left,
                identifier,
                [identifier, "uniprotsptrembl"],
            )
        )
        for query, trembl in result:
            if trembl:
                id_to_uniprot[query].append(trembl)
    print(len(id_to_uniprot), all_count)
    return dict(id_to_uniprot)


def map_groups_to_uniprot(protein_groups):
    regex = {
        "ensembl_peptide_id": re.compile(r"^ENSP\d{11}"),
        "refseq_peptide": re.compile(r"^NP_\d{6,}"),
        "refseq_peptide_predicted": re.compile(r"^XP_\d{9}"),
    }
    uniprot = re.compile(
        r"""^[A-Z]               # start with capital letter
        [A-Z\d]{5}([A-Z\d]{4})?  # match ids of length 6 or 10
        ([-_][-\d]+)?            # match variations like -8 and _9-6
        """,
        re.VERBOSE,
    )
    removed_protein_ids = []

    extracted_ids = {k: set() for k in regex.keys()}
    found_ids_per_group = []
    for group in protein_groups:
        found_in_group = set()
        for protein_id in group.split(";"):
            if match := uniprot.search(protein_id):
                found_in_group.add(match.group(0))
                continue
            for identifier, pattern in regex.items():
                if match := pattern.search(protein_id):
                    found_id = match.group(0)
                    extracted_ids[identifier].add(found_id)
                    found_in_group.add(found_id)
                    break  # can only match one regex
            else:
                removed_protein_ids.append(protein_id)
        found_ids_per_group.append(found_in_group)

    id_to_uniprot = map_ids(extracted_ids)
    new_groups = []

    for group in found_ids_per_group:
        all_ids_of_group = set()
        for old_id in group:
            if uniprot.search(old_id):
                all_ids_of_group.add(old_id)
            else:
                new_ids = id_to_uniprot.get(old_id, [])
                all_ids_of_group.update(new_ids)
        new_groups.append(";".join(sorted(all_ids_of_group)))
    return new_groups


if __name__ == "__main__":
    df, _ = max_quant_import(
        None,
        # "/Users/fynnkroeger/Desktop/Studium/Bachelorprojekt/inputs/not-uniprot-maxquant.txt",
        "/Users/fynnkroeger/Desktop/Studium/Bachelorprojekt/inputs/proteinGroups_small.txt",
        "Intensity",
    )

    # df.to_csv("out_new.csv")
    df.to_csv("out_new.csv")
    # old_df = pd.read_csv("out.csv")

    print(df)
