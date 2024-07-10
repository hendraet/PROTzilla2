import pandas as pd

from protzilla.utilities import default_intensity_column


def long_to_wide(intensity_df: pd.DataFrame, value_name: str = None):
    """
    This function transforms the dataframe to a wide format that
    can be more easily handled by packages such as sklearn.
    Each sample gets one row with all observations as columns.

    :param intensity_df: the dataframe that should be transformed into
        long format
        :type intensity_df: pd.DataFrame

    :return: returns dataframe in wide format suitable for use by
        packages such as sklearn
    :rtype: pd.DataFrame
    """

    if intensity_df.duplicated(subset=["Sample", "Protein ID"]).any():
        intensity_df = intensity_df.groupby(["Sample", "Protein ID"]).mean().reset_index()
        intensity_df = intensity_df.dropna()

    values_name = default_intensity_column(intensity_df) if value_name is None else value_name
    intensity_df = pd.pivot(
        intensity_df, index="Sample", columns="Protein ID", values=values_name
    )
    intensity_df = intensity_df.fillna(intensity_df.mean())
    return intensity_df


def wide_to_long(wide_df: pd.DataFrame, original_long_df: pd.DataFrame):
    """
    This functions transforms the dataframe from a wide
    format to the typical protzilla long format.

    :param wide_df: the dataframe in wide format that
        should be changed
    :type wide_df: pd.DataFrame
    :param original_long_df: the original long protzilla format
        dataframe, that was the source of the wide format dataframe
    :type orginal_long_df: pd.DataFrame

    :return: returns dataframe in typical protzilla long format
    :rtype: pd.DataFrame
    """
    # Read out info from original dataframe
    intensity_name = default_intensity_column(original_long_df)

    # Identify the additional columns from the original long dataframe
    additional_columns = ['Modification', 'Retention Time']
    existing_additional_columns = [col for col in additional_columns if col in original_long_df.columns]

    # Melt the wide format back to long format
    melted_df = pd.melt(
        wide_df,
        id_vars="Sample",
        var_name="Protein ID",
        value_name=intensity_name,
    )
    melted_df.sort_values(
        by=["Sample", "Protein ID"],
        ignore_index=True,
        inplace=True,
    )

    # Add back the additional columns if they exist in the original dataframe
    for col in existing_additional_columns:
        melted_df[col] = original_long_df[col]

    return melted_df


def is_long_format(df: pd.DataFrame):
    required_columns = {"Sample", "Protein ID"}
    additional_columns = {"Gene", "Retention time"}
    return required_columns.issubset(df.columns) and any(col in df.columns for col in additional_columns)


def is_intensity_df(df: pd.DataFrame):
    """
    Checks if the dataframe is an intensity dataframe.
    An intensity dataframe should have the columns "Sample", "Protein ID" and
    and intensity column.

    :param df: the dataframe that should be checked
    :type df: pd.DataFrame

    :return: returns True if the dataframe is an intensity dataframe
    :rtype: bool
    """
    if not isinstance(df, pd.DataFrame):
        return False

    required_columns = {"Sample", "Protein ID"}
    if not required_columns.issubset(df.columns):
        return False

    intensity_names = [
        "Intensity",
        "iBAQ",
        "LFQ intensity",
        "MaxLFQ Total Intensity",
        "MaxLFQ Intensity",
        "Total Intensity",
        "MaxLFQ Unique Intensity",
        "Unique Spectral Count",
        "Unique Intensity",
        "Spectral Count",
        "Total Spectral Count",
    ]

    for column_name in df.columns:
        if any(intensity_name in column_name for intensity_name in intensity_names):
            return True

    return False
