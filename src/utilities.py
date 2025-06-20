"""
pyspark_utilities.py

This module contains reusable PySpark utility functions commonly
used in data pipelines,
such as removing duplicates and filling null values in DataFrames.
"""


# Function1: Remove duplicates in a PySpark DataFrame
def remove_duplicates(df, subset_cols=None):
    """
    Removes duplicate rows from the DataFrame.

    :param df: Input DataFrame
    :param subset_cols: List of columns to check for duplicates.
    If None, uses all columns.
    :return: Deduplicated DataFrame
    """
    return df.drop_duplicates(subset=subset_cols)

# Further enhancement scope: add functionality to provide ordering columns and
# keep first or last (use row number window function)


# Function2: Fill Null Values in a PySpark DataFrame
def fill_nulls(df, fill_dict):
    """
    Fills null values based on a provided mapping.

    :param df: Input DataFrame
    :param fill_map: Dictionary with column names as keys and fill
    values as values.
    :return: DataFrame with nulls filled
    """
    return df.fillna(fill_dict)
