"""
pyspark_utilities.py

This module contains reusable PySpark utility functions commonly
used in data pipelines,
such as removing duplicates and filling null values in DataFrames.
"""

from pyspark.sql.functions import col, to_json, explode_outer,lit, sha2, concat_ws, expr, when
from pyspark.sql.types import StructType, ArrayType
from typing import List, Dict, Union
from pyspark.sql import DataFrame

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

# Function3: flatten nested json dynamically including Struct and ArrayType (Optionally)

def flatten_json(df, explode_arrays=True):
    """
    Recursively flattens a nested DataFrame.
    
    :param df: Input PySpark DataFrame
    :param explode_arrays: If True, explode arrays into rows; if False, keep arrays as JSON strings.
    :return: Flattened PySpark DataFrame
    """
    while True:
        complex_fields = [
            (field.name, field.dataType)
            for field in df.schema.fields
            if isinstance(field.dataType, (StructType, ArrayType))
        ]
        
        if not complex_fields:
            break
        
        for col_name, col_type in complex_fields:
            if isinstance(col_type, StructType):
                # Expand struct into separate columns
                for subfield in col_type.fields:
                    new_col_name = f"{col_name}_{subfield.name}"
                    df = df.withColumn(new_col_name, col(f"{col_name}.{subfield.name}"))
                df = df.drop(col_name)
            
            elif isinstance(col_type, ArrayType):
                if explode_arrays:
                    # Explode arrays into multiple rows
                    df = df.withColumn(col_name, explode_outer(col(col_name)))
                else:
                    # Keep arrays as JSON strings
                    df = df.withColumn(col_name, to_json(col(col_name)))
    return df

# Function4: Generic PySpark Data Masking Function

def mask_dataframe(
    df: DataFrame,
    columns: Union[List[str], Dict[str, str]],
    default_mask: str = "****"
) -> DataFrame:
    """
    Mask sensitive data in given columns of a PySpark DataFrame.

    Args:
        df: Input PySpark DataFrame
        columns: 
            - If list, applies default masking to these columns.
            - If dict, specify {col: mask_type}, where mask_type ∈ {"full", "partial", "hash", "custom_expr"}.
        default_mask: Mask string for "full" masking.
        
    Returns:
        Masked PySpark DataFrame
    """

    if isinstance(columns, list):
        columns = {c: "full" for c in columns}

    masked_df = df

    for col_name, mask_type in columns.items():
        if col_name not in df.columns:
            continue

        if mask_type == "full":
            masked_df = masked_df.withColumn(col_name, lit(default_mask))

        elif mask_type == "partial":
            # Show first 2 chars, mask rest
            masked_df = masked_df.withColumn(
                col_name,
                when(col(col_name).isNotNull(),
                     expr(f"concat(substr({col_name}, 1, 2), repeat('*', greatest(length({col_name})-2,0)))")
                ).otherwise(lit(None))
            )

        elif mask_type == "hash":
            # Hash using SHA2-256
            masked_df = masked_df.withColumn(col_name, sha2(col(col_name).cast("string"), 256))

        elif mask_type.startswith("expr:"):
            # Custom SQL expression: e.g., {"phone": "expr:concat('XXX', substr(phone, -4, 4))"}
            custom_expr = mask_type.split("expr:", 1)[1]
            masked_df = masked_df.withColumn(col_name, expr(custom_expr))

        else:
            raise ValueError(f"Unsupported mask type '{mask_type}' for column '{col_name}'")

    return masked_df
