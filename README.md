[![Continuous Integration](https://github.com/nearform/pyspark-common-utilities/actions/workflows/ci.yml/badge.svg)](https://github.com/nearform/pyspark-common-utilities/actions/workflows/ci.yml)

# PYSPARK-COMMON-UTILITIES

### PYSPARK-COMMON-UTILITIES are reusable components which are small functionalities that are commonly part of data pipelines. We are creating functions for these that can be used when required.

## Data Validation

A simple and flexible set of utilities for validating PySpark DataFrame schemas through composable validation rules.

### Overview

This library provides a declarative way to define and enforce schema requirements on PySpark DataFrames. Instead of writing manual checks scattered throughout your code, you can define validation rules once and reuse them across your data pipeline.

### Key Features

- **Composable Rules**: Build complex schema validations by chaining simple rules together
- **Clear Reporting**: Get detailed validation reports with pass/fail status for each rule
- **Flexible Validation**: Choose between soft validation (returns report) or strict validation (raises exception)
- **Early Exit Option**: Stop validation on first error or run all checks to see all issues at once

### Quick Start

```python

# Create your DataFrame
data = [(1, "Alice"), (2, "Bob")]
df = spark.createDataFrame(data, ["id", "name"])

# Define your schema validation rules
schema = (
    SchemaDefinition()
    .add_rule(ColumnExistsRule("id"))
    .add_rule(ColumnTypeRule("id", LongType()))
    .add_rule(NoExtraColumnsRule(allowed_columns=["id", "name"]))
)

# Validate
validator = DataFrameValidator(schema)
report = validator.validate(df, stop_on_first_error=False)

print(report)

# Output:
# Validation Report - VALID
# ============================================
# ✓ - Column id exists
# ✓ - Correct type: LongType()
# ✓ - All columns are allowed
#
# Total: 3 checks - 0 errors
```

### Available Rules

- **ColumnExistsRule**: Checks if a column exists in the DataFrame
- **ColumnTypeRule**: Validates that a column has the expected data type
- **NoExtraColumnsRule**: Ensures no unexpected columns are present

### Validation Methods

#### `validate(df, stop_on_first_error=False)`
Returns a detailed validation report with all results.
If `stop_on_first_error=True`, validation halts on the first failure.

#### `strict_validate(df, stop_on_first_error=True)`
Raises `NotValidSchemaException` if validation fails, useful for enforcing schema requirements in production pipelines.

```python
try:
    validator.strict_validate(df)
except NotValidSchemaException as e:
    print(e)
```


[![banner](https://raw.githubusercontent.com/nearform/.github/refs/heads/master/assets/os-banner-green.svg)](https://www.nearform.com/contact/?utm_source=open-source&utm_medium=banner&utm_campaign=os-project-pages)
