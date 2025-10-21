from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType


class NotValidSchemaException(Exception):
    pass


@dataclass(frozen=True)
class ValidationResult:
    """Represents the result of a single validation check."""

    is_valid: bool
    message: str

    def __str__(self):
        status = "✓" if self.is_valid else "✗"
        return f"{status} - {self.message}"


@dataclass(frozen=True)
class SchemaValidationReport:
    """Aggregates the results of multiple validation checks."""

    results: List[ValidationResult]

    @property
    def is_valid(self) -> bool:
        """True if all validation checks passed."""

        return all(r.is_valid for r in self.results)

    @property
    def errors(self) -> List[ValidationResult]:
        """List of failed validation checks."""

        return [r for r in self.results if not r.is_valid]

    def __str__(self):
        """
        Format:

        Validation Report - VALID [/INVALID]
        ============================================
        ✓ - Column A exists
        ✗ - Column B missing
        ...
        Total: X checks - Y errors
        """

        header = (
            f"Validation Report - {'VALID' if self.is_valid else 'INVALID'}"
        )
        divider = "=" * 60
        results_str = "\n".join(str(r) for r in self.results)
        summary = (
            f"\nTotal: {len(self.results)} checks - {len(self.errors)} errors"
        )
        return f"{header}\n{divider}\n{results_str}{summary}"


class ValidationRule(ABC):
    """Abstract base class for all validation rules."""

    @abstractmethod
    def validate(self, df: DataFrame) -> ValidationResult:
        """Validate the given DataFrame against this rule."""
        pass


class ColumnRelatedRule(ValidationRule, ABC):
    """Abstract base class for column-related validation rules."""

    def __init__(self, column_name: str):
        self._column_name = column_name


class ColumnExistsRule(ColumnRelatedRule):
    """Validation rule to check if a column exists in the DataFrame."""

    def validate(self, df: DataFrame) -> ValidationResult:
        """
        Validation message:
        Passed -> Column {column_name} exists
        Failed -> Column {column_name} missing
        """

        col = self._column_name
        is_valid = col in df.columns
        return ValidationResult(
            is_valid=is_valid,
            message=f"Column {col} exists"
            if is_valid
            else f"Column {col} missing",
        )


class ColumnTypeRule(ColumnRelatedRule):
    """Validation rule to check if a column has the expected data type."""

    def __init__(self, column_name: str, expected_type: DataType):
        super().__init__(column_name=column_name)
        self._expected_type = expected_type

    def validate(self, df: DataFrame) -> ValidationResult:
        """
        Validation string:
        Passed -> Correct type: {actual_type}
        Failed -> Expected {expected_type}, found {actual_type}
        Failed (col missing) -> Column {column_name} missing, cannot check type
        """
        col = self._column_name

        # First, check if the column exists

        def _column_exists() -> bool:
            existent_result = ColumnExistsRule(col).validate(df)
            return existent_result.is_valid

        if not _column_exists():
            return ValidationResult(
                is_valid=False,
                message=f"Column {col} missing, cannot check type",
            )

        # Then, check the column type

        actual_type = df.schema[col].dataType
        is_valid = isinstance(actual_type, type(self._expected_type))

        return ValidationResult(
            is_valid=is_valid,
            message=f"Correct type: {actual_type}"
            if is_valid
            else f"Expected {self._expected_type}, found {actual_type}",
        )


class NoExtraColumnsRule(ValidationRule):
    """
    Validation rule to ensure no extra columns are present in the DataFrame.
    """

    def __init__(self, allowed_columns: List[str]):
        self._allowed_columns = set(allowed_columns)

    def validate(self, df: DataFrame) -> ValidationResult:
        """
        Validation string:
        Passed -> All columns are allowed
        Failed -> Extra columns found: {extra_column_1, extra_column_2, ...}
        """
        extra_columns = set(df.columns) - self._allowed_columns
        is_valid = len(extra_columns) == 0

        return ValidationResult(
            is_valid=is_valid,
            message="All columns are allowed"
            if is_valid
            else f"Extra columns found: {', '.join(extra_columns)}",
        )


class SchemaDefinition:
    """Defines a schema through a collection of validation rules."""

    def __init__(self):
        self._rules: List[ValidationRule] = []

    @property
    def rules(self) -> List[ValidationRule]:
        return self._rules

    def add_rule(self, rule: ValidationRule) -> SchemaDefinition:
        """
        Add a validation rule to the schema definition.

        Example:
        schema = SchemaDefinition()
                .add_rule(ColumnExistsRule("column_a"))
                .add_rule(ColumnTypeRule("column_b", IntegerType()))

        """
        self._rules.append(rule)
        return self


class DataFrameValidator:
    """
    Validates a DataFrame against a defined schema.

    Example:
    df = spark.createDataFrame(...)
    schema = SchemaDefinition()
                .add_rule(ColumnExistsRule("column_a"))
                .add_rule(ColumnTypeRule("column_b", IntegerType()))
    schema = DataFrameValidator(schema)

    print(schema.validate(df))
    schema.strict_validate(df) # raises if not valid

    """

    def __init__(self, schema: SchemaDefinition):
        self._schema = schema

    def validate(
        self,
        df: DataFrame,
        stop_on_first_error: bool = True,
    ) -> SchemaValidationReport:
        """
        Returns a schema validation report.

        If stop_on_first_error is True, validation halts on the first failure.
        """

        results = []
        for rule in self._schema.rules:
            result = rule.validate(df)
            results.append(result)

            if stop_on_first_error and not result.is_valid:
                break
        return SchemaValidationReport(results)

    def strict_validate(self, df: DataFrame, stop_on_first_error: bool = True):
        """Raises if the DataFrame does not conform to the schema"""

        report = self.validate(df, stop_on_first_error=stop_on_first_error)

        if not report.is_valid:
            raise NotValidSchemaException(str(report))
