import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DataType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.schema_validator import (
    ColumnExistsRule,
    ColumnTypeRule,
    DataFrameValidator,
    NoExtraColumnsRule,
    NotValidSchemaException,
    SchemaDefinition,
    ValidationResult,
    ValidationRule,
)


@pytest.fixture
def spark():
    return SparkSession.builder.appName("test").getOrCreate()


@pytest.fixture
def valid_validation_rule_mock():
    class MockRulePassing(ValidationRule):
        def validate(self, df: DataFrame):
            return ValidationResult(is_valid=True, message="Mock rule passed")

    return MockRulePassing()


@pytest.fixture
def invalid_validation_rule_mock():
    class MockRuleFailing(ValidationRule):
        def validate(self, df: DataFrame):
            return ValidationResult(is_valid=False, message="Mock rule failed")

    return MockRuleFailing()


class TestSchemaDefinition:
    def test_add_rule(self, valid_validation_rule_mock: ValidationRule):
        schema = SchemaDefinition()
        schema.add_rule(valid_validation_rule_mock)
        assert len(schema.rules) == 1
        assert schema.rules[0] == valid_validation_rule_mock

    def test_rules_property(self, valid_validation_rule_mock: ValidationRule):
        schema = SchemaDefinition()
        schema.add_rule(valid_validation_rule_mock)
        rules = schema.rules
        assert isinstance(rules, list)
        assert rules[0] == valid_validation_rule_mock


class TestDataFrameValidator:
    def test_validation_passed(
        self, spark: SparkSession, valid_validation_rule_mock: ValidationRule
    ):
        schema = SchemaDefinition()
        schema.add_rule(valid_validation_rule_mock)
        df = spark.createDataFrame([], StructType([]))
        validator = DataFrameValidator(schema)
        report = validator.validate(df, stop_on_first_error=True)
        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_validation_failed(
        self, spark: SparkSession, invalid_validation_rule_mock: ValidationRule
    ):
        schema = SchemaDefinition()
        schema.add_rule(invalid_validation_rule_mock)
        df = spark.createDataFrame([], StructType([]))
        validator = DataFrameValidator(schema)
        report = validator.validate(df, stop_on_first_error=True)
        assert report.is_valid is False
        errors = report.errors
        assert len(errors) == 1
        assert errors[0].message == "Mock rule failed"

    def test_strict_validate_raises_exception_on_failure(
        self, spark: SparkSession, invalid_validation_rule_mock: ValidationRule
    ):
        schema = SchemaDefinition()
        schema.add_rule(invalid_validation_rule_mock)
        df = spark.createDataFrame([], StructType([]))
        validator = DataFrameValidator(schema)

        with pytest.raises(NotValidSchemaException):
            validator.strict_validate(df, stop_on_first_error=True)


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    schema = StructType(
        [
            StructField("existing_column", StringType(), True),
            StructField("another_existing_column", StringType(), True),
        ]
    )
    data = [("1", "value1"), ("2", "value2")]
    return spark.createDataFrame(data, schema)


class TestColumnRules:
    @pytest.mark.parametrize(
        "column_name, expected",
        [
            ("existing_column", True),
            ("missing_column", False),
        ],
    )
    def test_column_exists_rule(self, df: DataFrame, column_name: str, expected: bool):
        rule = ColumnExistsRule(column_name)
        result = rule.validate(df)
        assert result.is_valid == expected

    @pytest.mark.parametrize(
        "column_name, expected_type, expected",
        [
            ("existing_column", StringType(), True),
            ("existing_column", IntegerType(), False),
            ("missing_column", StringType(), False),
        ],
    )
    def test_column_type_rule(
        self,
        df: DataFrame,
        column_name: str,
        expected_type: DataType,
        expected: bool,
    ):
        rule = ColumnTypeRule(column_name, expected_type)
        result = rule.validate(df)
        assert result.is_valid == expected

    @pytest.mark.parametrize(
        "allowed_columns, expected",
        [
            (["existing_column", "another_existing_column"], True),
            (["existing_column"], False),
        ],
    )
    def test_no_extra_columns_rule(
        self, df: DataFrame, allowed_columns: list, expected: bool
    ):
        rule = NoExtraColumnsRule(allowed_columns=allowed_columns)
        result = rule.validate(df)
        assert result.is_valid is expected


class TestEndToEndSchemaValidation:
    def test_end_to_end_schema_validation(self, spark: SparkSession):
        schema = (
            SchemaDefinition()
            .add_rule(ColumnExistsRule("id"))
            .add_rule(ColumnTypeRule("id", LongType()))
            .add_rule(NoExtraColumnsRule(allowed_columns=["id", "name"]))
        )
        data = [(1, "Alice"), (2, "Bob")]
        df = spark.createDataFrame(data, ["id", "name"])

        validator = DataFrameValidator(schema)
        report = validator.validate(df, stop_on_first_error=False)

        assert report.is_valid is True
        assert len(report.errors) == 0
