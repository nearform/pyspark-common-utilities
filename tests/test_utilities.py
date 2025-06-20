from pyspark.sql import SparkSession
from src.utilities import remove_duplicates, fill_nulls

spark = SparkSession.builder.master("local[*]").appName("Test").getOrCreate()


def test_remove_duplicates():
    df = spark.createDataFrame([
        ("Alice", "NY"), ("Alice", "NY"), ("Bob", "LA")
    ], ["name", "city"])
    result = remove_duplicates(df)
    assert result.count() == 2


def test_fill_nulls():
    df = spark.createDataFrame([
        (None, "NY"), ("Bob", None)
    ], ["name", "city"])
    result = fill_nulls(df, {"name": "NA", "city": "Unknown City"})
    rows = result.collect()
    assert rows[0]["name"] == "NA"
    assert rows[1]["city"] == "Unknown City"
