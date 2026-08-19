import pytest
from pyspark.sql import SparkSession
from src.data_preprocessing import load_data, preprocess_data

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("TestPreprocessing").getOrCreate()

def test_load_data(spark):
    df = load_data(spark, "data/iris.csv")
    assert df.count() == 150
    assert "label" in df.columns

def test_preprocess_data(spark):
    df = load_data(spark, "data/iris.csv")
    transformed = preprocess_data(df)
    assert "features" in transformed.columns
    assert "indexed_label" in transformed.columns
    assert transformed.count() == 150