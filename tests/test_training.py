import pytest
from pyspark.sql import SparkSession
from src.train_model import train_and_evaluate

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("TestTraining").getOrCreate()

def test_train_accuracy(spark):
    _, accuracy = train_and_evaluate(spark, model_output="/tmp/test_model")
    assert accuracy >= 0.9