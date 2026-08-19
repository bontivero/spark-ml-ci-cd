import pytest
from pyspark.sql import SparkSession
from src.train_model import train_and_evaluate
from src.predict import load_model, predict_single

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("TestPrediction").getOrCreate()

def test_prediction_on_sample(spark):
    # Entrenar modelo temporal
    _, _ = train_and_evaluate(spark, model_output="/tmp/test_model_pred")
    model = load_model(spark, "/tmp/test_model_pred")
    # Cargar datos de prueba (CSV con filas conocidas)
    test_df = spark.read.csv("data/test_data.csv", header=True, inferSchema=True)
    # Tomar primera fila
    row = test_df.collect()[0]
    features = [row["sepal_length"], row["sepal_width"], row["petal_length"], row["petal_width"]]
    pred = predict_single(model, features)
    # La primera fila es setosa -> índice 0
    assert pred == 0