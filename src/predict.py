from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

def load_model(spark, model_path="models/iris_model"):
    return LogisticRegressionModel.load(model_path)

def predict_single(model, features):
    """
    Realiza una predicción usando el modelo cargado.
    Recibe una lista de 4 floats y devuelve la clase predicha (0,1,2).
    """
    vector = Vectors.dense(features)
    prediction = model.predict(vector)  # método directo, sin workers
    return int(prediction)