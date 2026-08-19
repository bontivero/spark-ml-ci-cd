from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
import uvicorn
from src.predict import load_model, predict_single
import os
import sys

# Configurar el Python que usarán Spark y sus workers
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

app = FastAPI(title="Iris Classifier API")

spark = SparkSession.builder.appName("IrisAPI").master("local[*]").getOrCreate()
model = load_model(spark)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Iris classifier API. Use /predict with POST."}

@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        input_list = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]
        pred = predict_single(model, input_list)
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        return {"prediction": pred, "species": species_map[pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)