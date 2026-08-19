from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.data_preprocessing import create_spark_session, load_data, preprocess_data
import os

def train_and_evaluate(spark, data_path="data/iris.csv", model_output="models/iris_model"):
    df = load_data(spark, data_path)
    transformed_df = preprocess_data(df)

    train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="indexed_label", maxIter=10)
    model = lr.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexed_label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    if not os.path.exists(model_output):
        os.makedirs(model_output, exist_ok=True)
    model.write().overwrite().save(model_output)

    return model, accuracy

if __name__ == "__main__":
    spark = create_spark_session("TrainIrisModel")
    model, acc = train_and_evaluate(spark)
    spark.stop()