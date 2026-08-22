import os
import sys
import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
import argparse
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.data_preprocessing import create_spark_session, load_data, preprocess_data


def train_and_evaluate(spark, data_path="data/iris.csv", model_output="models/iris_model", max_iter=10, reg_param=0.0):

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run() as run:
        df = load_data(spark, data_path)
        transformed_df = preprocess_data(df)

        train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

        lr = LogisticRegression(featuresCol="features", labelCol="indexed_label", maxIter=max_iter, regParam=reg_param)
        model = lr.fit(train_df)

        mlflow.log_param("maxIter", max_iter)
        mlflow.log_param("regParam", reg_param)
        mlflow.log_param("model_type", "LogisticRegression")

        predictions = model.transform(test_df)

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="indexed_label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator_acc.evaluate(predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        evaluator_prec = MulticlassClassificationEvaluator(
            labelCol="indexed_label", predictionCol="prediction", metricName="weightedPrecision"
        )
        precision_weighted = evaluator_prec.evaluate(predictions)
        print(f"Weighted Precision: {precision_weighted:.4f}")
        mlflow.log_metric("weighted_precision", precision_weighted)

        evaluator_rec = MulticlassClassificationEvaluator(
            labelCol="indexed_label", predictionCol="prediction", metricName="weightedRecall"
        )
        recall_weighted = evaluator_rec.evaluate(predictions)
        print(f"Weighted Recall: {recall_weighted:.4f}")
        mlflow.log_metric("weighted_recall", recall_weighted)

        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="indexed_label", predictionCol="prediction", metricName="f1"
        )
        f1_weighted = evaluator_f1.evaluate(predictions)
        print(f"F1 Score: {f1_weighted:.4f}")
        mlflow.log_metric("f1_score", f1_weighted)

        pred_pd = predictions.select("prediction", "indexed_label").toPandas()
        y_true = pred_pd["indexed_label"].astype(int).values
        y_pred = pred_pd["prediction"].astype(int).values

        # Determinar las clases presentes
        classes = sorted(set(y_true) | set(y_pred))
        n_classes = len(classes)

        # Calcular matriz de confusión manualmente
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            confusion_matrix[t, p] += 1

        print("Confusion Matrix:")
        print(confusion_matrix)

        np.save("confusion_matrix.npy", confusion_matrix)
        mlflow.log_artifact("confusion_matrix.npy")

        epsilon = 1e-10
        for cls in classes:
            tp = confusion_matrix[cls, cls]
            fp = confusion_matrix[:, cls].sum() - tp
            fn = confusion_matrix[cls, :].sum() - tp
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            print(f"Class {cls} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            mlflow.log_metric(f"precision_class_{cls}", precision)
            mlflow.log_metric(f"recall_class_{cls}", recall)
            mlflow.log_metric(f"f1_class_{cls}", f1)

        if not os.path.exists(model_output):
            os.makedirs(model_output, exist_ok=True)
        model.write().overwrite().save(model_output)

        mlflow.spark.log_model(
            model,
            "iris_model",
            pip_requirements=[
                f"pyspark=={pyspark.__version__}",
                "numpy==1.26.4",
                "pandas==2.3.3",
                "scikit-learn==1.9.0",
                "mlflow==3.15.1"
            ]
        )

        return model, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador Iris")
    parser.add_argument("--maxIter", type=int, default=10, help="Número máximo de iteraciones")
    parser.add_argument("--regParam", type=float, default=0.0, help="Parámetro de regularización")
    parser.add_argument("--data_path", type=str, default="data/iris.csv", help="Ruta al dataset")
    args = parser.parse_args()

    spark = create_spark_session("TrainIrisModel")
    model, acc = train_and_evaluate(
        spark,
        data_path=args.data_path,
        max_iter=args.maxIter,
        reg_param=args.regParam
    )
    spark.stop()