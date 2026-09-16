import os
import sys
import time
from argparse import ArgumentParser

import mlflow
import mlflow.spark
import numpy as np
import pyspark
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.data_preprocessing import create_spark_session, load_data, preprocess_data


def get_model(model_type, maxIter=10, regParam=0.0, maxDepth=5, numTrees=20):
    """Crea un estimador de Spark ML según el tipo de modelo."""
    if model_type == "logistic":
        return LogisticRegression(featuresCol="features", labelCol="indexed_label", maxIter=maxIter, regParam=regParam)
    elif model_type == "random_forest":
        return RandomForestClassifier(featuresCol="features", labelCol="indexed_label", numTrees=numTrees, maxDepth=maxDepth)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(featuresCol="features", labelCol="indexed_label", maxDepth=maxDepth)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")


def train_and_evaluate(spark, data_path="data/iris.csv", model_output=None, 
                    model_type="logistic", maxIter=10, regParam=0.0, maxDepth=5, numTrees=20):
    """Entrena un modelo, evalúa con métricas y registra en MLflow."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    if model_output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_output = f"models/iris_model_{model_type}_{timestamp}"

    with mlflow.start_run():
        df = load_data(spark, data_path)
        transformed_df = preprocess_data(df)

        train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)

        model = get_model(model_type, maxIter=maxIter, regParam=regParam, maxDepth=maxDepth, numTrees=numTrees)
        model = model.fit(train_df)

        mlflow.log_param("model_type", model_type)
        if model_type == "logistic":
            mlflow.log_param("maxIter", maxIter)
            mlflow.log_param("regParam", regParam)
        elif model_type in ["random_forest", "decision_tree"]:
            mlflow.log_param("maxDepth", maxDepth)
            if model_type == "random_forest":
                mlflow.log_param("numTrees", numTrees)

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
            model_type,
            pip_requirements=[
                f"pyspark=={pyspark.__version__}",
                "numpy==1.26.4",
                "pandas==2.3.3",
                "scikit-learn==1.9.0",
                "mlflow=={mlflow.__version__}"
            ]
        )

        return model, accuracy


if __name__ == "__main__":

    parser = ArgumentParser(description="Entrenamiento de clasificador Iris")
    parser.add_argument("--model_type", type=str, default="logistic",
                        choices=["logistic", "random_forest", "decision_tree"],
                        help="Tipo de modelo a entrenar")
    parser.add_argument("--maxIter", type=int, default=10, help="Número máximo de iteraciones (solo logistic)")
    parser.add_argument("--regParam", type=float, default=0.0, help="Regularización (solo logistic)")
    parser.add_argument("--maxDepth", type=int, default=5, help="Profundidad máxima del árbol (tree/forest)")
    parser.add_argument("--numTrees", type=int, default=20, help="Número de árboles (random forest)")
    args = parser.parse_args()

    spark = create_spark_session("TrainIrisModel")
    model, acc = train_and_evaluate(
        spark,
        model_type=args.model_type,
        maxIter=args.maxIter,
        regParam=args.regParam,
        maxDepth=args.maxDepth,
        numTrees=args.numTrees
    )
    spark.stop()