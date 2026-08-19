from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def create_spark_session(app_name="IrisPreprocessing"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()

def load_data(spark, path="data/iris.csv"):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumnRenamed("sepal_length", "sepal_length") \
        .withColumnRenamed("sepal_width", "sepal_width") \
        .withColumnRenamed("petal_length", "petal_length") \
        .withColumnRenamed("petal_width", "petal_width") \
        .withColumnRenamed("species", "label")
    return df

def build_preprocessing_pipeline():
    label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features"
    )
    return Pipeline(stages=[label_indexer, assembler])

def preprocess_data(df):
    pipeline = build_preprocessing_pipeline()
    model = pipeline.fit(df)
    transformed_df = model.transform(df)
    return transformed_df.select("features", "indexed_label")