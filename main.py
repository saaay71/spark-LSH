import os
import sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from LSHPipeline import build_LSH_pipeline
from pyspark.ml import PipelineModel


if __name__ == "__main__":
    # Initialize the spark context.
    spark = SparkSession\
        .builder\
        .appName("LSHPipeline")\
        .getOrCreate()

    INPUT_FILE_PATH = "./data.csv"
    MODEL_PATH = "./saved_model_path"

    df = spark.read.format("csv").option("header",True) .load(INPUT_FILE_PATH)

    df.show(truncate=False)

    LSH_pipeline = build_LSH_pipeline(inputCol="Text", outputCol="hashes", n_ngram=3, seedValue=110, p=5)

    train_df = df.select("Text")
    trained_LSHModel = LSH_pipeline.fit(train_df)
    trained_LSHModel.write().overwrite().save(MODEL_PATH)

    saved_LSHModel = PipelineModel.load(MODEL_PATH)

    df = saved_LSHModel.transform(df)
    df.show(truncate=False)


    spark.stop()
