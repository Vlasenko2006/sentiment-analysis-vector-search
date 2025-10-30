#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:03:20 2025

@author: andreyvlasenko
"""

import os
import requests
import zipfile
from transformers import pipeline
import shutil
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
from download_and_prepare_dataset import download_and_prepare_dataset

path_db = download_and_prepare_dataset()

print('path_db = ', path_db)





# Set HuggingFace cache directory for driver
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

# Set Java options for Spark compatibility with newer Java versions
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
java_opts = "--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
os.environ["SPARK_DRIVER_OPTS"] = java_opts
os.environ["SPARK_EXECUTOR_OPTS"] = java_opts

# Download and save HuggingFace model to the UC Volume (do this only once!)
# First, let's create and save the model
model_path = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"

# Create the model directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# Load and save the model (do this only once!)
if not os.path.exists(os.path.join(model_path, "config.json")):
    print("Loading and saving model...")
    pipe_temp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    pipe_temp.save_pretrained(model_path)
    print("Model saved!")

# Load the model from saved path
pipe = pipeline("sentiment-analysis", model=model_path)



#pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#pipe.save_pretrained(uc_local_model_path)
#print("Model saved at:", model_path)


# === 3. Pandas UDF for inference ===
def get_sentiment_pipe_and_status():
    from transformers import pipeline
    import os, traceback
    model_path_worker = "/Users/andreyvlasenko/tst/Request/my_volume/hf_model"  # Fixed path
    cache_path_worker = "/tmp/hf_cache"
    os.makedirs(cache_path_worker, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_path_worker
    os.environ["HF_HOME"] = cache_path_worker
    try:
        pipe = pipeline("sentiment-analysis", model=model_path_worker)
        return pipe, "ok"
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"load_error: {str(e)} | {tb}"

@F.pandas_udf(StringType())
def get_sentiment_udf(texts):
    import pandas as pd
    import traceback
    pipe, status = get_sentiment_pipe_and_status()
    if pipe is None:
        return pd.Series([f"MODEL_LOAD_ERROR: {status}"] * len(texts))
    results = []
    for t in texts:
        try:
            out = pipe(t)
            print(f"[UDF] Input: {t}, Output: {out}")
            results.append(out[0]['label'])
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[UDF] Exception during inference for input '{t}': {e}\n{tb}")
            results.append(f"INFER_ERROR: {str(e)} | {tb}")
    return pd.Series(results)

# === Initialize Spark Session ===
# Set Java options for compatibility with newer Java versions
java_options = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)

spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.driver.extraJavaOptions", java_options) \
    .config("spark.executor.extraJavaOptions", java_options) \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# === 4. Minimal test DataFrame ===
df = spark.createDataFrame([("I love this!",), ("This is terrible.",)], ["text"])
df_pred = df.withColumn("predicted", get_sentiment_udf(F.col("text")))

df_pred.show(truncate=False)

# Stop Spark session
spark.stop()