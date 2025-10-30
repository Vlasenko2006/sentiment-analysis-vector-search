#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:24:13 2025

@author: andreyvlasenko
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified version without Spark for testing sentiment analysis
Created on Thu Oct 30 09:03:20 2025

@author: andreyvlasenko
"""

import os
import requests
import zipfile
from transformers import pipeline
import shutil
import pandas as pd
from download_and_prepare_dataset import download_and_prepare_dataset

path_db = download_and_prepare_dataset()

print('path_db = ', path_db)

# Set HuggingFace cache directory for driver
cache_dir = "/tmp/hf_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir

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

print("Model loaded successfully!")

# Test with simple data
test_texts = [
    "I love this!",
    "This is terrible.",
    "The weather is nice today.",
    "I hate waiting in traffic.",
    # Adding neutral examples
    "The meeting is at 3 PM.",
    "This is a blue car.",
    "The report contains 50 pages.",
    "Please send the document.",
    "It is what it is."
]

print("\n=== Testing Sentiment Analysis ===")
for text in test_texts:
    result = pipe(text)
    print(f"Text: '{text}' -> Sentiment: {result[0]['label']} (confidence: {result[0]['score']:.3f})")

# If you want to test with pandas
print("\n=== Testing with Pandas DataFrame ===")
df = pd.DataFrame({"text": test_texts})

def get_sentiment(text):
    result = pipe(text)
    return result[0]['label']

df['sentiment'] = df['text'].apply(get_sentiment)
print(df)

print("\nScript completed successfully!")