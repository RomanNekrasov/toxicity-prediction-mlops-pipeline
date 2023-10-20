# importing Flask and other modules
from flask import Flask, jsonify, request
from component import *
import pandas as pd
import numpy as np
import pickle as pkl
from google.cloud import storage

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def say_hello():
    text = request.data
    cleaned_text = clean_text(text)
    df = pd.Series(name='comment_text')
    df = df.iloc[len(df)+1] = cleaned_text

    # load vectorizer from file
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('models_de2023_group1')
    blob = bucket.blob('vectorizer.pkl') # vectorizer.pkl is the name of the file in the bucket
    blob.download_to_filename('/tmp/vectorizer.pkl')

    with open('/tmp/vectorizer.pkl', 'rb') as f:
        vectorizer = pkl.load(f)
        df = vectorizer.transform(df)
    return jsonify(text=df['comment_text'][0])

# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)