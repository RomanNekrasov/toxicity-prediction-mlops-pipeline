# importing Flask and other modules
from flask import Flask, jsonify, request, current_app
from component import *
import pickle as pkl
import joblib
from google.cloud import storage


def load_vectorizor():
    # load vectorizer from file
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('models_de2023_group1')
    blob = bucket.blob('vectorizer_model.joblib') # vectorizer.pickle is the name of the file in the bucket
    blob.download_to_filename('/tmp/vectorizer_model.joblib')

    with open('/tmp/vectorizer_model.joblib', 'rb') as f:
        vectorizer = joblib.load(f)
        f.close()
        return vectorizer

def create_app():
    app = Flask(__name__)
    with app.app_context():
        load_vectorizor()
    return app

# creating the app
app = create_app()

# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def clean_data():
    vectorizer = load_vectorizor()
    text = request.json.get('text')
    if not text:
        return jsonify(error="Please provide a 'text' field in the request body."), 400
    cleaned_text = clean_text(text)
    cleaned_data = vectorizer.transform([cleaned_text])
    return jsonify(vector=str(cleaned_data)) # returning the cleaned text in json format


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))