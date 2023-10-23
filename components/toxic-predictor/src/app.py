# importing Flask and other modules
import os

from flask import Flask, jsonify, request
from component import *
from requests import post

# creating the app
app = Flask(__name__)

project_id = os.environ.get('PROJECT_ID')
model_repo = os.environ.get('MODEL_REPO')
clean_api_url = os.environ.get('CLEAN_API')
METRICS_PATH = ''
VALIDATION_DATA = False


# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def predict_instance():
    if request.method == 'POST':
        api_input_text = request.text.get('textarea')
        if not api_input_text:
            return jsonify(error="Please provide a 'text' field in the request body."), 400
        else:
            clean_response = post(clean_api_url, data={'text':api_input_text})
            clean_text = clean_response.json.get('clean_text')
            prediction_result = predict_multilabel_classifier(project_id, clean_text, model_repo, METRICS_PATH, VALIDATION_DATA)
            return prediction_result


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
