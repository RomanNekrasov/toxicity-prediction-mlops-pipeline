import os
from flask import Flask, jsonify, request
import sys
import logging
from component import *
from requests import post


# creating the app
app = Flask(__name__)

project_id = os.environ.get('PROJECT_ID')
model_repo = os.environ.get('MODEL_REPO')
clean_api_url = os.environ.get('CLEAN_API_URL')

# For testing locally
# model_repo = "/Users/huubvandevoort/Desktop/Data-Engineering/DataEngineering/development/model"
# clean_api_url = "http://127.0.0.1:5000"
# os.environ['PORT'] = "5001"

# A decorator used to tell the application which URL is associated function
# the complete URL will be http://ip:port/users?name=some_value
@app.route('/', methods=["POST"])
def predict_instance():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if request.method == 'POST':
        api_input_text = request.json.get('textarea')
        if not api_input_text:
            return jsonify(error="Please provide a 'text' field in the request body."), 400
        else:
            clean_response = post(url=clean_api_url, json={'text':api_input_text})
            clean_text = clean_response.json()['clean_text']
            logging.info('...Received cleaned text...')
            logging.info('...Making predictions now...')
            prediction_result = predict_multilabel_classifier(project_id, clean_text, model_repo,
                                                              metrics_path='', validation_data=False)
            return prediction_result


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
