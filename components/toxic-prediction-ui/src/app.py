from flask import Flask, request, render_template
from requests import post
import os

predictor_api_url = os.environ['PREDICTOR_API_URL']

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('templates/input-form-page.html')

    if request.method == 'POST':
        input_text = request.form['message']
        prediction_response = post(url=predictor_api_url, json={'textarea': input_text})
        return render_template('templates/reponse-page.html', prediction_response=prediction_response)
