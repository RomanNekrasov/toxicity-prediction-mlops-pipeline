from flask import Flask, request, render_template
from requests import post
import os

predictor_api_url = os.environ.get('PREDICTOR_API_URL')

# For testing locally
# predictor_api_url = "http://127.0.0.1:5001"
# os.environ['PORT'] = "5002"

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("input-form-page.html")
        # Do not write /templates/input-form-page.html because flask already knows its in templates

    if request.method == 'POST':
        input_text = request.form['message']
        prediction_response = post(url=predictor_api_url, json={'textarea': input_text})
        return render_template("response-page.html", prediction_response=prediction_response.json(),
                               str=str, float=float)
        # Do not write /templates/response-page.html because flask already knows its in templates


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))